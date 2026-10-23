"""Evaluate pinned perturbation variants through the shared LIBERO-family loop."""

from time import perf_counter

from .._libero_rollout import run_libero_episode, _episode_seed, _max_steps, _json_safe
from ..LIBERO.environment import create_libero_environment, reset_to_initial_state, step_libero
from ..provenance import package_versions, model_provenance, file_hash, save_report
from .config import LiberoPlusEvalConfig, LIBERO_PLUS_REVISION
from .environment import identify_benchmark, initial_states_for_variant


def select_variants(suite, records, config):
    # Official classification IDs are 1-based labels, not Python suite indices.
    by_name = {item["name"]: item for item in records}
    if len(by_name) != len(records):
        raise ValueError("Duplicate names in task classification")
    requested = tuple(range(int(suite.n_tasks))) if config.task_ids is None else config.task_ids
    known_categories = {record["category"] for record in records}
    if set(config.categories) - known_categories:
        raise ValueError(f"Unknown perturbation categories; choose {sorted(known_categories)}")
    selected = []
    for index in requested:
        if not 0 <= index < int(suite.n_tasks):
            raise ValueError(f"Variant index {index} outside suite")
        task = suite.get_task(index)
        if task.name not in by_name:
            raise ValueError(f"Variant {task.name!r} has no classification record")
        record = by_name[task.name]
        if config.categories and record["category"] not in config.categories:
            continue
        if config.difficulty_levels and record["difficulty_level"] not in config.difficulty_levels:
            continue
        selected.append({"task_id": index, "classification_id": record["id"], "name": task.name,
                         "category": record["category"], "difficulty_level": record["difficulty_level"],
                         "instruction": task.language})
    if not selected:
        raise ValueError("No variants match the task selection and filters")
    return selected


def evaluate_libero_plus(model, config=None):
    config = config or LiberoPlusEvalConfig()
    config.validate()
    for method in ("prepare_observation", "predict_action", "inference_defaults", "postprocess_actions"):
        if not callable(getattr(model, method, None)):
            raise TypeError(f"Evaluation adapter is missing {method}")
    suite, records, root, classification = identify_benchmark(config)
    selected = select_variants(suite, records, config)
    start = perf_counter()
    report = {"benchmark": "libero_plus", "benchmark_revision": LIBERO_PLUS_REVISION,
              "benchmark_root": str(root), "classification_sha256": file_hash(classification),
              "task_suite_name": config.task_suite_name, "suite_variant_count": int(suite.n_tasks),
              "partial_selection": len(selected) != int(suite.n_tasks), "selection": selected,
              "config": _json_safe(config), "model": model_provenance(model),
              "package_versions": package_versions(), "trials": [], "by_category": {}}
    try:
        for variant in selected:
            index = variant["task_id"]
            states = initial_states_for_variant(suite, index)
            if not len(states):
                raise ValueError(f"No initial states for variant {index}")
            env = create_libero_environment(suite.get_task(index), config.camera_resolution,
                                           _episode_seed(config.seed, index, 0))
            try:
                for trial in range(config.num_trials):
                    result = run_libero_episode(model=model, env=env, task=suite.get_task(index),
                        task_id=index, trial_index=trial, initial_state_index=trial % len(states),
                        initial_state=states[trial % len(states)], config=config,
                        seed=_episode_seed(config.seed, index, trial), max_steps=_max_steps(config),
                        reset_fn=reset_to_initial_state, step_fn=step_libero, environment="libero_plus")
                    report["trials"].append({**variant, **_json_safe(result)})
            finally:
                env.close()
    except Exception as exc:
        report["error"] = {"type": type(exc).__name__, "message": str(exc)}
        raise
    finally:
        report["duration_seconds"] = perf_counter() - start
        for trial in report["trials"]:
            group = report["by_category"].setdefault(trial["category"], {"successes": 0, "episodes": 0})
            group["successes"] += int(trial["success"])
            group["episodes"] += 1
        for group in report["by_category"].values():
            group["success_rate"] = group["successes"] / group["episodes"]
        report["episodes"] = len(report["trials"])
        report["successes"] = sum(int(trial["success"]) for trial in report["trials"])
        report["success_rate"] = report["successes"] / report["episodes"] if report["episodes"] else None
        save_report(report, config.output_dir, "libero_plus")
    return report
