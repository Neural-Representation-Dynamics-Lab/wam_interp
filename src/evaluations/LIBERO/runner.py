"""Closed-loop LIBERO evaluation around an already-loaded model adapter."""

from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any

from wam_interp.models.base import EvaluationModelAdapter

from .config import LiberoEvalConfig
from .environment import (
    create_libero_environment,
    get_benchmark_module,
    initial_states_for_task,
    reset_to_initial_state,
    step_libero,
)
from .results import EpisodeResult, LiberoEvaluationResult, TaskResult
from .._libero_rollout import (
    run_libero_episode, _episode_seed, _max_steps, _json_safe, _request_for_call,
)
from ..provenance import model_provenance, package_versions, write_report


def _get_suite(name: str) -> Any:
    benchmark, _, _ = get_benchmark_module()
    suites = benchmark.get_benchmark_dict()
    try:
        return suites[name]()
    except KeyError as exc:
        raise ValueError(
            f"LIBERO does not provide suite {name!r}; available suites: {sorted(suites)}"
        ) from exc


def _run_episode(**kwargs):
    """Keep frontend callbacks local for simulator injection and existing callers."""
    return run_libero_episode(reset_fn=reset_to_initial_state, step_fn=step_libero, **kwargs)


def evaluate_libero(
    model: EvaluationModelAdapter,
    config: LiberoEvalConfig | None = None,
) -> LiberoEvaluationResult:
    """Run LIBERO episodes using an already-created model adapter.

    This function never loads a model or checkpoint. ``model`` supplies the
    backend-specific observation preparation, inference, and action decoding.
    """
    config = config or LiberoEvalConfig()
    config.validate()
    required_methods = (
        "inference_defaults",
        "prepare_observation",
        "predict_action",
        "postprocess_actions",
    )
    missing_methods = [
        name for name in required_methods if not callable(getattr(model, name, None))
    ]
    if missing_methods:
        raise TypeError(
            "model is missing evaluation adapter methods: "
            f"{missing_methods}"
        )

    started_clock = perf_counter()
    started_at = datetime.now(timezone.utc).isoformat()
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    suite_name = config.task_suite_name.strip().lower()
    suite = _get_suite(suite_name)
    suite_size = getattr(suite, "n_tasks", None)
    if callable(suite_size):
        suite_size = suite_size()
    if suite_size is None:
        suite_size = len(suite.tasks)
    task_ids = tuple(range(int(suite_size))) if config.task_ids is None else config.task_ids
    invalid = [task_id for task_id in task_ids if task_id >= int(suite_size)]
    if invalid:
        raise ValueError(
            f"Task IDs {invalid} are outside {suite_name} with {suite_size} tasks"
        )

    output_dir = None if config.output_dir is None else Path(config.output_dir).expanduser().resolve()
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)

    task_results: list[TaskResult] = []
    max_steps = _max_steps(config)
    result = LiberoEvaluationResult(
        task_suite_name=suite_name,
        model_name=str(getattr(model, "name", type(model).__name__)),
        started_at=started_at,
        duration_seconds=0.0,
        config=_json_safe(config),
        tasks=task_results,
        partial_selection=config.task_ids is not None and len(task_ids) != int(suite_size),
        provenance={"model": model_provenance(model), "package_versions": package_versions()},
    )
    result_path = None if output_dir is None else output_dir / f"libero_{suite_name}_{run_id}.json"
    result.output_path = None if result_path is None else str(result_path)

    def flush():
        # Rewritten after every episode, so a crash or walltime kill keeps finished episodes.
        result.duration_seconds = perf_counter() - started_clock
        if result_path is not None:
            write_report(result.to_dict(), result_path)

    try:
        for task_id in task_ids:
            task = suite.get_task(int(task_id))
            initial_states = initial_states_for_task(suite, int(task_id))
            if len(initial_states) == 0:
                raise ValueError(f"LIBERO returned no initial states for task {task_id}")
            env_seed = _episode_seed(config.seed, int(task_id), 0)
            env = create_libero_environment(task, config.camera_resolution, env_seed)
            episodes: list[EpisodeResult] = []
            task_results.append(TaskResult(
                task_id=int(task_id),
                task_name=str(getattr(task, "name", f"task_{task_id}")),
                instruction=str(task.language),
                episodes=episodes,
            ))
            try:
                for trial_index in range(config.num_trials):
                    state_index = trial_index % len(initial_states)
                    seed = _episode_seed(config.seed, int(task_id), trial_index)
                    episode = _run_episode(
                        model=model,
                        env=env,
                        task=task,
                        task_id=int(task_id),
                        trial_index=trial_index,
                        initial_state_index=state_index,
                        initial_state=initial_states[state_index],
                        config=config,
                        seed=seed,
                        max_steps=max_steps,
                    )
                    episodes.append(episode)
                    flush()
            finally:
                close = getattr(env, "close", None)
                if callable(close):
                    close()
        result.completed = True
    except BaseException as exc:
        result.error = {"type": type(exc).__name__, "message": str(exc)}
        raise
    finally:
        flush()
    return result
