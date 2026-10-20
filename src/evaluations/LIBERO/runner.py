"""Closed-loop LIBERO evaluation around an already-loaded model adapter."""

from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
import json
from typing import Any, Mapping

from wam_interp.models.base import EvaluationModelAdapter

from wam_interp.types import InferenceRequest, InferenceResult

from .config import LiberoEvalConfig
from .environment import (
    create_libero_environment,
    get_benchmark_module,
    initial_states_for_task,
    LIBERO_DUMMY_ACTION,
    reset_to_initial_state,
    step_libero,
)
from .results import EpisodeResult, LiberoEvaluationResult, TaskResult


def _get_suite(name: str) -> Any:
    benchmark, _, _ = get_benchmark_module()
    suites = benchmark.get_benchmark_dict()
    try:
        return suites[name]()
    except KeyError as exc:
        raise ValueError(
            f"LIBERO does not provide suite {name!r}; available suites: {sorted(suites)}"
        ) from exc


def _episode_seed(base_seed: int | None, task_id: int, trial_index: int) -> int | None:
    if base_seed is None:
        return None
    # Stable arithmetic avoids Python's process-randomized hash() behavior.
    return int(base_seed) + int(task_id) * 1_000_003 + int(trial_index)


def _seed_environment(env: Any, seed: int | None) -> None:
    if seed is None:
        return
    seed_fn = getattr(env, "seed", None)
    if callable(seed_fn):
        seed_fn(int(seed))


def _max_steps(config: LiberoEvalConfig) -> int:
    if config.max_steps is not None:
        return config.max_steps
    suite_name = config.task_suite_name.strip().lower()
    return 700 if suite_name in {"libero_10", "libero_90"} else 400


def _to_numpy(value: Any):
    import numpy as np

    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value)


def _all_finite(value: Any) -> bool:
    import numpy as np

    return bool(np.isfinite(value).all())


def _json_safe(value: Any) -> Any:
    if is_dataclass(value):
        return _json_safe(asdict(value))
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_safe(item) for item in value]
    if hasattr(value, "detach") or hasattr(value, "tolist"):
        try:
            converted = value.detach().cpu() if hasattr(value, "detach") else value
            return _json_safe(converted.tolist())
        except (TypeError, ValueError, RuntimeError):
            return repr(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


def _request_for_call(
    config: LiberoEvalConfig,
    model: EvaluationModelAdapter,
    prepared_observation: Any,
    episode_seed: int | None,
    inference_call: int,
) -> InferenceRequest:
    options = dict(model.inference_defaults())
    options.setdefault("condition", "first_frame")
    options.update(config.inference_options)
    options.setdefault(
        "seed",
        None if episode_seed is None else int(episode_seed) + int(inference_call),
    )
    return InferenceRequest(
        observation=prepared_observation,
        action_horizon=config.action_horizon,
        **options,
    )


def _run_episode(
    *,
    model: EvaluationModelAdapter,
    env: Any,
    task: Any,
    task_id: int,
    trial_index: int,
    initial_state_index: int,
    initial_state: Any,
    config: LiberoEvalConfig,
    seed: int | None,
    max_steps: int,
) -> EpisodeResult:
    instruction = str(task.language)
    provenance = {"task_id": task_id, "trial_index": trial_index}
    _seed_environment(env, seed)
    observation = reset_to_initial_state(
        env,
        initial_state,
        instruction=instruction,
        metadata=provenance,
    )

    warmup_steps = 0
    total_reward = 0.0
    success = False
    done = False
    termination_reason = "max_steps"

    for _ in range(config.num_steps_wait):
        warmup = step_libero(
            env,
            LIBERO_DUMMY_ACTION,
            instruction=instruction,
            metadata=provenance,
        )
        observation = warmup.observation
        total_reward += warmup.reward
        warmup_steps += 1
        success, done = warmup.success, warmup.done
        if success or done:
            termination_reason = "success_during_warmup" if success else "environment_done_during_warmup"
            break

    policy_steps = 0
    inference_calls = 0
    inference_metadata: list[dict[str, Any]] = []
    begin_episode = getattr(model, "begin_episode", None)
    observe_step = getattr(model, "observe_step", None)
    if callable(begin_episode) and not (success or done):
        begin_episode(observation)

    while not success and not done and policy_steps < max_steps:
        prepared = model.prepare_observation(observation)
        request = _request_for_call(config, model, prepared, seed, inference_calls)
        prediction = model.predict_action(request)
        if not isinstance(prediction, InferenceResult):
            raise TypeError(
                "EvaluationModelAdapter.predict_action() must return InferenceResult; "
                f"got {type(prediction).__name__}"
            )
        inference_calls += 1
        call_metadata = _json_safe(prediction.metadata)
        if not isinstance(call_metadata, dict):
            call_metadata = {"model_metadata": call_metadata}
        if prediction.trace:
            call_metadata["trace"] = _json_safe(prediction.trace)
        inference_metadata.append(call_metadata)

        decoded = _to_numpy(
            model.postprocess_actions(prediction.actions, environment="libero")
        )
        if decoded.ndim == 3 and decoded.shape[0] == 1:
            decoded = decoded[0]
        if decoded.ndim != 2 or decoded.shape[1] != 7:
            raise ValueError(
                "LIBERO adapter postprocess_actions must return [time, 7] commands; "
                f"got shape {tuple(decoded.shape)}"
            )
        if decoded.shape[0] == 0:
            raise ValueError("The model adapter returned an empty action chunk")
        if not _all_finite(decoded):
            raise ValueError("The model adapter returned NaN or infinite LIBERO actions")

        execute_count = min(
            config.replan_steps,
            decoded.shape[0],
            max_steps - policy_steps,
        )
        for action in decoded[:execute_count]:
            step = step_libero(
                env,
                action,
                instruction=instruction,
                metadata=provenance,
            )
            observation = step.observation
            total_reward += step.reward
            policy_steps += 1
            if callable(observe_step):
                observe_step(observation, action)
            success, done = step.success, step.done
            if success or done:
                termination_reason = "success" if success else "environment_done"
                break

    if success and termination_reason == "max_steps":
        termination_reason = "success"

    return EpisodeResult(
        task_id=task_id,
        trial_index=trial_index,
        initial_state_index=initial_state_index,
        seed=seed,
        success=bool(success),
        termination_reason=termination_reason,
        policy_steps=policy_steps,
        warmup_steps=warmup_steps,
        inference_calls=inference_calls,
        total_reward=total_reward,
        inference_metadata=inference_metadata,
    )


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
    for task_id in task_ids:
        task = suite.get_task(int(task_id))
        initial_states = initial_states_for_task(suite, int(task_id))
        if len(initial_states) == 0:
            raise ValueError(f"LIBERO returned no initial states for task {task_id}")
        env_seed = _episode_seed(config.seed, int(task_id), 0)
        env = create_libero_environment(task, config.camera_resolution, env_seed)
        episodes: list[EpisodeResult] = []
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
        finally:
            close = getattr(env, "close", None)
            if callable(close):
                close()
        task_results.append(
            TaskResult(
                task_id=int(task_id),
                task_name=str(getattr(task, "name", f"task_{task_id}")),
                instruction=str(task.language),
                episodes=episodes,
            )
        )

    result = LiberoEvaluationResult(
        task_suite_name=suite_name,
        model_name=str(getattr(model, "name", type(model).__name__)),
        started_at=started_at,
        duration_seconds=perf_counter() - started_clock,
        config=_json_safe(config),
        tasks=task_results,
    )
    if output_dir is not None:
        result_path = output_dir / f"libero_{suite_name}_{run_id}.json"
        result.output_path = str(result_path)
        result_path.write_text(
            json.dumps(result.to_dict(), indent=2, sort_keys=True),
            encoding="utf-8",
        )
    return result
