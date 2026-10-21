"""Shared LIBERO-family rollout mechanics, with explicit frontend callbacks."""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Mapping, TYPE_CHECKING
from time import perf_counter

from wam_interp.types import InferenceRequest, InferenceResult

LIBERO_DUMMY_ACTION = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0)

if TYPE_CHECKING:
    from .LIBERO.config import LiberoEvalConfig
    from wam_interp.models.base import EvaluationModelAdapter

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


def run_libero_episode(
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
    reset_fn,
    step_fn,
    environment: str = "libero",
):
    instruction = str(task.language)
    provenance = {"task_id": task_id, "trial_index": trial_index}
    _seed_environment(env, seed)
    observation = reset_fn(
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
        warmup = step_fn(
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
        from .provenance import synchronize_adapter
        synchronize_adapter(model)
        started_inference = perf_counter()
        prediction = model.predict_action(request)
        synchronize_adapter(model)
        inference_latency_ms = (perf_counter() - started_inference) * 1000.0
        if not isinstance(prediction, InferenceResult):
            raise TypeError(
                "EvaluationModelAdapter.predict_action() must return InferenceResult; "
                f"got {type(prediction).__name__}"
            )
        inference_calls += 1
        call_metadata = _json_safe(prediction.metadata)
        if not isinstance(call_metadata, dict):
            call_metadata = {"model_metadata": call_metadata}
        call_metadata["inference_latency_ms"] = inference_latency_ms
        if prediction.trace:
            call_metadata["trace"] = _json_safe(prediction.trace)
        inference_metadata.append(call_metadata)

        decoded = _to_numpy(
            model.postprocess_actions(prediction.actions, environment=environment)
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
            step = step_fn(
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

    from .LIBERO.results import EpisodeResult
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
