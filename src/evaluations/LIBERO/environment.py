"""LIBERO-specific task, simulator, and raw observation frontend.

This module contains benchmark mechanics only. It does not import a model
backend or resize/normalize observations for a particular model.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from wam_interp.types import EnvironmentObservation


LIBERO_DUMMY_ACTION = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0)


@dataclass(frozen=True)
class LiberoStep:
    observation: EnvironmentObservation
    reward: float
    done: bool
    success: bool
    info: Mapping[str, Any]


def get_benchmark_module():
    """Import the optional upstream simulator package only when evaluation starts."""
    try:
        from libero.libero import benchmark, get_libero_path
        from libero.libero.envs import OffScreenRenderEnv
    except ImportError as exc:
        raise RuntimeError(
            "LIBERO evaluation needs the upstream LIBERO repository and its "
            "MuJoCo/robosuite dependencies installed in this Python environment."
        ) from exc
    return benchmark, get_libero_path, OffScreenRenderEnv


def create_libero_environment(task: Any, resolution: int, seed: int | None):
    benchmark, get_libero_path, offscreen_render_env = get_benchmark_module()
    del benchmark
    bddl_path = (
        Path(get_libero_path("bddl_files"))
        / task.problem_folder
        / task.bddl_file
    )
    if not bddl_path.is_file():
        raise FileNotFoundError(
            f"LIBERO BDDL task file does not exist: {bddl_path}. "
            "Check the upstream LIBERO path configuration."
        )

    env = offscreen_render_env(
        bddl_file_name=str(bddl_path),
        camera_heights=int(resolution),
        camera_widths=int(resolution),
    )
    if seed is not None:
        env.seed(int(seed))
    return env


def observation_from_libero(
    raw: Mapping[str, Any],
    *,
    instruction: str,
    metadata: Mapping[str, Any] | None = None,
) -> EnvironmentObservation:
    """Expose unmodified camera pixels and named raw robot state to any backend."""
    required = (
        "agentview_image",
        "robot0_eye_in_hand_image",
        "robot0_eef_pos",
        "robot0_eef_quat",
        "robot0_gripper_qpos",
    )
    missing = [key for key in required if key not in raw]
    if missing:
        raise KeyError(f"LIBERO observation is missing fields: {missing}")

    return EnvironmentObservation(
        cameras={
            "agentview": raw["agentview_image"],
            "wrist": raw["robot0_eye_in_hand_image"],
        },
        state={
            "eef_position": raw["robot0_eef_pos"],
            "eef_quaternion": raw["robot0_eef_quat"],
            "gripper_position": raw["robot0_gripper_qpos"],
        },
        instruction=instruction,
        metadata=dict(metadata or {}),
    )


def reset_to_initial_state(
    env: Any,
    initial_state: Any,
    *,
    instruction: str,
    metadata: Mapping[str, Any],
) -> EnvironmentObservation:
    env.reset()
    raw = env.set_init_state(initial_state)
    return observation_from_libero(raw, instruction=instruction, metadata=metadata)


def step_libero(
    env: Any,
    action: Any,
    *,
    instruction: str,
    metadata: Mapping[str, Any],
) -> LiberoStep:
    result = env.step(action)
    if not isinstance(result, tuple):
        raise TypeError(f"LIBERO env.step must return a tuple, got {type(result).__name__}")
    if len(result) == 4:
        raw, reward, done, info = result
    elif len(result) == 5:
        raw, reward, terminated, truncated, info = result
        done = bool(terminated) or bool(truncated)
    else:
        raise ValueError(f"Expected 4- or 5-value env.step result, got {len(result)}")

    checker = getattr(env, "check_success", None)
    if callable(checker):
        success = bool(checker())
    else:
        info_success = info.get("success", False) if isinstance(info, Mapping) else False
        success = bool(info_success or float(reward) > 0.0)
    return LiberoStep(
        observation=observation_from_libero(
            raw, instruction=instruction, metadata=metadata
        ),
        reward=float(reward),
        done=bool(done),
        success=success,
        info=info if isinstance(info, Mapping) else {},
    )


def initial_states_for_task(task_suite: Any, task_id: int):
    """Ask LIBERO's suite object for its fixed simulator states for this task."""
    return task_suite.get_task_init_states(int(task_id))
