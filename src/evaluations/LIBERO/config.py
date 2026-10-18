"""Configuration for live LIBERO rollouts; model loading is intentionally absent."""

from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, Mapping

from wam_interp.types import InferenceRequest


_VALID_SUITES = {
    "libero_spatial",
    "libero_object",
    "libero_goal",
    "libero_10",
    "libero_90",
}
_REQUEST_FIELDS = {
    item.name for item in fields(InferenceRequest)
} - {"observation", "action_horizon"}


@dataclass(frozen=True)
class LiberoEvalConfig:
    """Episode protocol plus shared inference options forwarded to the adapter.

    task_ids=None evaluates every task in the suite. ``inference_options``
    configures shared InferenceRequest fields; backend-specific validity is
    checked by that backend. ``run_metadata`` records caller-supplied details
    such as checkpoint identity and external asset revisions.
    """

    task_suite_name: str = "libero_spatial"
    task_ids: tuple[int, ...] | None = (0,)
    num_trials: int = 50
    max_steps: int | None = None
    num_steps_wait: int = 30
    replan_steps: int = 10
    action_horizon: int = 32
    seed: int | None = 0
    camera_resolution: int = 256
    inference_options: Mapping[str, Any] = field(default_factory=dict)
    run_metadata: Mapping[str, Any] = field(default_factory=dict)
    output_dir: Path | None = None
    record_actions: bool = False
    save_rollout_videos: bool = False

    def validate(self) -> None:
        suite = self.task_suite_name.strip().lower()
        if suite not in _VALID_SUITES:
            raise ValueError(
                f"Unknown LIBERO suite {self.task_suite_name!r}; "
                f"choose from {sorted(_VALID_SUITES)}"
            )
        if self.num_trials < 1:
            raise ValueError("num_trials must be at least 1")
        if self.max_steps is not None and self.max_steps < 1:
            raise ValueError("max_steps must be positive when provided")
        if self.num_steps_wait < 0:
            raise ValueError("num_steps_wait cannot be negative")
        if self.replan_steps < 1 or self.action_horizon < 1:
            raise ValueError("replan_steps and action_horizon must be positive")
        if self.camera_resolution < 1:
            raise ValueError("camera_resolution must be positive")
        if self.save_rollout_videos and self.output_dir is None:
            raise ValueError("output_dir is required when save_rollout_videos=True")
        if self.task_ids is not None:
            if not self.task_ids:
                raise ValueError("task_ids cannot be empty; use None for the full suite")
            if any(not isinstance(task_id, int) or task_id < 0 for task_id in self.task_ids):
                raise ValueError("task_ids must contain non-negative integers")
            if len(set(self.task_ids)) != len(self.task_ids):
                raise ValueError("task_ids must not contain duplicates")

        unknown = set(self.inference_options) - _REQUEST_FIELDS
        if unknown:
            raise ValueError(
                f"Unknown inference_options {sorted(unknown)}; "
                f"allowed shared request fields are {sorted(_REQUEST_FIELDS)}"
            )