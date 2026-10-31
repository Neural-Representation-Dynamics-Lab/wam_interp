from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, Mapping

from wam_interp.types import InferenceRequest

ROBOTWIN_REVISION = "c3ddfa8b97d5519efa828b075999bd0006778e5e"


@dataclass(frozen=True)
class RoboTwinEvalConfig:
    task_name: str
    robotwin_root: Path
    task_config: str = "demo_clean"
    num_episodes: int = 1
    seed: int = 0
    start_seed: int | None = None
    max_seed_attempts: int = 1000
    max_repeated_screen_errors: int = 20   # identical consecutive screening exceptions => systematic failure
    clear_cache_freq: int = 5              # upstream task_config default
    max_steps: int | None = None
    action_horizon: int = 32
    replan_steps: int = 28
    instruction_type: str = "unseen"
    instruction_pool_size: int = 100
    inference_options: Mapping[str, Any] = field(default_factory=dict)
    run_metadata: Mapping[str, Any] = field(default_factory=dict)
    output_dir: Path | None = None

    def validate(self):
        if not self.task_name or not self.task_name.replace("_", "").isalnum():
            raise ValueError("task_name must be an upstream task identifier")
        if self.task_config not in {"demo_clean", "demo_randomized"}:
            raise ValueError("task_config must be demo_clean or demo_randomized")
        if self.instruction_type not in {"seen", "unseen"}:
            raise ValueError("instruction_type must be seen or unseen")
        if min(self.num_episodes, self.action_horizon, self.replan_steps, self.max_seed_attempts, self.instruction_pool_size,
               self.max_repeated_screen_errors, self.clear_cache_freq) < 1:
            raise ValueError("Episode counts, action horizons, and attempt limits must be positive")
        if self.max_seed_attempts < self.num_episodes:
            raise ValueError("max_seed_attempts cannot be smaller than num_episodes")
        if self.max_steps is not None and self.max_steps < 1:
            raise ValueError("max_steps must be positive")
        allowed = {item.name for item in fields(InferenceRequest)} - {"observation", "action_horizon"}
        if set(self.inference_options) - allowed:
            raise ValueError(f"Unknown inference options: {sorted(set(self.inference_options) - allowed)}")
