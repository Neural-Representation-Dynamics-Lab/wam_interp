"""Small configuration layer independent of Hydra at import time."""

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class RunConfig:
    backend: str = "fastwam"
    task: str = "libero_optional_idm_2cam224_1e-4"
    checkpoint: Path = Path("checkpoints/model.pt")
    observation: Path = Path("observation.npz")
    prompt: str = ""
    condition: str = "first_frame"
    action_horizon: int = 32
    num_video_frames: int = 33
    num_inference_steps: int = 10
    seed: int = 0
    device: str = "cuda"
    dtype: str = "bfloat16"
    output: Path = Path("wam_interp_result.json")
    trace_modules: tuple[str, ...] = ()

    def validate(self) -> None:
        if not self.condition.strip():
            raise ValueError("condition must be a non-empty name")
        if self.action_horizon < 1 or self.num_inference_steps < 1:
            raise ValueError("action_horizon and num_inference_steps must be positive")
        if not self.prompt.strip():
            raise ValueError("A task prompt is required")
