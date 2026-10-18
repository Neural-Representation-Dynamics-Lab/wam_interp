"""Shared model-neutral request, observation, and result types."""

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class EnvironmentObservation:
    """Raw observation from a benchmark frontend, before model-specific transforms."""

    cameras: Mapping[str, Any]
    state: Mapping[str, Any]
    instruction: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PreparedObservation:
    """Model-ready observation; image is CHW float [-1,1], state is model-normalized."""

    image: Any
    proprio: Any | None
    prompt: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class InferenceRequest:
    observation: PreparedObservation
    condition: str = "first_frame"
    action_horizon: int = 32
    num_video_frames: int = 33
    num_inference_steps: int = 10
    seed: int | None = 0
    rand_device: str = "cpu"
    sigma_shift: float | None = None
    negative_prompt: str = ""
    text_cfg_scale: float = 1.0
    tiled: bool = False
    compile_action_infer: bool = False
    trace_modules: tuple[str, ...] = ()


@dataclass
class InferenceResult:
    """Normalized model output plus provenance; environment transforms are separate."""

    actions: Any
    metadata: dict[str, Any] = field(default_factory=dict)
    trace: list[Any] = field(default_factory=list)
