"""Framework-light trace collector."""

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class TraceEvent:
    point: str
    data: Any
    metadata: dict[str, Any] = field(default_factory=dict)


class TraceRecorder:
    def __init__(self, *, detach_to_cpu: bool = True):
        self.detach_to_cpu = detach_to_cpu
        self.events: list[TraceEvent] = []

    def _snapshot(self, value: Any):
        if hasattr(value, "detach"):
            value = value.detach()
            if self.detach_to_cpu and hasattr(value, "cpu"):
                value = value.cpu()
            if hasattr(value, "clone"):
                value = value.clone()
            return value
        if isinstance(value, tuple):
            return tuple(self._snapshot(item) for item in value)
        if isinstance(value, list):
            return [self._snapshot(item) for item in value]
        if isinstance(value, dict):
            return {key: self._snapshot(item) for key, item in value.items()}
        return value

    def record(self, point: str, data: Any, **metadata: Any) -> None:
        self.events.append(TraceEvent(point=point, data=self._snapshot(data), metadata=metadata))

    def save(self, path: str | Path) -> None:
        import torch
        torch.save([asdict(event) for event in self.events], Path(path))

