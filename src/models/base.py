"""Model-neutral adapter contract."""

from typing import Any, Protocol, runtime_checkable

from wam_interp.types import InferenceRequest, InferenceResult


@runtime_checkable
class ModelAdapter(Protocol):
    name: str

    def predict_action(self, request: InferenceRequest, recorder: Any | None = None) -> InferenceResult:
        """Return normalized actions and metadata for one observation."""
        ...

