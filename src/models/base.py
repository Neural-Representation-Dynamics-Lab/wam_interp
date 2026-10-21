"""Model-neutral adapter contracts."""

from collections.abc import Mapping
from typing import Any, Protocol, runtime_checkable

from wam_interp.types import (
    EnvironmentObservation,
    InferenceRequest,
    InferenceResult,
    PreparedObservation,
)


@runtime_checkable
class ModelAdapter(Protocol):
    name: str

    def predict_action(self, request: InferenceRequest, recorder: Any | None = None) -> InferenceResult:
        """Return normalized actions and metadata for one prepared observation."""
        ...


@runtime_checkable
class EvaluationModelAdapter(ModelAdapter, Protocol):
    """Adapter extensions required by a closed-loop environment evaluator.

    Preprocessing, model-configured request defaults, and action decoding are
    implemented by the backend. The evaluator only orchestrates the benchmark.

    Stateful backends may also define begin_episode(observation), called once
    after warm-up, and observe_step(observation, action), called after every
    executed policy step. Evaluators call them only when present.

    RoboTwin adapters additionally provide environment_action_type(environment),
    returning 'qpos' or 'ee'. LIBERO-family evaluators do not require this hook.
    """

    def inference_defaults(self) -> Mapping[str, Any]:
        """Return backend-configured InferenceRequest defaults."""
        ...

    def prepare_observation(self, observation: EnvironmentObservation) -> PreparedObservation:
        """Convert a benchmark's raw observation into this model's input format."""
        ...

    def postprocess_actions(self, actions: Any, *, environment: str) -> Any:
        """Decode normalized model actions into the selected environment's action format."""
        ...
