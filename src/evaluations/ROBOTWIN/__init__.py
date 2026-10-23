"""RoboTwin 2.0 evaluator; SAPIEN is imported only when evaluation starts."""

from .config import RoboTwinEvalConfig
from .runner import evaluate_robotwin

__all__ = ["RoboTwinEvalConfig", "evaluate_robotwin"]
