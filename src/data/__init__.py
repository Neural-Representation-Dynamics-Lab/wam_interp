"""Input-data adapters for model-ready observations."""

from .npz import load_observation_npz, prepare_observation

__all__ = ["load_observation_npz", "prepare_observation"]

