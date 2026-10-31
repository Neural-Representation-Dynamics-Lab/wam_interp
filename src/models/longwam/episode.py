"""Per-episode observation history for the Long-WAM benchmark adapters (upstream runtime/policy.py)."""

from collections import deque
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from wam_interp.types import PreparedObservation

from .adapter import LongWAMAdapter
from .transforms import build_observation_window, field_normalizer, load_dataset_stats, window_indices


def _plain(value: Any) -> Any:
    try:
        from omegaconf import OmegaConf
        if OmegaConf.is_config(value):
            value = OmegaConf.to_container(value, resolve=True)
    except ImportError:
        pass
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    return value


def _raw_dim(meta: Mapping[str, Any]) -> int:
    size = meta.get("raw_shape", meta["shape"])
    return int(size if isinstance(size, int) else size[-1])


class LongWAMEpisodeAdapter(LongWAMAdapter):
    """Benchmark adapter base: checkpoint data config, statistics and history window."""

    benchmark = ""

    def __init__(self, model: Any, *, dataset_stats: Mapping[str, Any] | str | Path,
                 data_config: Mapping[str, Any] | None = None):
        super().__init__(model)
        if data_config is None:
            data_config = (getattr(model, "_wam_interp_metadata", None) or {}).get("data_config")
            if data_config is None:
                raise ValueError("Pass data_config (the bundle's data.train) or load the model with load_longwam")
        self.data_config = _plain(data_config)
        self.dataset_stats = load_dataset_stats(dataset_stats)
        self.processor_config = self.data_config["processor"]
        if self.processor_config.get("action_state_transforms") not in (None, [], {}):
            raise NotImplementedError("Long-WAM adapters implement only action_state_transforms=null checkpoints")
        shape_meta = self.data_config["shape_meta"]
        if len(shape_meta["state"]) != 1 or len(shape_meta["action"]) != 1:
            raise ValueError("Long-WAM benchmark adapters expect one merged state key and one action key")
        self.image_meta = shape_meta["images"]
        self.state_key = str(shape_meta["state"][0]["key"])
        self.action_key = str(shape_meta["action"][0]["key"])
        self.state_dim = _raw_dim(shape_meta["state"][0])
        self.action_dim = _raw_dim(shape_meta["action"][0])
        if self.state_dim != getattr(model, "proprio_dim", self.state_dim):
            raise ValueError(f"Checkpoint proprio_dim {model.proprio_dim} != data state dim {self.state_dim}")
        if self.action_dim != int(model.action_expert.action_dim):
            raise ValueError(f"Checkpoint action_dim {model.action_expert.action_dim} != data action dim {self.action_dim}")
        self.state_normalizer = field_normalizer(self.processor_config, self.dataset_stats, "state", self.state_key)
        self.action_normalizer = field_normalizer(self.processor_config, self.dataset_stats, "action", self.action_key)
        self.video_size = tuple(int(value) for value in self.data_config["video_size"])

        # History contract checks from upstream native_config / RobotWinInference.
        self.past_obs_size = int(self.data_config.get("past_obs_size", 0) or 0)
        self.sample_stride = int(self.data_config.get("action_video_freq_ratio", 1) or 1)
        if self.past_obs_size < 0 or self.sample_stride <= 0 or self.past_obs_size % self.sample_stride:
            raise ValueError("past_obs_size must be a nonnegative multiple of action_video_freq_ratio")
        model_past = getattr(model, "longwam_past_obs_size", self.past_obs_size)
        if int(model_past) != self.past_obs_size:
            raise ValueError(f"Checkpoint model history ({model_past}) and data history ({self.past_obs_size}) disagree")
        factor = int(model.vae.temporal_downsample_factor)
        expected_clean = self.past_obs_size // self.sample_stride // factor + 1
        if int(model.num_clean_frames) != expected_clean:
            raise ValueError(f"History gives {expected_clean} clean latents but the model expects {model.num_clean_frames}")
        self._frames: deque = deque(maxlen=self.past_obs_size + 1)
        self._observed = 0
        self._in_episode = False

    # Benchmark-specific pieces.
    def _model_image(self, observation):
        raise NotImplementedError

    def _raw_state(self, observation):
        raise NotImplementedError

    def _image_metadata(self) -> dict[str, Any]:
        return {}

    def prepare_observation(self, observation) -> PreparedObservation:
        import torch

        image = self._model_image(observation)
        state = self._raw_state(observation)
        if tuple(state.shape) != (self.state_dim,):
            raise ValueError(f"Expected raw state of size {self.state_dim}, got {tuple(state.shape)}")
        proprio = self.state_normalizer.forward(torch.as_tensor(state, dtype=torch.float32).unsqueeze(0))
        return PreparedObservation(
            image=image, proprio=proprio, prompt=observation.instruction,
            metadata={**dict(observation.metadata), **self._image_metadata(),
                      "longwam_image_shape": list(image.shape), "state_key": self.state_key,
                      "dataset_stats": str(self.dataset_stats.get("_source", "in_memory"))},
        )

    def begin_episode(self, observation) -> None:
        """Reset history (upstream ``reset``) and record the first observation."""
        self._frames.clear()
        self._observed = 0
        self._in_episode = True
        self._append(observation)

    def observe_step(self, observation, action) -> None:
        """Record the observation produced by executing ``action`` (every control step)."""
        del action
        if not self._in_episode:
            raise RuntimeError("Call begin_episode() before observe_step()")
        self._append(observation)

    def _append(self, observation) -> None:
        self._frames.append(self._model_image(observation)[0].detach())
        self._observed += 1

    def predict_action(self, request, recorder=None):
        import torch

        if not self._in_episode or not self._frames:
            raise RuntimeError("Long-WAM conditions on observation history; call begin_episode() first")
        current = request.observation.image
        if current.ndim != 4 or not torch.equal(current[0], self._frames[-1]):
            raise ValueError("predict_action must use the latest observed frame; call observe_step() after every "
                             "executed action and prepare the current observation")
        frames = list(self._frames)
        window = build_observation_window(frames, self.past_obs_size, self.sample_stride)
        history = {
            "current_step": self._observed - 1, "observed_frames": self._observed,
            "buffered_frames": len(frames), "past_obs_size": self.past_obs_size,
            "sample_stride": self.sample_stride,
            "padded_frames": max(0, self.past_obs_size + 1 - len(frames)) if self.past_obs_size else 0,
            "window_observation_indices": (window_indices(self._observed, self.past_obs_size, self.sample_stride)
                                           if self.past_obs_size else [self._observed - 1]),
        }
        return self._infer(request, input_image=frames[-1].unsqueeze(0), obs_window=window,
                           recorder=recorder, history=history)

    def _denormalized_actions(self, actions):
        """Upstream ``_denormalize_action``: float32 CPU, normalizer backward, numpy ``[T, D]``."""
        import torch

        values = torch.as_tensor(actions.detach() if hasattr(actions, "detach") else actions)
        if values.ndim == 2:
            values = values.unsqueeze(0)
        if values.ndim != 3 or values.shape[0] != 1 or values.shape[-1] != self.action_dim:
            raise ValueError(f"Expected normalized actions [T,{self.action_dim}], got {tuple(values.shape)}")
        values = values.to(dtype=torch.float32, device="cpu")
        return self.action_normalizer.backward(values).numpy()[0]
