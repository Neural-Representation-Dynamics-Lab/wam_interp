"""FastWAM-specific LIBERO observation and action transforms.

The shared evaluator supplies raw LIBERO cameras/state. This adapter mirrors
FastWAM's training/evaluation transforms before inference and decodes normalized
actions into LIBERO's seven-dimensional command convention afterward.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any
import json
import math

import numpy as np

from wam_interp.models.fastwam.adapter import FastWAMAdapter
from wam_interp.types import EnvironmentObservation, PreparedObservation


class FastWAMLiberoAdapter(FastWAMAdapter):
    """An already-loaded FastWAM model plus its LIBERO transforms and stats.

    ``data_config`` is the resolved ``cfg.data.train`` mapping from FastWAM's
    Hydra config. ``dataset_stats`` is the released/training stats JSON (or its
    parsed mapping). This class does not load weights or compose Hydra config.
    """

    def __init__(
        self,
        model: Any,
        *,
        data_config: Mapping[str, Any],
        dataset_stats: Mapping[str, Any] | str | Path,
        binarize_gripper: bool = True,
    ):
        super().__init__(model)
        self.data_config = _plain_mapping(data_config)
        self.dataset_stats = _load_stats(dataset_stats)
        self.binarize_gripper = bool(binarize_gripper)

        processor = self.data_config.get("processor", {})
        self.processor_config = processor
        transforms = processor.get("action_state_transforms")
        if transforms not in (None, [], {}):
            raise NotImplementedError(
                "This adapter currently supports the FastWAM LIBERO config with "
                "processor.action_state_transforms=null. Implement that transform "
                "here before using a config that enables it."
            )
        shape_meta = self.data_config.get("shape_meta", {})
        self.image_meta = shape_meta.get("images", [])
        self.state_meta = shape_meta.get("state", [])
        self.action_meta = shape_meta.get("action", [])
        if len(self.state_meta) != 1 or len(self.action_meta) != 1:
            raise ValueError(
                "FastWAM LIBERO preprocessing expects one merged state key and "
                "one merged action key in data.train.shape_meta."
            )
        self.state_key = str(self.state_meta[0]["key"])
        self.action_key = str(self.action_meta[0]["key"])
        self.num_cameras = int(processor.get("num_output_cameras", len(self.image_meta)))
        if self.num_cameras not in (1, 2) or len(self.image_meta) < self.num_cameras:
            raise ValueError("FastWAM LIBERO adapter supports one or two configured cameras")
        self.concat_multi_camera = str(self.data_config.get("concat_multi_camera", "horizontal"))
        if self.concat_multi_camera not in ("horizontal", "vertical"):
            raise ValueError(
                "data.train.concat_multi_camera must be 'horizontal' or 'vertical'"
            )
        self.video_size = self.data_config.get("video_size")
        self._validate_stats()

    def _validate_stats(self) -> None:
        for kind, key in (("state", self.state_key), ("action", self.action_key)):
            try:
                stats = self.dataset_stats[kind][key]
            except (KeyError, TypeError) as exc:
                raise KeyError(
                    f"Dataset stats must contain [{kind!r}][{key!r}]"
                ) from exc
            if not isinstance(stats, Mapping):
                raise TypeError(f"Stats for {kind}/{key} must be a mapping")

    def inference_defaults(self) -> dict[str, Any]:
        """Derive FastWAM's sampled video-frame count from its training config."""
        num_frames = self.data_config.get("num_frames")
        frequency = self.data_config.get("action_video_freq_ratio")
        if num_frames is None or frequency is None or int(frequency) <= 0:
            return {}
        return {"num_video_frames": (int(num_frames) - 1) // int(frequency) + 1}

    def prepare_observation(
        self, observation: EnvironmentObservation
    ) -> PreparedObservation:
        """Rotate/crop/resize LIBERO cameras, build and normalize proprioception."""
        cameras = observation.cameras
        if "agentview" not in cameras:
            raise KeyError("FastWAM LIBERO preprocessing needs camera 'agentview'")
        camera_names = ["agentview"]
        if self.num_cameras == 2:
            if "wrist" not in cameras:
                raise KeyError("FastWAM's two-camera LIBERO config also needs camera 'wrist'")
            camera_names.append("wrist")

        resized = []
        for index, camera_name in enumerate(camera_names):
            shape = self.image_meta[index].get("shape")
            if shape is None or len(shape) != 3 or int(shape[0]) != 3:
                raise ValueError(
                    f"shape_meta.images[{index}].shape must be [3,H,W], got {shape}"
                )
            height, width = int(shape[1]), int(shape[2])
            image = _rotate_180(cameras[camera_name])
            resized.append(_center_crop_resize(image, width=width, height=height))

        if len(resized) == 1:
            rgb = resized[0]
        elif self.concat_multi_camera == "horizontal":
            rgb = np.concatenate(resized, axis=1)
        else:
            rgb = np.concatenate(resized, axis=0)

        if self.video_size is not None:
            expected_h, expected_w = (int(x) for x in self.video_size)
            if tuple(rgb.shape[:2]) != (expected_h, expected_w):
                raise ValueError(
                    "Configured per-camera shapes and concat_multi_camera produce "
                    f"{tuple(rgb.shape[:2])}, but data.train.video_size says "
                    f"({expected_h}, {expected_w})"
                )

        import torch

        image_tensor = torch.as_tensor(np.ascontiguousarray(rgb)).permute(2, 0, 1).unsqueeze(0)
        image_tensor = image_tensor.to(device=_model_device(self.model), dtype=_model_dtype(self.model))
        image_tensor = image_tensor * (2.0 / 255.0) - 1.0
        proprio = self._prepare_state(observation.state)
        return PreparedObservation(
            image=image_tensor,
            proprio=proprio,
            # FastWAMAdapter applies FASTWAM_PROMPT_TEMPLATE in predict_action.
            prompt=observation.instruction,
            metadata={
                **dict(observation.metadata),
                "camera_order": camera_names,
                "camera_concat": self.concat_multi_camera if len(camera_names) == 2 else None,
                "fastwam_image_shape": list(image_tensor.shape),
                "fastwam_state_key": self.state_key,
                "dataset_stats": str(self.dataset_stats.get("_source", "in_memory")),
            },
        )

    def _prepare_state(self, state: Mapping[str, Any]):
        try:
            position = np.asarray(state["eef_position"], dtype=np.float32).reshape(-1)
            quaternion = np.asarray(state["eef_quaternion"], dtype=np.float32).reshape(-1).copy()
            gripper = np.asarray(state["gripper_position"], dtype=np.float32).reshape(-1)
        except KeyError as exc:
            raise KeyError(f"LIBERO state is missing {exc.args[0]!r}") from exc
        if position.size != 3 or quaternion.size != 4:
            raise ValueError(
                "FastWAM LIBERO expects eef_position size 3 and eef_quaternion size 4; "
                f"got {position.size} and {quaternion.size}"
            )
        proprio = np.concatenate(
            [position, _quaternion_xyzw_to_axis_angle(quaternion), gripper]
        ).astype(np.float32, copy=False)
        stats = self.dataset_stats["state"][self.state_key]
        normalized = _normalize_field(
            proprio,
            stats,
            mode=_normalization_mode(self.processor_config, "state", self.state_key),
            use_stepwise=False,
            field_name=f"state/{self.state_key}",
        )
        import torch

        return torch.as_tensor(normalized, dtype=torch.float32, device=_model_device(self.model)).unsqueeze(0)

    def postprocess_actions(self, actions: Any, *, environment: str) -> np.ndarray:
        """Denormalize FastWAM actions and map its gripper to LIBERO's signs."""
        if environment.lower() != "libero":
            raise ValueError(
                "FastWAMLiberoAdapter only implements action decoding for environment='libero'"
            )
        values = _as_numpy(actions).astype(np.float32, copy=True)
        if values.ndim == 3 and values.shape[0] == 1:
            values = values[0]
        if values.ndim != 2 or values.shape[-1] != 7:
            raise ValueError(f"Expected normalized FastWAM actions [T,7], got {values.shape}")

        use_stepwise = bool(self.processor_config.get("use_stepwise_action_norm", False))
        mode = _normalization_mode(self.processor_config, "action", self.action_key)
        decoded = _denormalize_field(
            values,
            self.dataset_stats["action"][self.action_key],
            mode=mode,
            use_stepwise=use_stepwise,
            field_name=f"action/{self.action_key}",
        )
        # FastWAM's LIBERO data convention stores 0=close, 1=open. LIBERO's
        # controller expects -1=open, +1=close, hence the affine sign mapping.
        decoded[..., -1] = 1.0 - 2.0 * decoded[..., -1]
        if self.binarize_gripper:
            decoded[..., -1] = np.sign(decoded[..., -1])
        return decoded


def _plain_mapping(value: Any) -> dict[str, Any]:
    try:
        from omegaconf import OmegaConf
        if OmegaConf.is_config(value):
            value = OmegaConf.to_container(value, resolve=True)
    except ImportError:
        pass
    if not isinstance(value, Mapping):
        raise TypeError("data_config must be a mapping or an OmegaConf config")
    return _plain(value)


def _plain(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    return value


def _load_stats(stats: Mapping[str, Any] | str | Path) -> dict[str, Any]:
    if isinstance(stats, (str, Path)):
        path = Path(stats).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"FastWAM dataset stats JSON not found: {path}")
        with path.open("r", encoding="utf-8") as stream:
            result = json.load(stream)
        if not isinstance(result, dict):
            raise TypeError("Dataset stats JSON root must be an object")
        result["_source"] = str(path)
        return result
    if not isinstance(stats, Mapping):
        raise TypeError("dataset_stats must be a mapping or path to a stats JSON file")
    return _plain(stats)


def _model_device(model: Any):
    import torch

    device = getattr(model, "device", None)
    if device is not None:
        return torch.device(device)
    try:
        return next(model.parameters()).device
    except (AttributeError, StopIteration, TypeError):
        return torch.device("cpu")


def _model_dtype(model: Any):
    import torch

    dtype = getattr(model, "torch_dtype", None)
    if isinstance(dtype, torch.dtype):
        return dtype
    try:
        return next(model.parameters()).dtype
    except (AttributeError, StopIteration, TypeError):
        return torch.float32


def _rotate_180(image: Any) -> np.ndarray:
    image = np.asarray(image)
    if image.ndim != 3 or image.shape[-1] != 3:
        raise ValueError(f"Expected RGB camera image [H,W,3], got {image.shape}")
    return np.ascontiguousarray(image[::-1, ::-1])


def _center_crop_resize(image: np.ndarray, *, width: int, height: int) -> np.ndarray:
    from PIL import Image

    pil = Image.fromarray(image)
    source_w, source_h = pil.size
    scale = max(width / source_w, height / source_h)
    resized = pil.resize(
        (round(source_w * scale), round(source_h * scale)),
        resample=Image.Resampling.BILINEAR,
    )
    resized_w, resized_h = resized.size
    left = max((resized_w - width) // 2, 0)
    top = max((resized_h - height) // 2, 0)
    return np.asarray(
        resized.crop((left, top, left + width, top + height)), dtype=np.uint8
    )


def _quaternion_xyzw_to_axis_angle(quaternion: np.ndarray) -> np.ndarray:
    quaternion = quaternion.astype(np.float32, copy=True)
    quaternion[3] = np.clip(quaternion[3], -1.0, 1.0)
    denominator = math.sqrt(max(0.0, 1.0 - float(quaternion[3] * quaternion[3])))
    if math.isclose(denominator, 0.0):
        return np.zeros(3, dtype=np.float32)
    return (quaternion[:3] * (2.0 * math.acos(float(quaternion[3]))) / denominator).astype(np.float32)


def _as_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach()
        if hasattr(value, "float"):
            value = value.float()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value)

def _normalization_mode(processor: Mapping[str, Any], kind: str, key: str) -> str:
    exceptions = processor.get("norm_exception_mode") or {}
    selected = exceptions.get(kind, {}).get(key) if isinstance(exceptions, Mapping) else None
    return str(selected or processor.get("norm_default_mode", "min/max"))


def _stats_values(
    stats: Mapping[str, Any], *, mode: str, use_stepwise: bool, field_name: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, np.ndarray | None]:
    prefix = "stepwise_" if use_stepwise else "global_"

    def get(name: str):
        key = prefix + name
        if key not in stats:
            raise KeyError(f"{field_name} stats require key {key!r}")
        return np.asarray(stats[key], dtype=np.float32)

    template = get("min")
    if mode == "z-score":
        return template, template, get("mean"), get("std")
    if mode == "min/max":
        return get("min"), get("max"), None, None
    if mode == "q01/q99":
        return get("q01"), get("q99"), None, None
    try:
        constant_min, constant_max = (float(x) for x in mode.split("/"))
    except (ValueError, TypeError) as exc:
        raise ValueError(f"Unsupported normalization mode {mode!r} for {field_name}") from exc
    return (
        np.full_like(template, constant_min),
        np.full_like(template, constant_max),
        None,
        None,
    )


def _normalization_parameters(
    stats: Mapping[str, Any], *, mode: str, use_stepwise: bool, field_name: str
) -> tuple[np.ndarray, np.ndarray]:
    low, high, mean, std = _stats_values(
        stats, mode=mode, use_stepwise=use_stepwise, field_name=field_name
    )
    if mode == "z-score":
        scale = 1.0 / (std + 1e-8)
        offset = -mean / (std + 1e-8)
        return scale, offset
    value_range = high - low
    constant = value_range < 1e-4
    safe_range = np.where(constant, 2.0, value_range)
    scale = 2.0 / safe_range
    offset = -1.0 - scale * low
    offset = np.where(constant, -low, offset)
    return scale.astype(np.float32), offset.astype(np.float32)


def _normalize_field(
    value: np.ndarray,
    stats: Mapping[str, Any],
    *,
    mode: str,
    use_stepwise: bool,
    field_name: str,
) -> np.ndarray:
    scale, offset = _normalization_parameters(
        stats, mode=mode, use_stepwise=use_stepwise, field_name=field_name
    )
    return np.clip(value * scale + offset, -5.0, 5.0).astype(np.float32)


def _denormalize_field(
    value: np.ndarray,
    stats: Mapping[str, Any],
    *,
    mode: str,
    use_stepwise: bool,
    field_name: str,
) -> np.ndarray:
    scale, offset = _normalization_parameters(
        stats, mode=mode, use_stepwise=use_stepwise, field_name=field_name
    )
    return ((value - offset) / scale).astype(np.float32)