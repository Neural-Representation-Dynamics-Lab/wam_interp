"""Long-WAM benchmark transforms, ported from upstream runtime/ and benchmarks/ code at the pinned commit."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    import torch


def build_observation_window(
    obs_frame_buffer: "list[torch.Tensor]",
    past_obs_size: int,
    action_video_freq_ratio: int,
):
    """Build the v0b past->current observation window matching `RobotVideoDataset` sampling.

    Mirrors training exactly:
      * The dataset obs window spans raw-frame deltas [-past_obs_size .. ] (delta_timestamps in
        `base_lerobot_dataset`), then samples `video_sample_indices = range(0, num_frames, freq)`.
      * The clean (past->current) region is the first M latent frames = VAE of the first
        T_sampled = past_obs_size // freq + 1 sampled frames, i.e. raw deltas
        [-past_obs_size, -past_obs_size+freq, ..., -freq, 0] -> the last `past_obs_size + 1`
        raw frames, sub-sampled every `freq`, ending on the CURRENT frame.
      * Episode-start padding repeats the OLDEST available frame (matches lerobot
        `_get_query_indices`: `max(ep_start, min(ep_end-1, idx+delta))` clamps out-of-range
        past deltas to `ep_start`).

    Args:
        obs_frame_buffer: list of processed model-input frames, each [3,H,W], most-recent last.
            Must contain at least one frame (the current obs).
        past_obs_size: pixel `past_obs_size` (P). 0 -> returns None (v0a single-frame path).
        action_video_freq_ratio: sampling stride (`freq`).

    Returns:
        obs_window tensor [1, 3, T_sampled, H, W] with T_sampled = P // freq + 1, or None when
        past_obs_size == 0 (v0a: caller falls back to single `input_image`).
    """
    import torch

    if past_obs_size <= 0:
        return None
    if len(obs_frame_buffer) == 0:
        raise ValueError("`obs_frame_buffer` must contain at least the current observation frame.")
    if past_obs_size % action_video_freq_ratio != 0:
        raise ValueError(
            f"past_obs_size ({past_obs_size}) must be divisible by action_video_freq_ratio "
            f"({action_video_freq_ratio}) so the clean/noised boundary lands on a latent-frame edge."
        )

    needed = past_obs_size + 1  # raw frames at deltas [-P .. 0]
    # Pad-at-start by repeating the oldest available frame (repeat-oldest, matches lerobot).
    if len(obs_frame_buffer) < needed:
        pad = [obs_frame_buffer[0]] * (needed - len(obs_frame_buffer))
        raw_window = pad + list(obs_frame_buffer)
    else:
        raw_window = list(obs_frame_buffer[-needed:])
    # Sub-sample every `freq`: indices [0, freq, 2*freq, ..., P] -> deltas [-P, -P+freq, ..., 0].
    sampled = raw_window[0::action_video_freq_ratio]
    expected_t = past_obs_size // action_video_freq_ratio + 1
    if len(sampled) != expected_t:
        raise ValueError(
            f"obs window sampling produced {len(sampled)} frames, expected {expected_t} "
            f"(past_obs_size={past_obs_size}, action_video_freq_ratio={action_video_freq_ratio})."
        )
    # Stack to [1, 3, T_sampled, H, W].
    temporal_dim = sampled[0].ndim - 2  # [C,H,W] or view-preserving [V,C,H,W]
    window = torch.stack(sampled, dim=temporal_dim).unsqueeze(
        0
    )  # frames are [3,H,W] -> [3,T,H,W] -> [1,3,T,H,W]
    return window


def window_indices(num_observed: int, past_obs_size: int, stride: int) -> list[int]:
    """Episode-relative observation index used for each window slot (for provenance)."""
    needed = past_obs_size + 1
    first = num_observed - needed
    return [max(0, first + offset) for offset in range(0, needed, stride)]


def quat2axisangle(quat):
    """
    Copied from robosuite: https://github.com/ARISE-Initiative/robosuite/blob/eafb81f54ffc104f905ee48a16bb15f059176ad3/robosuite/utils/transform_utils.py#L490C1-L512C55

    Converts quaternion to axis-angle format.
    Returns a unit vector direction scaled by its angle in radians.

    Args:
        quat (np.array): (x,y,z,w) vec4 float angles

    Returns:
        np.array: (ax,ay,az) axis-angle exponential coordinates
    """
    # clip quaternion
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0

    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        # This is (close to) a zero degree rotation, immediately return
        return np.zeros(3)

    return (quat[:3] * 2.0 * math.acos(quat[3])) / den


def load_dataset_stats(stats: Mapping[str, Any] | str | Path) -> dict[str, Any]:
    """Plain JSON statistics; ``_source`` records the file (ignored by provenance hashing)."""
    if isinstance(stats, (str, Path)):
        path = Path(stats).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Long-WAM dataset statistics not found: {path}")
        with path.open("r", encoding="utf-8") as stream:
            loaded = json.load(stream)
        if not isinstance(loaded, dict):
            raise TypeError("Dataset statistics JSON root must be an object")
        loaded["_source"] = str(path)
        return loaded
    if not isinstance(stats, Mapping):
        raise TypeError("dataset_stats must be a mapping or a path to dataset_stats.json")
    return json.loads(json.dumps(dict(stats), default=lambda value: np.asarray(value).tolist()))


class FieldNormalizer:
    """``SingleFieldLinearNormalizer`` from upstream, with identical float32 torch arithmetic."""

    std_reg = 1e-8
    range_tol = 1e-4
    output_max = 1.0
    output_min = -1.0

    def __init__(self, stats: Mapping[str, Any], mode: str = "min/max"):
        import torch

        # Upstream load_dataset_stats_from_json: JSON lists -> float64 numpy -> torch -> float32.
        stats = {key: torch.from_numpy(np.array(value)).to(torch.float32) for key, value in stats.items()}
        self.stats = stats
        self.mode = mode

        if mode == "z-score":
            input_mean, input_std = stats["mean"], stats["std"]
            scale = 1.0 / (input_std + self.std_reg)
            offset = -input_mean / (input_std + self.std_reg)
        else:
            if mode == "min/max":
                input_min, input_max = stats["min"], stats["max"]
            elif mode == "q01/q99":
                input_min, input_max = stats["q01"], stats["q99"]
            else:
                # parse const_min/const_max
                input_min, input_max = map(float, mode.split("/"))
                input_min = torch.full_like(stats["min"], input_min)
                input_max = torch.full_like(stats["max"], input_max)

            input_range = input_max - input_min
            ignore_dim = input_range < self.range_tol
            input_range[ignore_dim] = self.output_max - self.output_min
            scale = (self.output_max - self.output_min) / input_range
            offset = self.output_min - scale * input_min
            offset[ignore_dim] = (self.output_max + self.output_min) / 2 - input_min[ignore_dim]

        self.scale = scale
        self.offset = offset

    def forward(self, x):
        import torch

        x = x * self.scale + self.offset
        x = torch.clamp(x, -5.0, 5.0)
        return x

    def backward(self, x):
        x = (x - self.offset) / self.scale
        return x


def field_normalizer(processor: Mapping[str, Any], stats: Mapping[str, Any], kind: str, key: str) -> FieldNormalizer:
    """Select stats and mode exactly as upstream ``LinearNormalizer.__init__`` does."""
    try:
        field = stats[kind][key]
    except (KeyError, TypeError) as exc:
        raise KeyError(f"Dataset statistics must contain [{kind!r}][{key!r}]") from exc
    prefix = "stepwise_" if kind == "action" and processor.get("use_stepwise_action_norm") else "global_"
    selected = {name.removeprefix(prefix): value for name, value in field.items() if name.startswith(prefix)}
    exceptions = processor.get("norm_exception_mode")
    if exceptions is not None and kind in exceptions and key in exceptions[kind]:
        mode = exceptions[kind][key]
    else:
        mode = processor["norm_default_mode"]
    return FieldNormalizer(selected, mode=str(mode))


def _center_crop_resize(image: np.ndarray, width: int, height: int) -> np.ndarray:
    from PIL import Image

    pil_image = Image.fromarray(image)
    src_w, src_h = pil_image.size
    scale = max(width / src_w, height / src_h)
    resized = pil_image.resize(
        (round(src_w * scale), round(src_h * scale)), resample=Image.BILINEAR
    )
    rw, rh = resized.size
    left = max((rw - width) // 2, 0)
    top = max((rh - height) // 2, 0)
    cropped = resized.crop((left, top, left + width, top + height))
    return np.asarray(cropped, dtype=np.uint8)


def _rgb_uint8(image: Any, name: str) -> np.ndarray:
    array = np.asarray(image)
    if array.ndim != 3 or array.shape[-1] != 3 or array.dtype != np.uint8:
        raise ValueError(f"Camera {name!r} must be uint8 RGB [H,W,3], got {array.dtype} {array.shape}")
    return array


def libero_model_image(cameras: Mapping[str, Any], image_meta, num_cameras: int, concatenation: str,
                       video_size, *, device, dtype):
    """Rotate, crop/resize and concatenate LIBERO cameras into [1,3,H,W] in [-1,1]."""
    import torch

    names = ("agentview", "wrist")[:num_cameras]
    missing = [name for name in names if name not in cameras]
    if missing:
        raise KeyError(f"Long-WAM LIBERO preprocessing needs cameras {missing}")
    # get_libero_image: rotate 180 degrees to match train preprocessing.
    rotated = [np.ascontiguousarray(_rgb_uint8(cameras[name], name)[::-1, ::-1]) for name in names]
    resized = []
    for index, image in enumerate(rotated):
        shape = image_meta[index]["shape"]
        if len(shape) != 3:
            raise ValueError(f"shape_meta.images[{index}].shape must be [C,H,W], got {shape}")
        resized.append(_center_crop_resize(image, width=int(shape[2]), height=int(shape[1])))
    if num_cameras == 1:
        rgb = resized[0]
    elif concatenation == "horizontal":
        rgb = np.concatenate(resized, axis=1)
    elif concatenation == "vertical":
        rgb = np.concatenate(resized, axis=0)
    else:
        raise ValueError(f"Invalid concat_multi_camera: {concatenation}")
    expected = tuple(int(value) for value in video_size)
    if tuple(rgb.shape[:2]) != expected:
        raise ValueError(f"Per-camera resize + concat gives {tuple(rgb.shape[:2])}, but video_size is {expected}")
    x = torch.tensor(rgb).permute(2, 0, 1).unsqueeze(0).to(device=device, dtype=dtype)
    x = x * (2.0 / 255.0) - 1.0
    return x, list(names)


def _resize_like_torchvision(tensor, size_hw):
    import torch.nn.functional as F

    size = [int(value) for value in size_hw]
    if list(tensor.shape[-2:]) == size:
        return tensor
    return F.interpolate(tensor.unsqueeze(0), size=size, mode="bilinear",
                         align_corners=False, antialias=True).squeeze(0)


def _resize_training_camera(image: np.ndarray, *, processor_size_hw, layout_size_hw):
    """Apply the same two resize stages used by the RoboTwin training dataset."""
    import torch

    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"Expected an RGB image [H,W,3], got {tuple(image.shape)}")
    tensor = torch.from_numpy(np.ascontiguousarray(image)).permute(2, 0, 1)
    tensor = tensor.to(dtype=torch.float32).div_(255.0)
    tensor = _resize_like_torchvision(tensor, processor_size_hw)
    return _resize_like_torchvision(tensor, layout_size_hw)


ROBOTWIN_CAMERAS = (("cam_high", "head", (256, 320)),
                    ("cam_left_wrist", "left_wrist", (128, 160)),
                    ("cam_right_wrist", "right_wrist", (128, 160)))


def robotwin_model_image(cameras: Mapping[str, Any], image_meta, *, device, dtype):
    """Head camera above [left | right] wrist cameras: [1,3,384,320] in [-1,1]."""
    import torch

    image_shapes = {str(meta["key"]): tuple(int(v) for v in meta["shape"][-2:]) for meta in image_meta}
    if set(image_shapes) != {key for key, _, _ in ROBOTWIN_CAMERAS}:
        raise ValueError(f"RoboTwin checkpoint image keys must be cam_high/left/right wrist, got {sorted(image_shapes)}")
    views = []
    for key, camera, layout in ROBOTWIN_CAMERAS:
        if camera not in cameras:
            raise KeyError(f"Long-WAM RoboTwin preprocessing needs camera {camera!r}")
        views.append(_resize_training_camera(_rgb_uint8(cameras[camera], camera),
                                             processor_size_hw=image_shapes[key], layout_size_hw=layout))
    head, left, right = views
    bottom = torch.cat([left, right], dim=2)
    image_tensor = torch.cat([head, bottom], dim=1).unsqueeze(0).to(device=device, dtype=dtype)
    return image_tensor.mul_(2.0).sub_(1.0)
