"""Load a deliberately small, explicit model-ready NPZ input format."""

from pathlib import Path
from typing import Any

from wam_interp.types import PreparedObservation


def load_observation_npz(path: str | Path, prompt: str) -> PreparedObservation:
    try:
        import numpy as np
    except ImportError as exc:
        raise RuntimeError("NumPy is required to read observation NPZ files") from exc

    with np.load(Path(path), allow_pickle=False) as data:
        missing = {"image", "proprio"} - set(data.files)
        if missing:
            raise ValueError(f"Observation NPZ is missing arrays: {sorted(missing)}")
        image = np.asarray(data["image"])
        proprio = np.asarray(data["proprio"], dtype=np.float32)

    if image.ndim != 3:
        raise ValueError(f"image must be a 3-D CHW or HWC array, got {image.shape}")
    if image.shape[0] == 3:
        pass
    elif image.shape[-1] == 3:
        image = np.moveaxis(image, -1, 0)
    else:
        raise ValueError(f"image must have 3 channels, got {image.shape}")
    if image.dtype == np.uint8:
        image = image.astype(np.float32) / 127.5 - 1.0
    else:
        image = image.astype(np.float32)
        if image.size and (image.min() < -1.001 or image.max() > 1.001):
            raise ValueError("Floating image arrays must already be scaled to [-1, 1]")
    if proprio.ndim != 1:
        raise ValueError(f"proprio must be a 1-D normalized vector, got {proprio.shape}")
    return prepare_observation(image=image, proprio=proprio, prompt=prompt,
                               metadata={"source": str(Path(path).resolve())})


def prepare_observation(*, image: Any, proprio: Any, prompt: str,
                        metadata: dict[str, Any] | None = None) -> PreparedObservation:
    """Convert validated model-ready arrays to tensors without applying transforms."""
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("PyTorch is required to prepare FastWAM inputs") from exc

    image_tensor = torch.as_tensor(image, dtype=torch.float32)
    proprio_tensor = torch.as_tensor(proprio, dtype=torch.float32)
    if image_tensor.ndim != 3 or image_tensor.shape[0] != 3:
        raise ValueError(f"image tensor must be [3,H,W], got {tuple(image_tensor.shape)}")
    if image_tensor.shape[-2] % 16 or image_tensor.shape[-1] % 16:
        raise ValueError("Image height and width must be divisible by 16")
    if proprio_tensor.ndim != 1:
        raise ValueError("proprio tensor must be one-dimensional")
    if not prompt.strip():
        raise ValueError("prompt cannot be empty")
    return PreparedObservation(image=image_tensor, proprio=proprio_tensor,
                               prompt=prompt, metadata=metadata or {})

