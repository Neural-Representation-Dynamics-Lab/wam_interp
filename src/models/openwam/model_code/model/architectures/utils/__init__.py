# wam_interp: vendored from OpenWAM@3950b6d9 (Apache-2.0); import paths rewritten from `openwam.*`.
"""Architecture-side shared utilities: common config/mask helpers."""

from wam_interp.models.openwam.model_code.model.architectures.utils.common import (
    VAE_TEMPORAL_FACTOR,
    compute_video_tokens_per_frame,
    downsample_video_mask_to_latent,
    resolve_bridge_layers,
)

__all__ = [
    "VAE_TEMPORAL_FACTOR",
    "compute_video_tokens_per_frame",
    "downsample_video_mask_to_latent",
    "resolve_bridge_layers",
]
