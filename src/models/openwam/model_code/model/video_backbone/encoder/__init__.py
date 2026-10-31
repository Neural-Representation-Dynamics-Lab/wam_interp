# wam_interp: vendored from OpenWAM@3950b6d9 (Apache-2.0); import paths rewritten from `openwam.*`.
"""Pluggable video encoder subsystem.

Only activated when ``video_backbone.from_scratch=true`` AND
``video_backbone.encoder`` is set in yaml. Under any other combination the
backbone keeps using its built-in ``pipe.vae`` and the encoder package is inert.

wam_interp: trimmed to wan22_vae; dinov3 / flux2_vae / vjepa21 are not vendored.
"""

from __future__ import annotations

from wam_interp.models.openwam.model_code.model.video_backbone.encoder.base import VideoEncoder, VideoEncoderProperties
from wam_interp.models.openwam.model_code.model.video_backbone.encoder.registry import (
    _VIDEO_ENCODER_REGISTRY,
    build_video_encoder,
    register_video_encoder,
)

__all__ = [
    "VideoEncoder",
    "VideoEncoderProperties",
    "WanVideoVAEEncoder",
    "build_video_encoder",
    "register_video_encoder",
    "_VIDEO_ENCODER_REGISTRY",
]

from wam_interp.models.openwam.model_code.model.video_backbone.encoder.wan22_vae import WanVideoVAEEncoder  # noqa: E402, F401
