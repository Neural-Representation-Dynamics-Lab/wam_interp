# wam_interp: vendored from OpenWAM@3950b6d9 (Apache-2.0); import paths rewritten from `openwam.*`.
"""Video backbone package for OpenWAM.

wam_interp: trimmed to the Wan backbones; Cosmos backbones are not vendored.
"""

from __future__ import annotations

from wam_interp.models.openwam.model_code.model.video_backbone.base import BlockLoopState, VideoBackbone
from wam_interp.models.openwam.model_code.model.video_backbone.registry import (
    _VIDEO_BACKBONE_REGISTRY,
    build_video_backbone,
    register_video_backbone,
)

__all__ = [
    "BlockLoopState",
    "VideoBackbone",
    "Wan22Ti2v",
    "Wan21",
    "build_video_backbone",
    "register_video_backbone",
    "_VIDEO_BACKBONE_REGISTRY",
]

# Built-in registrations (kept at the bottom so the implementation can import
# from ``registry`` / ``base`` without circular issues).
from wam_interp.models.openwam.model_code.model.video_backbone.wan_backbone import Wan21, Wan22Ti2v  # noqa: E402

register_video_backbone("wan22_ti2v_5b")(Wan22Ti2v)
register_video_backbone("wan21_vace_1_3b")(Wan21)
register_video_backbone("wan21_i2v_14b_480p")(Wan21)
