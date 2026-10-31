# wam_interp: vendored from OpenWAM@3950b6d9 (Apache-2.0); import paths rewritten from `openwam.*`.
"""Optional deploy-time accelerators (toggled via ``cfg.optimization``).

- :class:`DiTVelocityCache`: Skip redundant DiT forward passes

The sync/async execution mechanisms live in ``openwam.deploy.executors``.
"""

from wam_interp.models.openwam.model_code.deploy.optimizations.dit_cache import DiTVelocityCache

__all__ = [
    "DiTVelocityCache",
]
