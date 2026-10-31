"""OpenWAM inference backend; the vendored model code is imported only when a checkpoint loads."""

from .adapter import OpenWAMAdapter
from .loader import UPSTREAM_REVISION, load_openwam

__all__ = ["OpenWAMAdapter", "UPSTREAM_REVISION", "load_openwam"]
