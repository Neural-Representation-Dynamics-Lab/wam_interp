"""Faster-WAM inference backend; heavy imports remain inside the loader."""

from .adapter import FasterWAMAdapter
from .loader import compose_fasterwam_config, load_fasterwam

__all__ = ["FasterWAMAdapter", "compose_fasterwam_config", "load_fasterwam"]
