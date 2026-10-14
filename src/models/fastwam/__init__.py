"""FastWAM backend, imported only when requested."""

from .adapter import FastWAMAdapter
from .loader import compose_fastwam_config, load_fastwam

__all__ = ["FastWAMAdapter", "compose_fastwam_config", "load_fastwam"]

