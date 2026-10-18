"""FastWAM backend, imported only when requested."""

from .adapter import FastWAMAdapter
from .loader import compose_fastwam_config, load_fastwam

__all__ = [
    "FastWAMAdapter",
    "FastWAMLiberoAdapter",
    "compose_fastwam_config",
    "load_fastwam",
]


def __getattr__(name: str):
    # Keep NumPy/Pillow and image-processing imports lazy for inference-only use.
    if name == "FastWAMLiberoAdapter":
        from .libero_adapter import FastWAMLiberoAdapter
        return FastWAMLiberoAdapter
    raise AttributeError(name)