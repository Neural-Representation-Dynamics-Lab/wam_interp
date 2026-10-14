"""Model interfaces and registered built-in backends."""

from .registry import (
    get_model_adapter,
    load_model,
    register_model_adapter,
    register_model_loader,
    registered_model_adapters,
    registered_model_loaders,
)
# Import backend modules for registration. Their heavy dependencies are lazy.
from .fastwam.adapter import FastWAMAdapter
from .fastwam.loader import load_fastwam_backend

__all__ = [
    "FastWAMAdapter",
    "get_model_adapter",
    "load_model",
    "register_model_adapter",
    "register_model_loader",
    "registered_model_adapters",
    "registered_model_loaders",
]

