"""Model interfaces and registered built-in backends."""

from .registry import (
    get_model_adapter,
    load_model,
    load_model_adapter,
    register_model_adapter,
    register_model_loader,
    registered_model_adapters,
    registered_model_loaders,
)
# Import backend modules for registration. Their heavy dependencies are lazy.
from .fastwam.adapter import FastWAMAdapter
from .fastwam.loader import load_fastwam_backend
from .lingbot_va.adapter import LingBotVAAdapter
from .lingbot_va.loader import load_lingbot_va

__all__ = [
    "FastWAMAdapter",
    "LingBotVAAdapter",
    "get_model_adapter",
    "load_model",
    "load_model_adapter",
    "register_model_adapter",
    "register_model_loader",
    "registered_model_adapters",
    "registered_model_loaders",
]

