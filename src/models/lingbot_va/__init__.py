"""LingBot-VA backend, imported only when requested."""

from .adapter import LingBotVAAdapter
from .loader import lingbot_va_config, load_lingbot_va

__all__ = [
    "LingBotVAAdapter",
    "LingBotVALiberoAdapter",
    "lingbot_va_config",
    "load_lingbot_va",
]


def __getattr__(name: str):
    if name == "LingBotVALiberoAdapter":
        from .libero_adapter import LingBotVALiberoAdapter
        return LingBotVALiberoAdapter
    raise AttributeError(name)
