"""Explicit registries for model loaders and inference adapters."""

from collections.abc import Callable
from typing import Any

_ADAPTERS: dict[str, Callable[..., Any]] = {}
_LOADERS: dict[str, Callable[..., Any]] = {}


def _register(registry, name: str):
    def decorator(factory: Callable[..., Any]):
        if name in registry:
            raise ValueError(f"Entry already registered: {name}")
        registry[name] = factory
        return factory
    return decorator


def register_model_adapter(name: str):
    return _register(_ADAPTERS, name)


def register_model_loader(name: str):
    return _register(_LOADERS, name)


def get_model_adapter(name: str, **kwargs: Any):
    try:
        factory = _ADAPTERS[name]
    except KeyError as exc:
        raise KeyError(f"Unknown model adapter {name!r}; registered: {sorted(_ADAPTERS)}") from exc
    return factory(**kwargs)


def load_model(name: str, **kwargs: Any):
    try:
        loader = _LOADERS[name]
    except KeyError as exc:
        raise KeyError(f"Unknown model loader {name!r}; registered: {sorted(_LOADERS)}") from exc
    return loader(**kwargs)


def registered_model_adapters() -> tuple[str, ...]:
    return tuple(sorted(_ADAPTERS))


def registered_model_loaders() -> tuple[str, ...]:
    return tuple(sorted(_LOADERS))

