"""Scoped output replacement at named module boundaries."""

from contextlib import ExitStack, contextmanager
from typing import Any, Callable, Iterator


@contextmanager
def replace_module_output(model: Any, module_name: str,
                          transform: Callable[[Any], Any]) -> Iterator[None]:
    """Apply transform to one module's output while its context is active.

    Useful for projection-level patches. It does not capture post-RoPE/cache
    values unless the selected module itself produces those values.
    """
    modules = dict(model.named_modules())
    if module_name not in modules:
        raise ValueError(f"Unknown module path: {module_name}")
    module = modules[module_name]
    calls = []

    def hook(_module, _inputs, output):
        calls.append(1)
        return transform(output)

    handle = module.register_forward_hook(hook)
    try:
        yield
    finally:
        handle.remove()
    # Reached only when the body finished without an exception.
    if not calls:
        raise RuntimeError(f"Intervention on {module_name!r} never ran: that module is not called by the "
                           "model's forward pass, so the edit had no effect. Target a module that is called.")


@contextmanager
def replace_module_outputs(model: Any, transforms: dict[str, Callable[[Any], Any]]) -> Iterator[None]:
    with ExitStack() as stack:
        for path, transform in transforms.items():
            stack.enter_context(replace_module_output(model, path, transform))
        yield

