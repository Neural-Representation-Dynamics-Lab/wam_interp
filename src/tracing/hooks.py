"""Scoped module hooks. These do not imply that every MoT boundary is visible."""

from contextlib import ExitStack, contextmanager
from typing import Any, Iterator


def _compact(value: Any):
    if hasattr(value, "shape"):
        return {"shape": tuple(value.shape), "dtype": str(getattr(value, "dtype", "?"))}
    if isinstance(value, (tuple, list)):
        return [_compact(item) for item in value]
    if isinstance(value, dict):
        return {key: _compact(item) for key, item in value.items()}
    return type(value).__name__


@contextmanager
def capture_module_outputs(model: Any, module_names: tuple[str, ...],
                           recorder: Any | None) -> Iterator[None]:
    if not module_names or recorder is None:
        yield
        return
    modules = dict(model.named_modules())
    missing = [name for name in module_names if name not in modules]
    if missing:
        raise ValueError(f"Unknown module paths: {missing}")
    with ExitStack() as stack:
        for name in module_names:
            module = modules[name]

            def hook(_module, inputs, output, path=name):
                recorder.record(
                    "module_forward",
                    {"inputs": inputs, "output": output},
                    module=path,
                    input_summary=_compact(inputs),
                    output_summary=_compact(output),
                )
            stack.callback(module.register_forward_hook(hook).remove)
        yield