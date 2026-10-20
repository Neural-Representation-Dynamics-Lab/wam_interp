"""Load a released LingBot-VA checkpoint directory into a single-GPU runtime."""

from copy import deepcopy
from pathlib import Path

from wam_interp.models.registry import register_model_loader


def lingbot_va_config(name: str = "libero"):
    if name != "libero":
        raise ValueError(f"Only the LIBERO config is ported so far, got {name!r}")
    from .model_code.va_libero_cfg import va_libero_cfg
    return deepcopy(va_libero_cfg)


def _resolve_dtype(name: str):
    import torch

    mapping = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}
    try:
        return mapping[name]
    except KeyError as exc:
        raise ValueError(f"Unsupported dtype {name!r}; choose {sorted(mapping)}") from exc


@register_model_loader("lingbot_va")
def load_lingbot_va(*, checkpoint_dir: str | Path, config_name: str = "libero",
                    device: str = "cuda", dtype: str = "bfloat16", config=None):
    """checkpoint_dir holds the HF repo layout: transformer/, vae/, text_encoder/, tokenizer/."""
    from .runtime import LingBotVARuntime

    cfg = lingbot_va_config(config_name) if config is None else config
    return LingBotVARuntime(cfg, checkpoint_dir, device=device, dtype=_resolve_dtype(dtype))
