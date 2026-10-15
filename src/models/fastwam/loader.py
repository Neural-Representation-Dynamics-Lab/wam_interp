"""Compose this project's FastWAM configs and use the FastWAM checkpoint API."""

from pathlib import Path
from typing import Any

from wam_interp.models.registry import register_model_loader


def compose_fastwam_config(task: str, repository_root: str | Path,
                           overrides: list[str] | None = None):
    try:
        from hydra import compose, initialize_config_dir
    except ImportError as exc:
        raise RuntimeError("Install wam-interp[fastwam] to compose FastWAM configs") from exc

    config_dir = (Path(repository_root) / "configs" / "fastwam").resolve()
    if not config_dir.is_dir():
        raise FileNotFoundError(f"FastWAM config directory not found: {config_dir}")
    extra = [f"task={task}", "EVALUATION.compile_action_infer=false"]
    extra.extend(overrides or [])
    with initialize_config_dir(version_base=None, config_dir=str(config_dir)):
        return compose(config_name="sim_libero", overrides=extra)


def _resolve_dtype(name: str):
    import torch

    mapping = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}
    try:
        return mapping[name]
    except KeyError as exc:
        raise ValueError(f"Unsupported dtype {name!r}; choose {sorted(mapping)}") from exc


def load_fastwam(config: Any, checkpoint: str | Path, *, device: str = "cuda",
                 dtype: str = "bfloat16"):
    """Instantiate the composed config model and load via FastWAM.load_checkpoint."""
    try:
        from hydra.utils import instantiate
    except ImportError as exc:
        raise RuntimeError("Install wam-interp[fastwam] to load FastWAM") from exc

    model = instantiate(config.model, model_dtype=_resolve_dtype(dtype), device=device)
    model.load_checkpoint(str(checkpoint))
    model.to(device=device)
    model.eval()
    return model


@register_model_loader("fastwam")
def load_fastwam_backend(*, task: str, repository_root: str | Path,
                         checkpoint: str | Path, device: str = "cuda",
                         dtype: str = "bfloat16"):
    config = compose_fastwam_config(task, repository_root)
    return load_fastwam(config, checkpoint, device=device, dtype=dtype)

