"""Compose this project's FastWAM configs and use the FastWAM checkpoint API."""

from pathlib import Path
from typing import Any

from wam_interp.models.registry import register_model_loader


def _default_repository_root() -> Path:
    """Return this checkout's root without assuming its directory name."""
    return Path(__file__).resolve().parents[3]


def _fastwam_config_dir(repository_root: str | Path | None) -> Path:
    if repository_root is None:
        roots = [_default_repository_root()]
    else:
        roots = [Path(repository_root).expanduser().resolve(), _default_repository_root()]

    for root in roots:
        config_dir = root / "configs" / "fastwam"
        try:
            if config_dir.is_dir():
                return config_dir
        except OSError:
            continue

    expected = roots[0] / "configs" / "fastwam"
    raise FileNotFoundError(f"FastWAM config directory not found: {expected}")


def compose_fastwam_config(task: str, repository_root: str | Path | None = None,
                           overrides: list[str] | None = None, *, benchmark: str = "libero"):
    try:
        from hydra import compose, initialize_config_dir
    except ImportError as exc:
        raise RuntimeError("Install the repository with its FastWAM dependencies to compose configs") from exc

    config_dir = _fastwam_config_dir(repository_root)
    if benchmark not in {"libero", "libero_plus", "robotwin"}:
        raise ValueError(f"Unsupported benchmark {benchmark!r}")
    extra = [f"task={task}"]
    if benchmark != "robotwin":
        extra.append("EVALUATION.compile_action_infer=false")
    extra.extend(overrides or [])
    with initialize_config_dir(version_base=None, config_dir=str(config_dir)):
        return compose(config_name="sim_robotwin" if benchmark == "robotwin" else "sim_libero", overrides=extra)


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
    checkpoint = Path(checkpoint).expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"FastWAM checkpoint not found: {checkpoint}")
    try:
        from hydra.utils import instantiate
    except ImportError as exc:
        raise RuntimeError("Install the repository with its FastWAM dependencies to load FastWAM") from exc

    model = instantiate(config.model, model_dtype=_resolve_dtype(dtype), device=device)
    model.load_checkpoint(str(checkpoint))
    model.to(device=device)
    model.eval()
    from omegaconf import OmegaConf
    model._wam_interp_metadata = {"checkpoint": str(checkpoint),
        "resolved_model_config": OmegaConf.to_container(config.model, resolve=True)}
    return model


@register_model_loader("fastwam")
def load_fastwam_backend(*, task: str, repository_root: str | Path | None = None,
                         checkpoint: str | Path, device: str = "cuda",
                         dtype: str = "bfloat16", benchmark: str = "libero", overrides=None):
    config = compose_fastwam_config(task, repository_root, overrides, benchmark=benchmark)
    return load_fastwam(config, checkpoint, device=device, dtype=dtype)

