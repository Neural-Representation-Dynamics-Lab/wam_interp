"""Compose this project's FastWAM configs and use the FastWAM checkpoint API."""

from pathlib import Path
from typing import Any

from wam_interp.models.registry import register_model_loader

# Sampler shift upstream publishes for each released checkpoint (FastWAM README at 7faa711).
RELEASED_CHECKPOINT_EVALUATION = {
    "libero_uncond_2cam224_1e-4": {"sigma_shift": 5.0},
    "robotwin_uncond_3cam_384_1e-4": {"sigma_shift": 5.0},
    "libero_optional_idm_2cam224_1e-4": {"sigma_shift": 1.0},
}
EVALUATION_KEYS = ("sigma_shift", "num_inference_steps", "text_cfg_scale", "negative_prompt", "rand_device")


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
                           overrides: list[str] | None = None, *, benchmark: str = "libero",
                           released_checkpoint_defaults: bool = True):
    """Compose the FastWAM eval config; ``released_checkpoint_defaults`` applies the published sampler shift."""
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
    given = {item.split("=", 1)[0].lstrip("+~") for item in overrides or []}
    if released_checkpoint_defaults:
        for key, value in RELEASED_CHECKPOINT_EVALUATION.get(task, {}).items():
            if f"EVALUATION.{key}" not in given:
                extra.append(f"EVALUATION.{key}={value}")
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
                 dtype: str = "bfloat16", allow_partial_checkpoint: bool = False):
    """Instantiate the composed config model and load via FastWAM.load_checkpoint."""
    checkpoint = Path(checkpoint).expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"FastWAM checkpoint not found: {checkpoint}")
    try:
        from hydra.utils import instantiate
    except ImportError as exc:
        raise RuntimeError("Install the repository with its FastWAM dependencies to load FastWAM") from exc

    from wam_interp.models.checkpoints import check_checkpoint_loads, record_state_dict_loads

    model = instantiate(config.model, model_dtype=_resolve_dtype(dtype), device=device)
    modules = {name: getattr(model, name, None) for name in ("mot", "video_expert", "proprio_encoder")}
    with record_state_dict_loads(modules) as loads:
        model.load_checkpoint(str(checkpoint))
    load_summary = check_checkpoint_loads(loads, model, checkpoint, allow_partial=allow_partial_checkpoint)
    model.to(device=device)
    model.eval()
    from omegaconf import OmegaConf
    evaluation = {}
    for key in EVALUATION_KEYS:
        try:
            evaluation[key] = OmegaConf.select(config, f"EVALUATION.{key}", default=None)
        except Exception:  # unresolved interpolation outside a Hydra run
            evaluation[key] = None
    model._wam_interp_metadata = {"checkpoint": str(checkpoint), "checkpoint_load": load_summary,
        "resolved_model_config": OmegaConf.to_container(config.model, resolve=True),
        "evaluation": evaluation}
    return model


@register_model_loader("fastwam")
def load_fastwam_backend(*, task: str, repository_root: str | Path | None = None,
                         checkpoint: str | Path, device: str = "cuda",
                         dtype: str = "bfloat16", benchmark: str = "libero", overrides=None):
    config = compose_fastwam_config(task, repository_root, overrides, benchmark=benchmark)
    return load_fastwam(config, checkpoint, device=device, dtype=dtype)

