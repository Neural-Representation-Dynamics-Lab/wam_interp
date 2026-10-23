"""Compose local inference configs and load released native Faster-WAM weights."""

from pathlib import Path

from wam_interp.models.registry import register_model_loader

UPSTREAM_REVISION = "83667817df0d4f823f39d90700e61ea2f432ac45"


def compose_fasterwam_config(task="libero_fasterwam_2cam224_1e-4", repository_root=None,
                           overrides=None, *, benchmark="libero"):
    from hydra import compose, initialize_config_dir

    root = Path(repository_root).expanduser().resolve() if repository_root else Path(__file__).resolve().parents[3]
    config_dir = root / "configs" / "fasterwam"
    if not config_dir.is_dir():
        raise FileNotFoundError(f"Faster-WAM config directory not found: {config_dir}")
    if benchmark not in {"libero", "libero_plus", "robotwin"}:
        raise ValueError(f"Unsupported benchmark {benchmark!r}")
    with initialize_config_dir(version_base=None, config_dir=str(config_dir)):
        return compose(config_name="sim_robotwin" if benchmark == "robotwin" else "sim_libero",
                       overrides=[f"task={task}", *(overrides or [])])


def load_fasterwam(config, checkpoint, *, device="cuda", dtype="bfloat16"):
    checkpoint = Path(checkpoint).expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Faster-WAM checkpoint not found: {checkpoint}")
    from hydra.utils import instantiate
    from omegaconf import OmegaConf
    from wam_interp.models.fastwam.loader import _resolve_dtype

    model = instantiate(config.model, model_dtype=_resolve_dtype(dtype), device=device)
    model.load_checkpoint(str(checkpoint))
    model.to(device=device)
    model.eval()
    model._wam_interp_metadata = {
        "checkpoint": str(checkpoint),
        "checkpoint_size_bytes": checkpoint.stat().st_size,
        "upstream_revision": UPSTREAM_REVISION,
        "resolved_config": OmegaConf.to_container(config, resolve=True),
        "model_dtype": dtype,
        "model_device": str(device),
    }
    return model


@register_model_loader("fasterwam")
def load_fasterwam_backend(*, task="libero_fasterwam_2cam224_1e-4", repository_root=None,
                          checkpoint, device="cuda", dtype="bfloat16", overrides=None, benchmark="libero"):
    # Reject missing weights before importing Hydra or allocating model assets.
    if not Path(checkpoint).expanduser().is_file():
        raise FileNotFoundError(f"Faster-WAM checkpoint not found: {checkpoint}")
    config = compose_fasterwam_config(task, repository_root, overrides, benchmark=benchmark)
    return load_fasterwam(config, checkpoint, device=device, dtype=dtype)
