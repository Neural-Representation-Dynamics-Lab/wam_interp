"""Load a released Long-WAM bundle (config.yaml + model.pt) strictly, following upstream inference.py."""

import hashlib
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from wam_interp.models.registry import register_model_loader

UPSTREAM_REPOSITORY = "https://github.com/NVlabs/LongLive"
UPSTREAM_SUBDIRECTORY = "Long-WAM"
UPSTREAM_REVISION = "242635807b4ebb89fa2609a84df7eacde6aed92f"

# Released bundles at the revisions in upstream configs/checkpoints.yaml, with their SHA-256 sums.
RELEASED_BUNDLES = {
    "libero_idm": {"repo_id": "Efficient-Large-Model/Long-WAM-LIBERO-IDM",
                   "revision": "eca2710140fc0d03a2b0259db1d49d5ca60d5122", "benchmark": "libero", "condition": "idm",
                   "sha256": {"model.pt": "2596885d07af089e83ebaf2e1cfc6cee9dc4001e5cfd90b279e60d02a32aaf17",
                              "config.yaml": "16be18488282faad024cc47a91508a59faf9bdc992bda1c4721ffb8dcaf34d2d",
                              "dataset_stats.json": "bd8ab0bd5cf832546277ba661c0c1efde9e9642977fdc1fec599469bf090d168"}},
    "libero_cod": {"repo_id": "Efficient-Large-Model/Long-WAM-LIBERO-COD",
                   "revision": "b73f1c20bbdf7611f4a180c6117dbdbb8416cd54", "benchmark": "libero", "condition": "codenoise",
                   "sha256": {"model.pt": "03815c5bec95740f6aeed933dd9bf9bcb2a894fa3e291b54327f174ce6d655a2",
                              "config.yaml": "4b09e37f5faf7ed7b27a09eab23bcb91290fe6370e2c8765e17b350fd2595ddf",
                              "dataset_stats.json": "1db2f7afc6ee777e785c84c36f4a141a6e34f911dcb6b67f70ce116d55d5c740"}},
    "robotwin2_idm": {"repo_id": "Efficient-Large-Model/Long-WAM-RoboTwin2.0-IDM",
                      "revision": "f456378dec6c4d38ee74717f5d71c0e7490bcc78", "benchmark": "robotwin", "condition": "idm",
                      "sha256": {"model.pt": "8013982c70d7dbd6cb8cd17b71a251409c69eb58e39f1e26005515262d11d711",
                                 "config.yaml": "e4d9f600a46c078f461de083092aef2f454937b53c538f5977e5265bad720707",
                                 "dataset_stats.json": "639e69c81e35568e9a49a60b4ede4f28735594c9ab325ee73931d6360bfd5834"}},
    "robotwin2_cod": {"repo_id": "Efficient-Large-Model/Long-WAM-RoboTwin2.0-COD",
                      "revision": "07b23f10f0b580b8f972c3fed24a86c6a113f7e8", "benchmark": "robotwin", "condition": "codenoise",
                      "sha256": {"model.pt": "25d310fd4586307ec728a1289b6f3b39dda84b51c0893b0b4691c098a7aa70cc",
                                 "config.yaml": "9c7af3145afb870b1ca02c7c669b88ec9b5114fbf62f1faa6ab1151d4ef4aff2",
                                 "dataset_stats.json": "639e69c81e35568e9a49a60b4ede4f28735594c9ab325ee73931d6360bfd5834"}},
}

# Wan assets the bundles load, pinned to the revisions current on 2026-10-30.
BASE_ASSETS = {
    "Wan-AI/Wan2.2-TI2V-5B": {"revision": "921dbaf3f1674a56f47e83fb80a34bac8a8f203e",
                              "files": ("Wan2.2_VAE.pth", "models_t5_umt5-xxl-enc-bf16.pth")},
    "Wan-AI/Wan2.1-T2V-1.3B": {"revision": "37ec512624d61f7aa208f7ea8140a131f93afc9a",
                               "files": ("google/umt5-xxl/special_tokens_map.json", "google/umt5-xxl/spiece.model",
                                         "google/umt5-xxl/tokenizer.json", "google/umt5-xxl/tokenizer_config.json")},
}

_FACTORIES = {"longwam.runtime.create_longwam": "wam_interp.models.longwam.runtime.create_longwam"}
_DTYPES = {"bf16": "bfloat16", "fp16": "float16", "no": "float32",
           "bfloat16": "bfloat16", "float16": "float16", "float32": "float32"}


def migrate_config_names(value):
    """Read old resolved YAML without rewriting tensor keys or filesystem paths."""
    if isinstance(value, Mapping):
        out = {}
        for key, item in value.items():
            name = "longwam_" + key[6:] if key.startswith("arwam_") else key
            out[name] = migrate_config_names(item)
        return out
    if isinstance(value, list):
        return [migrate_config_names(item) for item in value]
    if not isinstance(value, str):
        return value
    if value.startswith("fastwam."):
        value = "longwam." + value[len("fastwam.") :]
        replacements = {
            "create_arwam": "create_longwam",
            "create_fastwam_joint": "create_longwam_joint",
            "create_fastwam_idm": "create_longwam_idm",
            "create_fastwam": "create_longwam_base",
            "processors.fastwam_processor.FastWAMProcessor": "processors.longwam_processor.LongWAMProcessor",
            "wan22.ar_wam.ARWAM": "wan22.long_wam.LongWAM",
            "wan22.fastwam.FastWAM": "wan22.longwam_base.LongWAMBase",
        }
        for old, new in replacements.items():
            value = value.replace(old, new)
    return value.replace("${model.arwam_", "${model.longwam_")


def load_run_config(path, *, domain=None):
    """Upstream ``load_run_config`` (custom OmegaConf resolvers are not registered;
    the released bundle configs contain no interpolations)."""
    from omegaconf import OmegaConf

    raw = OmegaConf.load(Path(path).expanduser().resolve(strict=True))
    cfg = OmegaConf.create(migrate_config_names(OmegaConf.to_container(raw, resolve=False)))
    # Resolve architecture before selecting one domain of a replay mixture.
    cfg.model = OmegaConf.create(OmegaConf.to_container(cfg.model, resolve=True))
    if "domains" in cfg.data.train:
        if domain not in cfg.data.train.domains:
            raise ValueError(f"Choose an inference domain from {list(cfg.data.train.domains)}")
        node = OmegaConf.to_container(cfg.data.train.domains[domain], resolve=True)
        cfg.data.train = OmegaConf.create(node)
    return cfg


def model_config_for_inference(cfg, *, load_text_encoder=True):
    from omegaconf import OmegaConf

    model_cfg = OmegaConf.create(OmegaConf.to_container(cfg.model, resolve=True))
    model_cfg.load_text_encoder = bool(load_text_encoder)
    model_cfg.skip_dit_load_from_pretrain = True
    if model_cfg.get("_target_") == "longwam.runtime.create_longwam":
        model_cfg.skip_video_dit_load_from_pretrain = True
        model_cfg.skip_action_dit_load_from_pretrain = True
        model_cfg.longlive_video_weights = None
    model_cfg.action_dit_pretrained_path = None
    return model_cfg


def setup_model_paths() -> None:
    """Upstream paths.setup_model_paths: where the Wan VAE/T5/tokenizer assets live."""
    os.environ.setdefault("DIFFSYNTH_MODEL_BASE_PATH", os.environ.get("LONGWAM_MODEL_ROOT", "./models"))
    os.environ.setdefault("DIFFSYNTH_DOWNLOAD_SOURCE", "huggingface")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _torch_dtype(name: str):
    import torch

    try:
        return getattr(torch, _DTYPES[str(name).strip().lower()])
    except KeyError as exc:
        raise ValueError(f"Unsupported dtype/mixed_precision {name!r}; choose from {sorted(_DTYPES)}") from exc


def load_longwam(config, checkpoint, *, device: str = "cuda", dtype: str | None = None,
                 load_text_encoder: bool = True, allow_partial_checkpoint: bool = False, domain: str | None = None):
    """Build the bundle's model and load ``model.pt`` strictly; ``dtype`` defaults to the bundle's ``mixed_precision``."""
    checkpoint = Path(checkpoint).expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Long-WAM checkpoint not found: {checkpoint}")
    config_path = None
    if isinstance(config, (str, Path)):
        config_path = Path(config).expanduser().resolve()
        run = load_run_config(config_path, domain=domain)
    else:
        run = config
    import torch
    from hydra.utils import instantiate
    from omegaconf import OmegaConf

    from wam_interp.models.checkpoints import check_checkpoint_loads, record_state_dict_loads

    if str(device).startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; no implicit CPU fallback for Long-WAM inference")
    model_past = int(run.model.get("longwam_past_obs_size", 0))
    data_past = int(run.data.train.get("past_obs_size", 0) or 0)
    if model_past != data_past:
        raise ValueError(f"Checkpoint model ({model_past}) and data ({data_past}) history lengths disagree")
    if int(run.model.get("longwam_num_imagine_frames", 0)) <= 0:
        raise ValueError("IDM and CodeDenoise require future-video latents (longwam_num_imagine_frames > 0)")
    model_cfg = model_config_for_inference(run, load_text_encoder=load_text_encoder)
    target = model_cfg.get("_target_")
    if target not in _FACTORIES:
        raise ValueError(f"Unsupported Long-WAM model target {target!r}; supported: {sorted(_FACTORIES)}")
    model_cfg._target_ = _FACTORIES[target]
    torch_dtype = _torch_dtype(dtype if dtype is not None else run.get("mixed_precision", "bf16"))

    setup_model_paths()
    model = instantiate(model_cfg, model_dtype=torch_dtype, device=device)
    modules = {"mot": model.mot, "proprio_encoder": getattr(model, "proprio_encoder", None)}
    with record_state_dict_loads(modules) as loads:
        payload = model.load_checkpoint(str(checkpoint), strict=not allow_partial_checkpoint)
    load_summary = check_checkpoint_loads(loads, model, checkpoint, allow_partial=allow_partial_checkpoint)
    step, stored_dtype = payload.get("step"), payload.get("torch_dtype")
    del payload
    model = model.to(device).eval()

    config_sha = None if config_path is None else _sha256(config_path)
    released = [name for name, bundle in RELEASED_BUNDLES.items() if bundle["sha256"]["config.yaml"] == config_sha]
    model._wam_interp_metadata = {
        "backend": "longwam",
        "upstream_repository": UPSTREAM_REPOSITORY,
        "upstream_subdirectory": UPSTREAM_SUBDIRECTORY,
        "upstream_revision": UPSTREAM_REVISION,
        "run_config": None if config_path is None else str(config_path),
        "run_config_sha256": config_sha,
        "released_bundle": released[0] if released else None,
        "checkpoint": str(checkpoint),
        "checkpoint_size_bytes": checkpoint.stat().st_size,
        "checkpoint_load": load_summary,
        "checkpoint_strict": not allow_partial_checkpoint,
        "checkpoint_step": step,
        "checkpoint_torch_dtype": stored_dtype,
        "inference_mode": "codenoise" if bool(model.joint_denoise) else "idm",
        "resolved_model_config": OmegaConf.to_container(model_cfg, resolve=True),
        "data_config": OmegaConf.to_container(run.data.train, resolve=True),
        "evaluation": {"num_inference_steps": run.get("eval_num_inference_steps"),
                       "mixed_precision": run.get("mixed_precision")},
        "model_dtype": str(torch_dtype),
        "model_device": str(device),
        "text_encoder_loaded": bool(load_text_encoder),
        "asset_paths": {key: None if value is None else str(value)
                        for key, value in dict(getattr(model, "model_paths", {}) or {}).items()},
        "asset_environment": {key: os.environ.get(key) for key in ("DIFFSYNTH_MODEL_BASE_PATH", "DIFFSYNTH_DOWNLOAD_SOURCE")},
    }
    return model


@register_model_loader("longwam")
def load_longwam_backend(*, bundle=None, config=None, checkpoint=None, device: str = "cuda",
                         dtype: str | None = None, load_text_encoder: bool = True,
                         allow_partial_checkpoint: bool = False) -> Any:
    """``bundle`` is a downloaded bundle directory (config.yaml + model.pt); or pass both files."""
    if bundle is not None:
        bundle = Path(bundle).expanduser()
        config = bundle / "config.yaml" if config is None else config
        checkpoint = bundle / "model.pt" if checkpoint is None else checkpoint
    if config is None or checkpoint is None:
        raise ValueError("Pass bundle=<dir> or both config=<config.yaml> and checkpoint=<model.pt>")
    # Reject missing files before importing Hydra or allocating model assets.
    for path in (config, checkpoint):
        if not Path(path).expanduser().is_file():
            raise FileNotFoundError(f"Long-WAM bundle file not found: {path}")
    return load_longwam(config, checkpoint, device=device, dtype=dtype, load_text_encoder=load_text_encoder,
                        allow_partial_checkpoint=allow_partial_checkpoint)
