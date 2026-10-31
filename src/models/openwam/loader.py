"""Load a released OpenWAM checkpoint directory strictly through upstream's loader."""

import io
import os
import pickle
import re
import tempfile
from pathlib import Path

from wam_interp.models.registry import register_model_loader

UPSTREAM_REVISION = "3950b6d9ea848dc980a82a89b4554af2c082544a"
_VENDORED = "wam_interp.models.openwam.model_code"
_CLASS_PATH = re.compile(r"^openwam\.(model|deploy|dataloader)\.")
_DTYPES = {"bfloat16": "bf16", "float16": "fp16", "float32": "no"}
# normalization_stats.npy is a pickled dict; only numpy array reconstruction is allowed.
_STATS_GLOBALS = {("numpy._core.multiarray", "_reconstruct"), ("numpy.core.multiarray", "_reconstruct"),
                  ("numpy", "ndarray"), ("numpy", "dtype")}


# Upstream configs/deploy.yaml, the setting used for OpenWAM's published benchmark numbers.
UPSTREAM_DEPLOY_OPTIMIZATIONS = {
    "dit_cache": {"enabled": True, "cosine_threshold": 0.99, "max_skips": 3},
    "compile": {"enabled": True,
                "self_attn": {"torch_mode": "default", "dynamic": False},
                "cross_attn": {"torch_mode": "default", "dynamic": False},
                "idm": {"torch_mode": "reduce-overhead", "dynamic": False,
                        "video_loop": {"enabled": True}, "action_cache": {"enabled": True}},
                "tri_system": {"torch_mode": "reduce-overhead", "dynamic": False}},
}


def research_config_overrides(*, denoise_steps=10, decode_video=False, prompt_embed_cache=True,
                              upstream_optimizations=False):
    """Deploy settings: DiT cache and compile off so hooks always run; ``upstream_optimizations=True`` uses upstream's."""
    import copy

    optimization = {"decode_video": bool(decode_video), "dit_cache": {"enabled": False},
                    "compile": {"enabled": False},
                    "prompt_embed_cache": {"enabled": bool(prompt_embed_cache), "maxsize": 32}}
    if upstream_optimizations:
        optimization.update(copy.deepcopy(UPSTREAM_DEPLOY_OPTIMIZATIONS))
    return {
        "inference": {"denoise_steps": int(denoise_steps), "denoise_mode": "sync", "lead_modality": "video",
                      "variance_shift_alpha": 1.0, "linear_offset": 0.0},
        "optimization": optimization,
    }


class _StatsUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if (module, name) not in _STATS_GLOBALS:
            raise pickle.UnpicklingError(f"normalization_stats.npy references {module}.{name}; refusing to load")
        return super().find_class(module, name)


def read_normalization_stats(path):
    """Read normalization_stats.npy without executing arbitrary pickle code."""
    import numpy as np

    stream = io.BytesIO(Path(path).read_bytes())
    version = np.lib.format.read_magic(stream)
    read_header = np.lib.format.read_array_header_1_0 if version == (1, 0) else np.lib.format.read_array_header_2_0
    shape, _fortran, dtype = read_header(stream)
    if dtype != np.dtype(object) or shape != ():
        raise ValueError(f"{path} must hold one pickled dict, got dtype={dtype} shape={shape}")
    value = _StatsUnpickler(stream).load()
    value = value.item() if hasattr(value, "item") else value
    if not isinstance(value, dict):
        raise ValueError(f"{path} must hold a dict of statistics, got {type(value).__name__}")
    return value


def _rewrite_class_paths(node):
    if isinstance(node, dict):
        return {key: _rewrite_class_paths(value) for key, value in node.items()}
    if isinstance(node, list):
        return [_rewrite_class_paths(value) for value in node]
    if isinstance(node, str) and _CLASS_PATH.match(node):
        return _VENDORED + node[len("openwam"):]
    return node


def _video_num_frames(dataloader):
    """Upstream deploy/server.py::_infer_video_num_frames."""
    frames = int(dataloader.get("num_frames", 33))
    stride = int(dataloader.get("video_stride", 1) or 1)
    return (frames - 1) // (stride if stride > 0 else 1) + 1


def merge_deploy_config(training_cfg, overrides):
    """Upstream deploy/server.py::merge_deploy_cfg: inference shapes from the dataloader, then overrides."""
    from omegaconf import OmegaConf

    deploy = OmegaConf.create(overrides)
    dl = OmegaConf.select(training_cfg, "dataloader", default=None)
    if dl is not None:
        dl = OmegaConf.to_container(dl, resolve=True)
        for key, value in (("num_frames", dl.get("num_frames", 33)), ("video_num_frames", _video_num_frames(dl)),
                           ("height", dl.get("height", 384)), ("width", dl.get("width", 320))):
            if OmegaConf.select(deploy, f"inference.{key}", default=None) is None:
                OmegaConf.update(deploy, f"inference.{key}", value, merge=False)
    return OmegaConf.merge(training_cfg, deploy)


def _record_architecture_loads():
    """Record load_state_dict results inside upstream's BaseWAMArchitecture.load_checkpoint."""
    from contextlib import contextmanager
    from wam_interp.models.checkpoints import record_state_dict_loads
    from .model_code.model.architectures.base import BaseWAMArchitecture

    @contextmanager
    def recording():
        records, original = [], BaseWAMArchitecture.load_checkpoint

        def load_checkpoint(self, path, strict=True):
            with record_state_dict_loads({"architecture": self}) as calls:
                original(self, path, strict=strict)
            records.extend(calls)

        BaseWAMArchitecture.load_checkpoint = load_checkpoint
        try:
            yield records
        finally:
            BaseWAMArchitecture.load_checkpoint = original
    return recording()


def check_architecture_load(records, architecture, checkpoint):
    """Raise unless the checkpoint filled every architecture weight; return a provenance summary."""
    problems = []
    if len(records) != 1:
        problems.append(f"expected one architecture load_state_dict call, saw {len(records)}")
    for record in records:
        for kind in ("missing_keys", "unexpected_keys"):
            if record[kind]:
                problems.append(f"{len(record[kind])} {kind} (e.g. {', '.join(record[kind][:5])})")
    meta = [name for name, value in architecture.state_dict().items() if value.device.type == "meta"]
    if meta:
        problems.append(f"{len(meta)} tensors left on the meta device (e.g. {', '.join(meta[:5])})")
    if problems:
        raise RuntimeError(f"OpenWAM checkpoint {checkpoint} did not load completely: " + "; ".join(problems))
    dtypes = {}
    for value in architecture.state_dict().values():
        dtypes[str(value.dtype)] = dtypes.get(str(value.dtype), 0) + value.numel()
    return {"tensors": len(architecture.state_dict()), "missing_keys": 0, "unexpected_keys": 0,
            "parameter_count_by_dtype": dtypes}


class _ShadowCheckpoint:
    """Temporary directory mirroring a checkpoint dir with a rewritten config.yaml."""

    def __init__(self, checkpoint_dir, config):
        import yaml

        self._tmp = tempfile.TemporaryDirectory(prefix="openwam_ckpt_")
        self.path = Path(self._tmp.name)
        for entry in Path(checkpoint_dir).iterdir():
            if entry.name != "config.yaml":
                os.symlink(entry.resolve(), self.path / entry.name)
        (self.path / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    def cleanup(self):
        self._tmp.cleanup()


class OpenWAMModel:
    """Loaded checkpoint: upstream architecture + JointInferenceEngine with research settings."""

    def __init__(self, cfg, architecture, engine, metadata):
        self.cfg = cfg
        self.architecture = architecture
        self.engine = engine
        self._wam_interp_metadata = metadata

    @property
    def device(self):
        return self.architecture.device

    @property
    def dtype(self):
        return self.architecture.dtype

    def named_modules(self, *args, **kwargs):
        """Module paths for tracing/interventions are architecture paths, e.g. 'action_backbone.blocks.0'."""
        return self.architecture.named_modules(*args, **kwargs)


def resolve_checkpoint_file(checkpoint_dir, ckpt_name=None):
    checkpoint_dir = Path(checkpoint_dir).expanduser().resolve()
    if ckpt_name is not None:
        path = checkpoint_dir / ckpt_name
        if not path.is_file():
            raise FileNotFoundError(f"OpenWAM checkpoint not found: {path}")
        return path
    steps = []
    for path in checkpoint_dir.glob("checkpoint_step_*.safetensors"):
        match = re.fullmatch(r"checkpoint_step_(\d+)\.safetensors", path.name)
        if match:
            steps.append((int(match.group(1)), path))
    if not steps:
        raise FileNotFoundError(f"No checkpoint_step_<N>.safetensors in {checkpoint_dir}")
    return max(steps)[1]


def load_openwam(checkpoint_dir, *, ckpt_name=None, device="cuda", dtype="bfloat16", denoise_steps=10,
                 decode_video=False, prompt_embed_cache=True, upstream_optimizations=False):
    """Build the upstream architecture and engine from a released checkpoint directory."""
    import yaml

    checkpoint_dir = Path(checkpoint_dir).expanduser().resolve()
    config_path = checkpoint_dir / "config.yaml"
    if not config_path.is_file():
        raise FileNotFoundError(f"config.yaml not found in {checkpoint_dir}")
    checkpoint = resolve_checkpoint_file(checkpoint_dir, ckpt_name)
    if dtype not in _DTYPES:
        raise ValueError(f"Unsupported dtype {dtype!r}; choose {sorted(_DTYPES)}")
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    stats_path = checkpoint_dir / "normalization_stats.npy"
    if stats_path.is_file():
        read_normalization_stats(stats_path)  # upstream later np.load(allow_pickle=True)s this file
    config = _rewrite_class_paths(config)
    backbone = config["model"]["video_backbone"]
    if not backbone.get("components"):
        raise ValueError("This loader supports self-contained checkpoints with model.video_backbone.components")
    backbone["model_path"] = None  # only a tokenizer fallback in the components path; never read Wan base weights
    if isinstance(backbone.get("encoder"), dict):
        backbone["encoder"]["model_path"] = None
    config.setdefault("training", {})["mixed_precision"] = _DTYPES[dtype]

    from omegaconf import OmegaConf
    from .model_code.deploy.engine import JointInferenceEngine
    from .model_code.deploy.model_loader import load_from_checkpoint_dir

    overrides = research_config_overrides(denoise_steps=denoise_steps, decode_video=decode_video,
                                          prompt_embed_cache=prompt_embed_cache,
                                          upstream_optimizations=upstream_optimizations)
    shadow = _ShadowCheckpoint(checkpoint_dir, config)
    try:
        with _record_architecture_loads() as records:
            training_cfg, architecture = load_from_checkpoint_dir(str(shadow.path), device=device,
                                                                  ckpt_name=checkpoint.name)
    finally:
        shadow.cleanup()  # tokenizer, stats and weights are fully read during load
    load_summary = check_architecture_load(records, architecture, checkpoint)
    cfg = merge_deploy_config(training_cfg, overrides)
    engine = JointInferenceEngine(cfg=cfg, architecture=architecture)
    if (engine._dit_cache is not None) != bool(upstream_optimizations):
        raise RuntimeError(f"dit_cache should be {'on' if upstream_optimizations else 'off'} here")
    metadata = {
        "backend": "openwam",
        "checkpoint_dir": str(checkpoint_dir),
        "checkpoint": str(checkpoint),
        "checkpoint_size_bytes": checkpoint.stat().st_size,
        "checkpoint_load": load_summary,
        "upstream_revision": UPSTREAM_REVISION,
        "architecture": type(architecture).__name__,
        "deploy_overrides": overrides,
        "upstream_optimizations": bool(upstream_optimizations),
        # Not resolved: the saved training config carries a Hydra-only ${now:...} interpolation.
        "config": OmegaConf.to_container(cfg, resolve=False),
        "model_dtype": dtype,
        "model_device": str(device),
    }
    return OpenWAMModel(cfg, architecture, engine, metadata)


@register_model_loader("openwam")
def load_openwam_backend(*, checkpoint_dir, ckpt_name=None, device="cuda", dtype="bfloat16", denoise_steps=10,
                         decode_video=False, prompt_embed_cache=True):
    return load_openwam(checkpoint_dir, ckpt_name=ckpt_name, device=device, dtype=dtype,
                        denoise_steps=denoise_steps, decode_video=decode_video,
                        prompt_embed_cache=prompt_embed_cache)
