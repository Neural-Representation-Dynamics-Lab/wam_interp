"""Small shared provenance helpers; importing them does not load simulators."""

from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
import hashlib
import json
import subprocess
from datetime import datetime, timezone


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def checkout_revision(root, expected=None):
    root = Path(root).expanduser().resolve()
    result = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"],
                            capture_output=True, text=True, check=True)
    actual = result.stdout.strip()
    if expected is not None and actual != expected:
        raise RuntimeError(f"Expected benchmark revision {expected}, found {actual} at {root}")
    return actual


def package_versions():
    result = {}
    for name in ("torch", "numpy", "mujoco", "robosuite", "sapien", "mplib", "diffusers", "transformers"):
        try:
            result[name] = version(name)
        except PackageNotFoundError:
            result[name] = None
    return result


def model_provenance(adapter):
    model = getattr(adapter, "model", None)
    try:
        source_revision = checkout_revision(Path(__file__).resolve().parents[2])
    except (OSError, subprocess.CalledProcessError):
        source_revision = None
    transforms = getattr(adapter, "transforms", None)
    statistics = getattr(adapter, "dataset_stats", getattr(transforms, "stats", None))
    statistics_hash = None
    if statistics is not None:
        statistics_hash = hashlib.sha256(json.dumps(statistics, sort_keys=True).encode()).hexdigest()
    return {
        "source_revision": source_revision,
        "model_name": getattr(adapter, "name", type(adapter).__name__),
        "model_dtype": str(getattr(model, "torch_dtype", getattr(model, "dtype", "unknown"))),
        "model_device": str(getattr(model, "device", "unknown")),
        "statistics_content_sha256": statistics_hash,
        **getattr(model, "_wam_interp_metadata", {}),
    }


def synchronize_adapter(adapter):
    device = getattr(getattr(adapter, "model", None), "device", "cpu")
    if str(device).startswith("cuda"):
        import torch
        torch.cuda.synchronize(device)


def save_report(report, output_dir, prefix):
    if output_dir is None:
        return None
    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{prefix}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')}.json"
    report["output_path"] = str(path)
    path.write_text(json.dumps(report, indent=2, sort_keys=True, default=str), encoding="utf-8")
    return str(path)
