"""Small shared provenance helpers; importing them does not load simulators."""

from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
import hashlib
import json
import os
import signal
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


def source_state(root):
    """HEAD plus whether the checkout has uncommitted changes (and a hash of them)."""
    root = Path(root).expanduser().resolve()
    status = subprocess.run(["git", "-C", str(root), "status", "--porcelain"],
                            capture_output=True, text=True, check=True).stdout
    diff = subprocess.run(["git", "-C", str(root), "diff", "HEAD"], capture_output=True, check=True).stdout
    return {"revision": checkout_revision(root), "dirty": bool(status.strip()),
            "dirty_paths": len(status.splitlines()),
            "status_and_diff_sha256": hashlib.sha256(status.encode() + diff).hexdigest()}


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
        source = source_state(Path(__file__).resolve().parents[2])
    except (OSError, subprocess.CalledProcessError):
        source = {"revision": None, "dirty": None}
    transforms = getattr(adapter, "transforms", None)
    statistics = getattr(adapter, "dataset_stats", None)
    for name in ("dataset_stats", "stats"):
        if statistics is None:
            statistics = getattr(transforms, name, None)
    statistics_hash = None
    if statistics is not None:
        # Content only: the loader adds the file path as "_source", which must not change the hash.
        content = {key: value for key, value in statistics.items() if key != "_source"}
        statistics_hash = hashlib.sha256(json.dumps(content, sort_keys=True).encode()).hexdigest()
    return {
        "source_revision": source["revision"],
        "source_dirty": source["dirty"],
        "source_state": source,
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


def report_path(output_dir, prefix):
    """Timestamped report path in ``output_dir`` (created), or None when not saving."""
    if output_dir is None:
        return None
    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir / f"{prefix}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')}.json"


def write_report(report, path):
    """Atomically (re)write ``report`` as JSON at ``path``; safe to call after every episode."""
    path = Path(path)
    report["output_path"] = str(path)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True, default=str), encoding="utf-8")
    os.replace(temporary, path)
    return str(path)


def save_report(report, output_dir, prefix):
    path = report_path(output_dir, prefix)
    return None if path is None else write_report(report, path)


def install_termination_handler():
    """Turn SIGTERM (Slurm walltime/preemption) into SystemExit so ``finally`` blocks save results."""
    def handler(signum, _frame):
        raise SystemExit(128 + signum)
    return signal.signal(signal.SIGTERM, handler)
