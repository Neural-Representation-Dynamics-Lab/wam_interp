"""Export numerical obstacle receipts, simulator trajectories and blinded source examples."""

import argparse
import hashlib
import json
import platform
import shutil
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import numpy as np
from e1_obstacle_report import side_scores, summarize
from PIL import Image
from wam_interp.experiments.clip_study import save


def export(root, dest):
    dest.mkdir(parents=True, exist_ok=True)
    # Recreate a report at any stage; incomplete cases stay visibly pending.
    summary = {m: summarize(root / "qualified" / m) for m in ("fastwam", "longwam")}
    for backend in ("lingbot_va", "openwam"):
        summary[backend] = [
            json.loads(p.read_text())
            for p in sorted(
                (root / "cache" / backend).glob("task_*/state_*/seed_*/behavior.json")
            )
        ]
    summary["cache_calibration"] = {}
    for backend in ("lingbot_va", "openwam"):
        rows = []
        directions = {}
        for feature in sorted(
            (root / "cache" / backend).glob("task_*/state_*/seed_*/side_difference.npy")
        ):
            receipt = feature.with_name("behavior.json")
            if not receipt.exists():
                receipt = feature.with_name("calibration.json")
            if not receipt.is_file():
                continue
            case = json.loads(receipt.read_text())
            rows.append(
                {
                    k: case[k]
                    for k in (
                        "task",
                        "state",
                        "seed",
                        "sham_drift_mm",
                        "passed_sham",
                        "kv_side_rms",
                    )
                }
            )
            vec = np.load(feature).reshape(-1)
            norm = np.linalg.norm(vec)
            if norm:
                directions[(case["task"], case["state"], case["seed"])] = vec / norm
        summary["cache_calibration"][backend] = {
            "rows": rows,
            "side_test_cosines": side_scores(directions),
        }
    summary["task3_alternate_geometry"] = {
        m: summarize(root / "task3_matched" / m) for m in ("fastwam", "longwam")
    }
    sched = root / "fasterwam_schedule.json"
    summary["fasterwam"] = json.loads(sched.read_text()) if sched.is_file() else None
    audit = root / "fasterwam_real" / "audit.json"
    if audit.is_file():
        summary["fasterwam"]["released_checkpoint"] = json.loads(audit.read_text())
        shutil.copytree(audit.parent, dest / "fasterwam_real", dirs_exist_ok=True)
    save(dest / "pilot.json", summary)
    examples = []
    for section in (
        "qualified",
        "task3_matched",
        "cache",
        "settled",
        "precontact",
        "pilot",
        "prefix",
        "matched",
    ):
        if not (root / section).is_dir():
            continue
        for p in (root / section).glob("*/*/*/*/*"):
            if p.suffix == ".json" or (
                section in ("qualified", "task3_matched", "cache")
                and p.name == "paths.npz"
            ):
                out = dest / p.relative_to(root)
                out.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(p, out)
    for m in ("fastwam", "longwam"):
        for p in sorted((root / "qualified" / m).glob("*/*/*/sources.json")):
            data = json.loads(p.read_text())
            if not data["passed"]:
                continue
            source = p.with_name("sources.npz")
            with np.load(source, allow_pickle=False) as clips:
                for k in ("clear", "minus", "plus", "off_minus", "off_plus"):
                    token = hashlib.sha256(
                        f"obstacle26/{m}/{data['task']}/{data['state']}/{k}".encode()
                    ).hexdigest()[:12]
                    imgs = [Image.fromarray(v) for v in clips[k]]
                    folder = dest / "review"
                    folder.mkdir(exist_ok=True)
                    imgs[0].save(
                        folder / (token + ".gif"),
                        save_all=True,
                        append_images=imgs[1:],
                        duration=240,
                        loop=0,
                    )
                    examples.append(
                        {
                            "id": token,
                            "backend": m,
                            "task": data["task"],
                            "state": data["state"],
                            "future": k,
                        }
                    )
            break
    save(dest / "review_key.json", examples)
    import torch

    packages = {}
    for name in ("torch", "numpy", "mujoco", "robosuite", "diffusers", "transformers"):
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            packages[name] = None
    save(
        dest / "environment.json",
        {
            "python": platform.python_version(),
            "gpu": torch.cuda.get_device_name() if torch.cuda.is_available() else None,
            "packages_export_environment": packages,
            "precision": "bfloat16",
            "suite": "libero_10",
        },
    )
    for f in (
        "dev.json",
        "qualified.exit",
        "retry.exit",
        "fasterwam_schedule.json",
        "linux_tests.log",
    ):
        if (root / f).is_file():
            shutil.copyfile(root / f, dest / f)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("root", type=Path)
    p.add_argument("dest", type=Path)
    args = p.parse_args()
    export(args.root, args.dest)
