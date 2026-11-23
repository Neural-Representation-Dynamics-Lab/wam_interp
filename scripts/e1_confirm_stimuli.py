"""Quantify visible motion and colour differences in every saved source pair, without action outputs."""

import argparse
import json
from pathlib import Path

import numpy as np


def difference(first, second):
    if first.shape != second.shape or not np.array_equal(first[0], second[0]):
        raise ValueError("Source views must share their anchor and dimensions")
    delta = first[1:].astype(np.float32) - second[1:]
    return {
        "mae_255": float(np.abs(delta).mean()),
        "rms_255": float(np.sqrt(np.square(delta).mean())),
        "changed_fraction": float(np.any(delta != 0, axis=-1).mean()),
    }


def audit(source, robotwin):
    with np.load(source, allow_pickle=False) as z:
        if robotwin:
            cameras = ("head", "left_wrist", "right_wrist")

            def view(key, cam):
                return z[key + "_" + cam]
        else:
            cameras = ("agentview", "wrist")

            def view(key, cam):
                frames = z[key]
                cut = frames.shape[2] // 2
                return frames[:, :, :cut] if cam == "agentview" else frames[:, :, cut:]

        result = {}
        for cam in cameras:
            names = (
                "left_neutral",
                "right_neutral",
                "left_warm",
                "left_cool",
                "right_warm",
                "right_cool",
            )
            raw = {k: view(k, cam) for k in names}
            metrics = {"motion": difference(raw["left_neutral"], raw["right_neutral"])}
            for side in ("left", "right"):
                metrics[f"colour_{side}"] = difference(raw[side + "_warm"], raw[side + "_cool"])
            if "fixed_sham" in z.files:
                metrics["wrist_stabilization_sham"] = difference(
                    view("fixed_sham", cam), view("sham", cam)
                )
            result[cam] = metrics
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True, help="Split directory")
    a = p.parse_args()
    spec = json.loads(a.manifest.read_text())
    prefix = spec.get("source_backend", "fastwam")
    rows = []
    for t in spec["tasks"]:
        for idx in spec["states"][str(t)]:
            for s in spec["seeds"]:
                folder = a.out / f"task_{t}/state_{idx}/seed_{s}"
                src = json.loads((folder / f"{prefix}_source.json").read_text())
                row = {"task": t, "state": idx, "seed": s, "source_pass": src["passed"]}
                if src["passed"]:
                    row["views"] = audit(
                        folder / f"{prefix}_source.npz", spec.get("suite", "").startswith("RoboTwin")
                    )
                    row["source_hash"] = src["sha256"]
                rows.append(row)
    (a.out / "pixel_contrasts.json").write_text(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
