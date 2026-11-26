"""Summarize development exclusions, per-start steering, and a blinded held-out side decoder."""

import argparse
import json
from pathlib import Path

import numpy as np
from wam_interp.experiments.clip_study import save
from wam_interp.experiments.obstacle import Barrier, score


def side_scores(directions):
    # Task-0 states 0/1 fit a direction; 2/3 test its orientation without using policy actions.
    training = [
        v for (t, i, s), v in directions.items() if t == 0 and i in (0, 1) and s == 0
    ]
    decoder = {}
    if len(training) == 2:
        template = np.mean(training, axis=0)
        for (task, idx, seed), v in directions.items():
            if task == 0 and idx in (2, 3):
                decoder[f"{task}/{idx}/{seed}"] = float(
                    template @ v / max(np.linalg.norm(template), 1e-12)
                )
    return decoder


def summarize(folder):
    rows, directions = [], {}
    for p in sorted(folder.glob("task_*/state_*/seed_*/sources.json")):
        meta = json.loads(p.read_text())
        row = {k: meta[k] for k in ("task", "state", "seed")}
        row["source_pass"] = meta["passed"]
        row["source_issues"] = [
            k for k, v in meta["placements"].items() if not v["passed"]
        ]
        calibration = p.with_name("calibration.json")
        if calibration.is_file():
            record = json.loads(calibration.read_text())
            row["sham_pass"] = record["eligible"]
            row["sham_drift_mm"] = record.get("clear_sham_drift_mm")
        marker = p.with_name("behavior.json")
        if marker.exists():
            data = json.loads(marker.read_text())
            row["behavior_pass"] = data["passed"]
            if "metrics" in data:
                row["sham_drift_mm"] = data["clear_sham_drift_mm"]
                for metric in (
                    "future_only",
                    "both_minus_live",
                    "live_only",
                    "off_route",
                ):
                    sides = [
                        data["metrics"][metric + "_" + s]["open_side_mm"]
                        for s in ("minus", "plus")
                        if metric + "_" + s in data["metrics"]
                    ]
                    row[metric] = float(np.mean(sides)) if len(sides) == 2 else None
                with np.load(p.with_name("paths.npz")) as paths:
                    row["future_only_mm_at_8"] = float(
                        np.mean(
                            [
                                score(
                                    paths["clear__" + side],
                                    paths["clear__clear"],
                                    Barrier(**meta["placements"][side]["geometry"]),
                                    8,
                                )["open_side_mm"]
                                for side in ("minus", "plus")
                            ]
                        )
                    )
        feature = p.with_name("side_difference.npy")
        if feature.is_file():
            a = np.load(feature).reshape(-1)
            n = np.linalg.norm(a)
            if n:
                directions[(row["task"], row["state"], row["seed"])] = a / n
        rows.append(row)
    decoder = side_scores(directions)
    groups = {}
    for task in sorted({r["task"] for r in rows}):
        eligible = [r for r in rows if r["task"] == task and r.get("behavior_pass")]
        groups[str(task)] = {
            "count": len(eligible),
            "attempted": sum(r["task"] == task for r in rows),
        }
        if eligible:
            groups[str(task)]["mean_steering_mm"] = {
                k: float(np.mean([r[k] for r in eligible]))
                for k in ("future_only", "both_minus_live", "live_only", "off_route")
            }
            starts = sorted({r["state"] for r in eligible})
            if len(starts) >= 3:
                per_start = [
                    np.mean([r["future_only"] for r in eligible if r["state"] == i])
                    for i in starts
                ]
                draws = np.random.default_rng(20261126).choice(
                    per_start, size=(10000, len(starts)), replace=True
                )
                groups[str(task)]["development_future_only_ci95_mm"] = np.quantile(
                    draws.mean(1), [0.025, 0.975]
                ).tolist()
    return {
        "rows": rows,
        "by_task": groups,
        "side_test_cosines": decoder,
        "task0_side_check_passed": len(decoder) == 2
        and all(x > 0 for x in decoder.values()),
    }


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("root", type=Path)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    save(a.out, {m: summarize(a.root / m) for m in ("fastwam", "longwam")})
