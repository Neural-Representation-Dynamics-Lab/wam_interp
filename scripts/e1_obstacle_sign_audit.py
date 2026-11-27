"""Check signed steering against recorded physical box centers without using the scorer."""

import argparse
import json
from pathlib import Path

import numpy as np

CONTRASTS = {
    "future_only": ("clear__{side}", "clear__clear", ""),
    "both_minus_live": ("{side}__{side}", "{side}__clear", ""),
    "live_only": ("{side}__clear", "clear__clear", ""),
    "off_route": ("clear__off_{side}", "clear__clear", "off_"),
}


def check(actual, expected):
    if not np.isclose(actual, expected, rtol=0, atol=1e-9):
        raise ValueError(f"Independent sign check differs: {actual} vs {expected} mm")
    return abs(actual - expected)


def audit(report, qualified):
    data = json.loads(report.read_bytes())
    checks = []
    by_task = {}
    for task, result in data["tasks"].items():
        values = {key: [] for key in CONTRASTS}
        rows = [r for r in result["rows"] if r.get("behavior_pass")]
        if not rows or len({r["state"] for r in rows}) != result["eligible_starts"]:
            raise ValueError("Incomplete eligible-start records")
        for row in rows:
            folder = (
                qualified
                / data["backend"]
                / f"task_{task}/state_{row['state']}/seed_{row['seed']}"
            )
            placements = json.loads((folder / "sources.json").read_bytes())[
                "placements"
            ]
            original = json.loads((folder / "behavior.json").read_bytes())["metrics"]
            with np.load(folder / "paths.npz") as paths:
                for label, (arm, reference, prefix) in CONTRASTS.items():
                    pair = []
                    for side, other in (("minus", "plus"), ("plus", "minus")):
                        center, mirror = [
                            np.array(placements[prefix + s]["geometry"]["center"])[:2]
                            for s in (side, other)
                        ]
                        away = mirror - center
                        if np.linalg.norm(away) == 0:
                            raise ValueError("Mirrored centers coincide")
                        treated, baseline = (
                            paths[arm.format(side=side)],
                            paths[reference.format(side=side)],
                        )
                        delta = (treated[16] - treated[0]) - (
                            baseline[16] - baseline[0]
                        )
                        value = float(
                            1000 * (delta[:2] @ (away / np.linalg.norm(away)))
                        )
                        checks.append(
                            check(value, original[f"{label}_{side}"]["open_side_mm"])
                        )
                        pair.append(value)
                    check(np.mean(pair), row[label])
                    values[label].append((row["state"], pair))
        by_task[task] = {}
        for label, entries in values.items():
            starts = sorted({state for state, _ in entries})
            state_means = [
                np.mean([v for state, v in entries if state == s], axis=0)
                for s in starts
            ]
            sides = np.mean(state_means, axis=0)
            estimate = float(np.mean(sides))
            check(estimate, result[label]["mean_mm"])
            by_task[task][label] = {
                "minus_mm": float(sides[0]),
                "plus_mm": float(sides[1]),
                "mean_mm": estimate,
            }
    return {
        "individual_checks": len(checks),
        "max_error_mm": max(checks),
        "tasks": by_task,
    }


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--report", type=Path, required=True)
    p.add_argument("--qualified", type=Path, required=True)
    args = p.parse_args()
    print(json.dumps(audit(args.report, args.qualified), indent=2, allow_nan=False))
