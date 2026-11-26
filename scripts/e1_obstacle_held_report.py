"""Score the frozen obstacle replication against a development-only side template."""

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from e1_obstacle_report import summarize
from wam_interp.experiments.clip_study import save


def report(manifest, development, held):
    spec = json.loads(manifest.read_text())
    if spec["split"] != "confirmation" or spec["reasons"]:
        raise ValueError("Expected an eligible frozen design")
    backend = spec["backend"]
    ref = []
    for idx in spec["side_template_states"]:
        v = np.load(
            development / backend / f"task_0/state_{idx}/seed_0/side_difference.npy"
        ).reshape(-1)
        ref.append(v / np.linalg.norm(v))
    template = np.mean(ref, axis=0)
    template /= np.linalg.norm(template)
    rows = summarize(held / backend)["rows"]
    by_id = {(r["task"], r["state"], r["seed"]): r for r in rows}
    res = {
        "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
        "backend": backend,
        "tasks": {},
    }
    for t in spec["tasks"]:
        tasks = []
        for idx in spec["states"][str(t)]:
            for seed in spec["seeds"]:
                row = by_id.get(
                    (t, idx, seed),
                    {"task": t, "state": idx, "seed": seed, "missing": True},
                )
                side = (
                    held
                    / backend
                    / f"task_{t}/state_{idx}/seed_{seed}/side_difference.npy"
                )
                if side.is_file():
                    v = np.load(side).reshape(-1)
                    n = np.linalg.norm(v)
                    row["held_side_cosine"] = float(v @ template / n) if n else 0.0
                tasks.append(row)
        good = [r for r in tasks if r.get("behavior_pass")]
        states = sorted({r["state"] for r in good})
        cs = [r["held_side_cosine"] for r in tasks if "held_side_cosine" in r]
        minimum = max(
            spec["planned_usable_per_task"],
            math.ceil(spec["coverage_floor_per_task"] * len(spec["states"][str(t)])),
        )
        enough = (
            len(states) >= minimum
            and bool(cs)
            and np.mean(np.asarray(cs) > 0) >= spec["held_side_accuracy_floor"]
        )
        stats = {
            "rows": tasks,
            "eligible_starts": len(states),
            "attempted_starts": len(spec["states"][str(t)]),
            "required_usable": minimum,
            "side_correct": sum(v > 0 for v in cs),
            "side_total": len(cs),
            "meets_registered_checks": bool(enough),
        }
        if states:
            rng = np.random.default_rng(20261126)
            ids = rng.integers(len(states), size=(10000, len(states)))
            for key in (
                "future_only",
                "live_only",
                "both_minus_live",
                "off_route",
                "future_only_mm_at_8",
                "future_only_displacement_mm_at_16",
                "off_route_displacement_mm_at_16",
            ):
                values = np.array(
                    [np.mean([r[key] for r in good if r["state"] == i]) for i in states]
                )
                draws = values[ids].mean(1)
                stats[key] = {
                    "mean_mm": float(values.mean()),
                    "ci95_mm": np.quantile(draws, [0.025, 0.975]).tolist(),
                }
                if key in ("off_route", "off_route_displacement_mm_at_16"):
                    ci = np.quantile(draws, [0.05, 0.95])
                    stats[key]["ci90_mm"] = ci.tolist()
                    stats[key]["within_margin"] = bool(
                        enough
                        and np.all(np.abs(ci) < spec["off_route_equivalence_margin_mm"])
                    )
        res["tasks"][str(t)] = stats
    return res


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("manifest", "development", "held", "out"):
        p.add_argument("--" + name, required=True, type=Path)
    a = p.parse_args()
    save(a.out, report(a.manifest, a.development, a.held))
