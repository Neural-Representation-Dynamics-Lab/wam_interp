"""Choose fresh starts and freeze the confirmatory design from independent development receipts."""

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from e1_obstacle_report import summarize


def freeze(root, backend, tasks):
    pilot = summarize(root / backend)
    eligible = [r for r in pilot["rows"] if r.get("behavior_pass")]
    counts = {t: sum(r["task"] == t for r in eligible) for t in tasks}
    reasons = []
    if not pilot["task0_side_check_passed"]:
        reasons.append("Need both independent task-0 side checks")
    if min(counts.values()) < 2:
        reasons.append(
            "Insufficient independent replay-valid starts on each intended task to estimate variability"
        )
    delta = 2.0
    design = {
        "backend": backend,
        "split": "confirmation",
        "tasks": tasks,
        "seeds": [1],
        "development_replay_counts": counts,
        "minimum_relevant_steering_mm": delta,
        "off_route_equivalence_margin_mm": delta,
        "primary_control": 16,
        "cross_model_control": 8,
        "unit": "initial state, mean over mirrored sides",
        "source_station": 28,
        "held_side_accuracy_floor": 0.75,
        "side_template_states": [0, 1],
        "analysis": "mean signed future-only minus clear at control 16; paired cluster bootstrap over starts; 90% equivalence interval for off-route within +/-2 mm",
        "coverage_floor_per_task": 0.75,
        "sham_margin_mm": 20,
        "minimum_retained_controls": 24,
        "power": 0.8,
        "alpha": 0.05,
        "reasons": reasons,
    }
    if reasons:
        return design
    # Use the larger task-specific SD. No balancing of intervention signs after inspecting held outcomes.
    sd = max(
        float(np.std([r["future_only"] for r in eligible if r["task"] == t], ddof=1))
        for t in tasks
    )
    need = max(8, math.ceil(((1.96 + 0.84) * sd / delta) ** 2))
    attempted = math.ceil(need / 0.75)
    candidates = np.arange(8, 20)
    design.update(
        pilot_upper_task_sd_mm=sd,
        planned_usable_per_task=need,
        attempted_per_task=attempted,
    )
    if attempted > len(candidates):
        reasons.append("Requested precision exceeds the reserved independent starts")
        return design
    rng = np.random.default_rng(20261126)
    design["states"] = {
        str(t): rng.choice(candidates, size=attempted, replace=False).tolist()
        for t in tasks
    }
    return design


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--backend", choices=["fastwam", "longwam"], required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--task0-only", action="store_true")
    a = p.parse_args()
    design = freeze(a.root, a.backend, [0] if a.task0_only else [0, 3])
    data = (json.dumps(design, indent=2, sort_keys=True) + "\n").encode()
    if a.out.exists() and a.out.read_bytes() != data:
        raise ValueError(
            "Design already recorded with different inputs; use a separate output"
        )
    a.out.write_bytes(data)
    print(hashlib.sha256(data).hexdigest(), a.out, design["reasons"])
