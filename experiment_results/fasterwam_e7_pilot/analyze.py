"""Summarise the E7 pilot from the result JSONs in {id,ood}/<condition>/ next to this file.

Writes summary.json here. Standard library only: python analyze.py
"""

import json
import random
import sys
from collections import defaultdict
from math import comb, sqrt
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CONDITIONS = ("normal", "current_mask", "future_mask", "future1_mask", "future2_mask", "resample")
PROBES = ("normal", "resample", "current_mask", "future1_mask", "future2_mask", "future_mask")
PARTS = ("executed", "late", "chunk")
BOOT = 10_000


def load(arm, condition):
    files = sorted((ROOT / arm / condition).glob("*.json"))
    if not files:
        return None
    report = json.loads(files[-1].read_text())
    key = (lambda t: (t["task_id"], t["trial_index"])) if arm == "id" else (lambda t: t["task_id"])
    return {"completed": report.get("completed", False), "path": str(files[-1].relative_to(ROOT)),
            "config": report["config"], "trials": {key(t): t for t in report["trials"]}}


def wilson(k, n, z=1.96):
    if n == 0:
        return None
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return centre - half, centre + half


def mcnemar(b, c):
    n = b + c
    if n == 0:
        return 1.0
    return min(1.0, 2 * sum(comb(n, k) for k in range(min(b, c) + 1)) / 2 ** n)


def mean(values):
    return sum(values) / len(values)


def boot_mean(values, rng):
    means = sorted(mean(rng.choices(values, k=len(values))) for _ in range(BOOT))
    return means[int(0.025 * BOOT)], means[int(0.975 * BOOT)]


def boot_diff(a, b, rng):
    diffs = sorted(mean(rng.choices(a, k=len(a))) - mean(rng.choices(b, k=len(b))) for _ in range(BOOT))
    return diffs[int(0.025 * BOOT)], diffs[int(0.975 * BOOT)]


def call_diffs(call, horizon, executed):
    """Per-probe mean |action difference| over the executed steps, the later unexecuted steps and the chunk."""
    out = {}
    for probe, d in call.items():
        late = (horizon * d["chunk"] - executed * d["executed"]) / (horizon - executed)
        out[probe] = {"executed": d["executed"], "late": late, "chunk": d["chunk"]}
    return out


def main():
    rng = random.Random(0)
    runs = {arm: {c: load(arm, c) for c in CONDITIONS} for arm in ("id", "ood")}
    summary = {"files": {arm: {c: r and {"path": r["path"], "completed": r["completed"], "n": len(r["trials"])}
                               for c, r in runs[arm].items()} for arm in runs}}

    # Success and paired comparison with normal, on the units both runs have.
    summary["paired"] = {}
    for arm, by_cond in runs.items():
        normal = by_cond["normal"]["trials"]
        rows = {}
        for c, run in by_cond.items():
            if run is None:
                continue
            units = [u for u in run["trials"] if u in normal]
            n_ok = sum(normal[u]["success"] for u in units)
            c_ok = sum(run["trials"][u]["success"] for u in units)
            b = sum(normal[u]["success"] and not run["trials"][u]["success"] for u in units)
            d = sum(run["trials"][u]["success"] and not normal[u]["success"] for u in units)
            rows[c] = {"n": len(units), "normal": n_ok, "condition": c_ok, "condition_wilson95": wilson(c_ok, len(units)),
                       "normal_only": b, "condition_only": d, "mcnemar_p": mcnemar(b, d),
                       "relative_drop": None if n_ok == 0 else (n_ok - c_ok) / n_ok,
                       "max_steps_failures": sum(run["trials"][u]["termination_reason"] == "max_steps"
                                                 for u in units if not run["trials"][u]["success"])}
        summary["paired"][arm] = rows

    # Future effect minus size-matched current-frame effect = mean(current - future) per unit.
    summary["future_minus_current"] = {}
    for future in ("future1_mask", "future2_mask", "future_mask"):
        per_arm = {}
        for arm, by_cond in runs.items():
            cur, fut = by_cond["current_mask"], by_cond[future]
            if cur is None or fut is None:
                continue
            units = [u for u in cur["trials"] if u in fut["trials"]]
            values = [int(cur["trials"][u]["success"]) - int(fut["trials"][u]["success"]) for u in units]
            if values:
                per_arm[arm] = {"n": len(values), "mean": mean(values), "ci95": boot_mean(values, rng),
                                "values": values}
        if len(per_arm) == 2:
            per_arm["ood_minus_id"] = {"mean": per_arm["ood"]["mean"] - per_arm["id"]["mean"],
                                       "ci95": boot_diff(per_arm["ood"]["values"], per_arm["id"]["values"], rng)}
        for arm in ("id", "ood"):
            per_arm.get(arm, {}).pop("values", None)
        summary["future_minus_current"][future] = per_arm

    # OOD success per factor and level.
    cells = defaultdict(dict)
    for c, run in runs["ood"].items():
        if run is None:
            continue
        counts = defaultdict(lambda: [0, 0])
        for t in run["trials"].values():
            cell = f"{t['category']}:{t['difficulty_level']}"
            counts[cell][0] += t["success"]
            counts[cell][1] += 1
        for cell, (ok, n) in counts.items():
            cells[cell][c] = f"{ok}/{n}"
    summary["ood_cells"] = dict(sorted(cells.items()))

    # Open-loop action change along the normal runs: per-call mean |diff|, averaged per episode, then over
    # episodes. "late" is the part of the chunk after the executed steps, which is never executed.
    summary["probes"], summary["probes_ci95"] = {}, {}
    episodes = {}
    for arm in ("id", "ood"):
        config = runs[arm]["normal"]["config"]
        horizon, executed = config["action_horizon"], config["replan_steps"]
        groups = defaultdict(lambda: defaultdict(list))
        for t in runs[arm]["normal"]["trials"].values():
            calls = [call_diffs(m["future_ablation"]["action_abs_diff"], horizon, executed)
                     for m in t["inference_metadata"] if "action_abs_diff" in m.get("future_ablation", {})]
            for probe in PROBES:
                for part in PARTS:
                    episode = mean([call[probe][part] for call in calls])
                    groups["all"][f"{probe}:{part}"].append(episode)
                    if arm == "ood":
                        groups[t["category"]][f"{probe}:{part}"].append(episode)
        summary["probes"][arm] = {g: {k: mean(v) for k, v in d.items()} for g, d in groups.items()}
        summary["probes_ci95"][arm] = {k: boot_mean(v, rng) for k, v in groups["all"].items()}
        episodes[arm] = groups["all"]
    summary["probes_ci95"]["ood_minus_id"] = {k: boot_diff(episodes["ood"][k], episodes["id"][k], rng)
                                              for k in episodes["id"]}

    (ROOT / "summary.json").write_text(json.dumps(summary, indent=2))
    json.dump(summary, sys.stdout, indent=1)


if __name__ == "__main__":
    main()
