"""Closed-loop partial masks of latent frame 0 and 1, paired with the pilot runs.

Level 0 is the pilot normal run and level 100 the pilot full mask (../fasterwam_e7_pilot), so every level is
paired on the same states and seeds. Writes summary.json next to this file. Standard library only.
"""

import json
import random
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
PILOT = HERE.parent / "fasterwam_e7_pilot"
sys.path.insert(0, str(PILOT))
from analyze import BOOT, boot_mean, mcnemar, mean, wilson  # noqa: E402

LEVELS = (90, 95, 100)
MASKS = ("future1_mask", "current_mask")


def load(arm, condition):
    root = PILOT if condition in ("normal", *MASKS) else HERE
    files = sorted((root / arm / condition).glob("*.json"))
    if not files:
        return None
    report = json.loads(files[-1].read_text())
    key = (lambda t: (t["task_id"], t["trial_index"])) if arm == "id" else (lambda t: t["task_id"])
    return {"completed": report.get("completed", False), "trials": {key(t): t for t in report["trials"]}}


def condition(mask, level):
    return mask if level == 100 else f"{mask}_{level}"


def relative_drop(pairs):
    normal = sum(n for n, _ in pairs)
    return None if normal == 0 else (normal - sum(c for _, c in pairs)) / normal


def boot(stat, values, rng):
    draws = sorted(s for s in (stat(rng.choices(values, k=len(values))) for _ in range(BOOT)) if s is not None)
    return draws[int(0.025 * len(draws))], draws[int(0.975 * len(draws))]


def boot_gap(stat, ood, id_, rng):
    """Bootstrap interval of stat(ood) - stat(id), resampling each arm on its own."""
    gaps = []
    for _ in range(BOOT):
        a, b = stat(rng.choices(ood, k=len(ood))), stat(rng.choices(id_, k=len(id_)))
        if a is not None and b is not None:
            gaps.append(a - b)
    gaps.sort()
    return gaps[int(0.025 * len(gaps))], gaps[int(0.975 * len(gaps))]


def main():
    rng = random.Random(0)
    names = ["normal"] + [condition(m, p) for m in MASKS for p in LEVELS]
    runs = {arm: {c: load(arm, c) for c in names} for arm in ("id", "ood")}
    summary = {"files": {arm: {c: r and {"n": len(r["trials"]), "completed": r["completed"]}
                               for c, r in by_cond.items()} for arm, by_cond in runs.items()}}

    summary["paired"], pairs = {}, defaultdict(dict)
    for arm, by_cond in runs.items():
        normal = by_cond["normal"]["trials"]
        rows = {}
        for c, run in by_cond.items():
            if run is None or c == "normal":
                continue
            units = [u for u in run["trials"] if u in normal]
            pairs[arm][c] = [(int(normal[u]["success"]), int(run["trials"][u]["success"])) for u in units]
            b = sum(n and not k for n, k in pairs[arm][c])
            d = sum(k and not n for n, k in pairs[arm][c])
            ok = sum(k for _, k in pairs[arm][c])
            rows[c] = {"n": len(units), "normal": sum(n for n, _ in pairs[arm][c]), "condition": ok,
                       "condition_wilson95": wilson(ok, len(units)), "normal_only": b, "condition_only": d,
                       "mcnemar_p": mcnemar(b, d), "relative_drop": relative_drop(pairs[arm][c]),
                       "relative_drop_ci95": boot(relative_drop, pairs[arm][c], rng)}
        summary["paired"][arm] = rows

    # Same question as the pilot at each level: does the frame-1 drop differ between ID and OOD, and how
    # does it compare with masking the same positions of the current frame?
    summary["ood_minus_id"], summary["current_minus_future1"] = {}, {}
    for level in LEVELS:
        f1, cur = condition("future1_mask", level), condition("current_mask", level)
        if all(f1 in pairs[arm] for arm in ("id", "ood")):
            summary["ood_minus_id"][f1] = {
                "relative_drop": relative_drop(pairs["ood"][f1]) - relative_drop(pairs["id"][f1]),
                "ci95": boot_gap(relative_drop, pairs["ood"][f1], pairs["id"][f1], rng)}
        per_arm = {}
        for arm, by_cond in runs.items():
            if by_cond[f1] is None or by_cond[cur] is None:
                continue
            units = [u for u in by_cond[f1]["trials"] if u in by_cond[cur]["trials"]]
            values = [int(by_cond[cur]["trials"][u]["success"]) - int(by_cond[f1]["trials"][u]["success"])
                      for u in units]
            per_arm[arm] = {"n": len(values), "mean": mean(values), "ci95": boot_mean(values, rng)}
        summary["current_minus_future1"][str(level)] = per_arm

    cells = defaultdict(dict)
    for c, run in runs["ood"].items():
        if run is None:
            continue
        counts = defaultdict(lambda: [0, 0])
        for t in run["trials"].values():
            counts[t["category"]][0] += t["success"]
            counts[t["category"]][1] += 1
        for category, (ok, n) in counts.items():
            cells[category][c] = f"{ok}/{n}"
    summary["ood_by_factor"] = dict(sorted(cells.items()))

    (HERE / "summary.json").write_text(json.dumps(summary, indent=2))
    json.dump(summary, sys.stdout, indent=1)


if __name__ == "__main__":
    main()
