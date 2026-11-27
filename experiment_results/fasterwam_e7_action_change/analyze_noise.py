"""Relative drop of each condition against the pilot normal run, ID and OOD, paired on states and seeds.

Compares the future1_mask_90 drop with the drop from adding recorded future1_mask_90 changes to the
normal chunk (no mask), from the ID bank and the OOD bank, drawn per call (inject_*) or replayed in
call order from one solved bank episode (seq_*). The normal runs that recorded the banks are
a second normal sample and give the run-to-run flips. Writes noise_summary.json. Standard library only.
"""

import json
import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PARTIAL = HERE.parent / "fasterwam_e7_partial"
PILOT = HERE.parent / "fasterwam_e7_pilot"
sys.path.insert(0, str(PARTIAL))
from analyze_partial import boot, boot_gap, relative_drop  # noqa: E402
from analyze import mcnemar, wilson  # noqa: E402  (pilot helpers, on the path via analyze_partial)

SOURCES = {"normal_rerun": HERE, "inject_idbank": HERE, "inject_oodbank": HERE, "seq_idbank": HERE,
           "seq_oodbank": HERE, "future1_mask_90": PARTIAL, "current_mask_90": PARTIAL}
FOLDER = {"normal_rerun": "normal_deltas"}


def load(arm, condition):
    files = sorted((SOURCES.get(condition, PILOT) / arm / FOLDER.get(condition, condition)).glob("*.json"))
    if not files:
        return None
    report = json.loads(files[-1].read_text())
    key = (lambda t: (t["task_id"], t["trial_index"])) if arm == "id" else (lambda t: t["task_id"])
    return {"completed": report.get("completed", False), "trials": {key(t): t["success"] for t in report["trials"]}}


def main():
    rng = random.Random(0)
    summary, pairs = {"paired": {}, "ood_minus_id": {}}, {}
    for arm in ("id", "ood"):
        normal = load(arm, "normal")["trials"]
        rows = {}
        for c in SOURCES:
            run = load(arm, c)
            if run is None:
                continue
            p = [(int(normal[u]), int(ok)) for u, ok in run["trials"].items() if u in normal]
            pairs[arm, c] = p
            b, d = sum(n and not k for n, k in p), sum(k and not n for n, k in p)
            ok = sum(k for _, k in p)
            rows[c] = {"n": len(p), "completed": run["completed"], "condition": ok,
                       "condition_wilson95": wilson(ok, len(p)), "normal_only": b, "condition_only": d,
                       "mcnemar_p": mcnemar(b, d), "relative_drop": relative_drop(p),
                       "relative_drop_ci95": boot(relative_drop, p, rng)}
        summary["paired"][arm] = rows
    for c in SOURCES:
        if ("id", c) in pairs and ("ood", c) in pairs:
            summary["ood_minus_id"][c] = {
                "relative_drop": relative_drop(pairs["ood", c]) - relative_drop(pairs["id", c]),
                "ci95": boot_gap(relative_drop, pairs["ood", c], pairs["id", c], rng)}
    (HERE / "noise_summary.json").write_text(json.dumps(summary, indent=2))
    json.dump(summary, sys.stdout, indent=1)


if __name__ == "__main__":
    main()
