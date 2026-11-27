"""Summarise Fast-WAM normal on the E7 episodes and pair it with the Faster-WAM pilot's normal.

Reads {id,ood}/normal/*.json next to this file (the OOD set comes in two lane files) and the pilot's
normal runs. Writes summary.json and failures.json (the OOD failures, the rescue starting set).
Standard library only.
"""

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
PILOT = HERE.parent / "fasterwam_e7_pilot"
sys.path.insert(0, str(PILOT))
from analyze import mcnemar, wilson  # noqa: E402

EPISODE = ("seed", "initial_state_index", "instruction", "category", "difficulty_level")


def trials(folder, arm):
    key = (lambda t: (t["task_id"], t["trial_index"])) if arm == "id" else (lambda t: t["task_id"])
    out = {}
    for path in sorted(folder.glob("*.json")):
        report = json.loads(path.read_text())
        if not report.get("completed"):
            raise SystemExit(f"{path} did not complete")
        for t in report["trials"]:
            if key(t) in out:
                raise SystemExit(f"{key(t)} appears twice")
            out[key(t)] = t
    return out


def rate(rows):
    k, n = sum(t["success"] for t in rows), len(rows)
    return {"successes": k, "n": n, "wilson95": wilson(k, n)}


def paired(fast, faster):
    if set(fast) != set(faster):
        raise SystemExit("the runs cover different episodes")
    for k in fast:
        if any(fast[k].get(f) != faster[k].get(f) for f in EPISODE):
            raise SystemExit(f"episode {k} differs from the pilot")
    counts = Counter((fast[k]["success"], faster[k]["success"]) for k in fast)
    only_fast, only_faster = counts[(True, False)], counts[(False, True)]
    return {"both": counts[(True, True)], "only_fastwam": only_fast, "only_fasterwam": only_faster,
            "neither": counts[(False, False)], "mcnemar_p": mcnemar(only_fast, only_faster)}


def main():
    summary, failures = {}, []
    for arm in ("id", "ood"):
        fast = trials(HERE / arm / "normal", arm)
        faster = trials(PILOT / arm / "normal", arm)
        block = {"fastwam": rate(fast.values()), "fasterwam": rate(faster.values()),
                 "paired": paired(fast, faster)}
        if arm == "ood":
            cells = defaultdict(list)
            for k, t in fast.items():
                cells[t["category"]].append(k)
            block["by_factor"] = {}
            for category, keys in sorted(cells.items()):
                levels = {str(level): rate([fast[k] for k in keys if fast[k]["difficulty_level"] == level])
                          for level in sorted({fast[k]["difficulty_level"] for k in keys})}
                block["by_factor"][category] = {
                    "fastwam": rate([fast[k] for k in keys]), "fasterwam": rate([faster[k] for k in keys]),
                    "fastwam_by_level": levels,
                    "paired": paired({k: fast[k] for k in keys}, {k: faster[k] for k in keys})}
            for k, t in sorted(fast.items()):
                if not t["success"]:
                    failures.append({"variant": k, "category": t["category"], "level": t["difficulty_level"],
                                     "base_task": t["name"], "seed": t["seed"],
                                     "termination": t["termination_reason"], "policy_steps": t["policy_steps"],
                                     "fasterwam_success": faster[k]["success"]})
        else:
            block["failures"] = [list(k) for k, t in sorted(fast.items()) if not t["success"]]
        summary[arm] = block
    summary["ood"]["failure_terminations"] = dict(Counter(f["termination"] for f in failures))
    (HERE / "summary.json").write_text(json.dumps(summary, indent=2))
    (HERE / "failures.json").write_text(json.dumps(failures, indent=2))
    json.dump(summary, sys.stdout, indent=1)


if __name__ == "__main__":
    main()
