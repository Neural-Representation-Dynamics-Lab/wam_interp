"""Summarise the scripted-future runs in {id,ood}/rows.json next to this file.

Per start the two action seeds are averaged. Each branch (native future, first-object future,
second-object future) gives the end-effector shift on the tabletop axis from the first object to the
second, as a share of the two objects' separation. "toward" is second minus first, in metres and as a
share. OOD is compared with ID raw and with ID reweighted to the OOD task mix (bootstrap within tasks).
Writes summary.json. Standard library only.
"""

import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "fasterwam_e7_pilot"))
from analyze import boot_diff, boot_mean, mean  # noqa: E402

BRANCHES = ("native", "first", "second")
SHARES = BRANCHES + ("toward_share",)


def starts(arm):
    report = json.loads((HERE / arm / "rows.json").read_text())
    used, rejected = [], Counter()
    for row in report["rows"]:
        if "rejected" in row:
            rejected["approach limits" if row["rejected"].startswith("approach") else "scripted run"] += 1
            continue
        first, second = row["labels"]
        separation = row["approach"][first]["target_xy_separation_m"]
        seeds = list(row["seeds"].values())
        shift = {branch: mean([s["axis_shift_m"][label] for s in seeds]) / separation
                 for branch, label in zip(BRANCHES, ("native", first, second))}
        used.append({"unit": row["unit"], "task": row["base_task"], "category": row.get("category"),
                     "separation_m": separation, **shift,
                     "toward_m": mean([s["toward_shown_m"] for s in seeds]),
                     "toward_share": shift["second"] - shift["first"],
                     "both_closer": mean([all(s["closer_to_shown"].values()) for s in seeds])})
    return report.get("completed", False), used, dict(rejected), len(report["rows"])


def group(rows, key):
    out = defaultdict(list)
    for row in rows:
        out[row[key]].append(row)
    return out


def task_matched_gap(key, id_used, ood_used):
    """OOD mean minus ID mean reweighted to the OOD task counts (OOD tasks missing from ID are dropped)."""
    by_task = group(id_used, "task")
    ood = [u for u in ood_used if u["task"] in by_task]
    weights = Counter(u["task"] for u in ood)
    reweighted = sum(w * mean([u[key] for u in by_task[t]]) for t, w in weights.items()) / len(ood)
    return mean([u[key] for u in ood]) - reweighted


def stratified_boot(key, id_used, ood_used, rng, n=2000):
    id_tasks, ood_tasks = group(id_used, "task"), group(ood_used, "task")
    draws = sorted(task_matched_gap(key,
                                    [rng.choice(v) for v in id_tasks.values() for _ in v],
                                    [rng.choice(v) for v in ood_tasks.values() for _ in v]) for _ in range(n))
    return [draws[int(0.025 * n)], draws[int(0.975 * n)]]


def main():
    rng = random.Random(0)
    arms = {arm: starts(arm) for arm in ("id", "ood") if (HERE / arm / "rows.json").exists()}
    summary = {}
    for arm, (done, used, rejected, total) in arms.items():
        block = {"completed": done, "starts": total, "used": len(used), "rejected": rejected}
        for key in ("toward_m", "both_closer") + SHARES:
            values = [u[key] for u in used]
            block[key] = {"mean": mean(values), "ci95": boot_mean(values, rng)}
        block["by_task"] = {t: {"n": len(v), **{k: mean([u[k] for u in v]) for k in SHARES}}
                            for t, v in sorted(group(used, "task").items())}
        if arm == "ood":
            block["by_factor"] = {c: {"n": len(v), **{k: mean([u[k] for u in v]) for k in SHARES + ("both_closer",)}}
                                  for c, v in sorted(group(used, "category").items())}
        summary[arm] = block
    if len(arms) == 2:
        id_used, ood_used = arms["id"][1], arms["ood"][1]
        gap = {"raw": {}, "task_matched": {}}
        for key in ("toward_m", "both_closer") + SHARES:
            a, b = [u[key] for u in ood_used], [u[key] for u in id_used]
            gap["raw"][key] = {"mean": mean(a) - mean(b), "ci95": boot_diff(a, b, rng)}
        for key in SHARES:
            gap["task_matched"][key] = {"mean": task_matched_gap(key, id_used, ood_used),
                                        "ci95": stratified_boot(key, id_used, ood_used, rng)}
        gap["task_matched"]["ood_starts_without_id_task"] = sum(u["task"] not in group(id_used, "task")
                                                                for u in ood_used)
        summary["ood_minus_id"] = gap
    (HERE / "summary.json").write_text(json.dumps(summary, indent=2))
    json.dump(summary, sys.stdout, indent=1)


if __name__ == "__main__":
    main()
