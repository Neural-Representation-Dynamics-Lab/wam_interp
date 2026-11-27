"""Compare the recorded executed-step changes between ID and OOD normal runs.

Per episode: mean |change| over its calls (all dimensions, and per dimension), the mean signed change
per dimension, and the share of calls with a decoded gripper flip. Then the mean over episodes per arm,
with a bootstrap 95% interval and the OOD - ID gap. Standard library only.
"""

import json
import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "fasterwam_e7_pilot"))
from analyze import boot_diff, boot_mean, mean  # noqa: E402

DIMS = ("x", "y", "z", "rx", "ry", "rz", "gripper")


def episodes(arm, probe):
    path = sorted((HERE / arm / "normal_deltas").glob("*.json"))[-1]
    report = json.loads(path.read_text())
    out = []
    for trial in report["trials"]:
        calls = [call["future_ablation"] for call in trial["inference_metadata"]]
        deltas = [c["delta"][probe] for c in calls]
        cells = [[abs(v) for v in row] for d in deltas for row in d]
        out.append({
            "success": trial["success"],
            "category": trial.get("category"),
            "abs": mean([v for row in cells for v in row]),
            "abs_dim": [mean([row[k] for row in cells]) for k in range(7)],
            "signed_dim": [mean([row[k] for d in deltas for row in d]) for k in range(7)],
            "flip_share": mean([c["gripper_flips"][probe] > 0 for c in calls]),
            "calls": len(calls)})
    return report.get("completed", False), out


def main():
    rng = random.Random(0)
    summary = {}
    for probe in ("future1_mask_90", "current_mask_90"):
        arms = {arm: episodes(arm, probe) for arm in ("id", "ood")}
        block = {"completed": {arm: done for arm, (done, _) in arms.items()}}
        for key in ("abs", "flip_share"):
            values = {arm: [e[key] for e in eps] for arm, (_, eps) in arms.items()}
            block[key] = {arm: {"n": len(v), "mean": mean(v), "ci95": boot_mean(v, rng)} for arm, v in values.items()}
            block[key]["ood_minus_id"] = {"mean": mean(values["ood"]) - mean(values["id"]),
                                          "ci95": boot_diff(values["ood"], values["id"], rng)}
        for key in ("abs_dim", "signed_dim"):
            block[key] = {arm: dict(zip(DIMS, (mean([e[key][k] for e in eps]) for k in range(7))))
                          for arm, (_, eps) in arms.items()}
        # Same comparison on episodes the normal run solved, since failed episodes run longer.
        solved = {arm: [e["abs"] for e in eps if e["success"]] for arm, (_, eps) in arms.items()}
        block["abs_solved"] = {arm: {"n": len(v), "mean": mean(v)} for arm, v in solved.items()}
        block["abs_solved"]["ood_minus_id_ci95"] = boot_diff(solved["ood"], solved["id"], rng)
        summary[probe] = block
    (HERE / "deltas_summary.json").write_text(json.dumps(summary, indent=2))
    json.dump(summary, sys.stdout, indent=1)


if __name__ == "__main__":
    main()
