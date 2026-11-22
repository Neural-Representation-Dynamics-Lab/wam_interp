"""Analyze exported manual annotations; optionally combine per-state E1 reports."""
import argparse
import json
from pathlib import Path
import numpy as np
from wam_interp.experiments.annotation import analyze_annotations


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--annotations", type=Path, nargs="*", default=[])
    p.add_argument("--e1-summaries", type=Path, nargs="*", default=[])
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if not args.annotations and not args.e1_summaries: p.error("Supply annotations or E1 summaries")
    report = {"trajectories": [analyze_annotations(path) for path in args.annotations]}
    groups = {}
    flags = set()
    comparison_ids = set()
    seen_runs = set()
    for path in args.e1_summaries:
        if path.resolve() in seen_runs: raise ValueError("Duplicate E1 summary")
        seen_runs.add(path.resolve())
        value = json.loads(path.read_text())
        if "comparison_id" not in value: raise ValueError("Summary lacks comparison provenance; regenerate it")
        comparison_ids.add((value["comparison_id"], value["expected_sign"]))
        if len(comparison_ids) > 1: raise ValueError("Cannot aggregate incompatible E1 protocols or effect directions")
        flags.add(value["synthetic"])
        groups.setdefault(value["state_id"], []).append(value["opposite_minus_same"])
    flags.update(item["synthetic"] for item in report["trajectories"])
    if len(flags) > 1: raise ValueError("Do not combine synthetic and real results")
    report["synthetic"] = next(iter(flags))
    if groups:
        means = {key: float(np.mean(value)) for key, value in groups.items()}
        report["e1"] = {"per_state_mean_effect": means, "independent_states": len(means),
            "mean_effect": float(np.mean(list(means.values()))),
            "note": "Descriptive only. Combine only identical property, units and protocol; no significance claim."}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as f: json.dump(report, f, indent=2, allow_nan=False)
    print(json.dumps(report, indent=2))


if __name__ == "__main__": main()
