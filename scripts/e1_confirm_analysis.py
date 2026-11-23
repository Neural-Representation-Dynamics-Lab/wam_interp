"""Freeze a held-out design, identify edits at the intervention site, and score paired behavior."""

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from scipy.stats import binomtest, nct, norm, t

MODELS = ("fastwam", "longwam")
PROPERTIES = ("arm", "colour")
METRICS = (
    "motion_32_mm",
    "motion_10_mm",
    "background_rms_max_mm",
    "signed_background_half_mm",
    "selectivity_mm",
    "sham_drift_mm",
    "depicted_half_mm",
    "following_ratio",
)


def read(p):
    return json.loads(p.read_text(encoding="utf8"))


def write(p, obj, immutable=False):
    data = json.dumps(obj, indent=2, allow_nan=False)
    p.parent.mkdir(parents=True, exist_ok=True)
    if immutable and p.exists():
        if read(p) != obj:
            raise ValueError(f"Frozen inputs or values differ at {p}")
        return
    tmp = p.with_suffix(".tmp")
    tmp.write_text(data, encoding="utf8")
    tmp.replace(p)


def digest(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def cases(manifest, root):
    for task in manifest["tasks"]:
        for idx in manifest["states"][str(task)]:
            yield task, idx, [root / f"task_{task}/state_{idx}/seed_{seed}" for seed in manifest["seeds"]]


def collect(manifest, root, model):
    states = []
    for task, idx, folders in cases(manifest, root):
        seeds = []
        for seed, folder in zip(manifest["seeds"], folders):
            src = read(folder / (manifest.get("source_backend", "fastwam") + "_source.json"))
            item = {"seed": seed, "source_pass": src["passed"]}
            if src.get("attempts"):
                item["stimulus"] = src["attempts"][-1]
            elif "arms" in src:
                item["stimulus"] = src["arms"]
            feature = folder / f"{model}_features.json"
            if feature.exists():
                item["feature_rms"] = read(feature)["rms"]
            if src["passed"]:
                receipt = read(folder / f"{model}_behavior.json")
                if receipt.get("source_hash", src["sha256"]) != src["sha256"]:
                    raise ValueError("Behavior used a different source")
                item.update(receipt)
                if "metrics" in receipt:
                    item["metrics"] = dict(
                        receipt["metrics"],
                        depicted_half_mm=receipt["depicted_half_mm"],
                        following_ratio=receipt["metrics"]["motion_32_mm"] / receipt["depicted_half_mm"],
                    )
            seeds.append(item)
        record = {
            "task": task,
            "state": idx,
            "seeds": seeds,
            "eligible": all(k.get("eligible", False) for k in seeds),
        }
        if record["eligible"]:
            record["means"] = {k: float(np.mean([d["metrics"][k] for d in seeds])) for k in METRICS}
        states.append(record)
    return states


def groups(rows, tasks, k):
    return [
        np.array([row["means"][k] for row in rows if row["eligible"] and row["task"] == task])
        for task in tasks
    ]


def power_count(sd):
    for n in range(3, 3001):
        crit = t.ppf(0.975, n - 1)
        if (
            nct.sf(crit, n - 1, 4 * math.sqrt(n) / sd) + nct.cdf(-crit, n - 1, 4 * math.sqrt(n) / sd)
            >= 0.8
        ):
            return n
    return 3001


def within_sd(values):
    df = sum(max(0, len(x) - 1) for x in values)
    return (
        float(np.sqrt(sum(float(np.sum((x - x.mean()) ** 2)) for x in values if len(x)) / df))
        if df
        else None
    )


def planning_curve(values, attempted, margin, minimum, maximum):
    # Resample entire development starts including failures, independently within task.
    rng = np.random.default_rng(36031)
    curve = {}
    if any(len(g) < 2 for g in values):
        return curve
    expected = float(np.mean([g.mean() for g in values]))
    for size in range(minimum, maximum + 1):
        means, variances, valid = [], [], np.ones(4000, dtype=bool)
        for sample, n in zip(values, attempted):
            frame = np.concatenate((sample, np.full(n - len(sample), np.nan)))
            draw = frame[rng.integers(0, n, size=(4000, size))]
            counts = np.isfinite(draw).sum(axis=1)
            valid &= counts >= max(4, 0.5 * size)
            mean = np.nansum(draw, axis=1) / np.maximum(1, counts)
            var = np.nansum((draw - mean[:, None]) ** 2, axis=1) / np.maximum(1, counts - 1)
            means.append(mean)
            variances.append(var / np.maximum(1, counts))
        m = np.mean(means, axis=0)
        se = np.sqrt(np.sum(variances, axis=0)) / len(values)
        resolved = (
            (m + norm.ppf(0.95) * se < margin)
            if expected < margin
            else (m - norm.ppf(0.95) * se > margin)
        )
        curve[str(size)] = float((resolved & valid).mean())
    return curve


def freeze(spec, development, root):
    dev = read(development)
    details = {}
    design = read(spec)
    tasks = dev["tasks"]
    needed = 12
    for model in MODELS:
        rows = collect(dev, root, model)
        motion = groups(rows, tasks, "motion_32_mm")
        background = groups(rows, tasks, "background_rms_max_mm")
        sd = within_sd(motion)
        n_motion = power_count(max(sd or 8, 8))
        colour_sd = within_sd(background)
        margin = design["background_rms_margin_mm"]
        mu = (
            float(np.mean([g.mean() for g in background if len(g)]))
            if any(len(g) for g in background)
            else None
        )
        # For a mean on either side of the margin, target precision to resolve that side.
        n_background = None
        if mu is not None and colour_sd is not None and abs(mu - margin) > 1e-6:
            n_background = max(
                2, math.ceil(((norm.ppf(0.95) + norm.ppf(0.8)) * colour_sd / abs(mu - margin)) ** 2)
            )
        coverage = min(len(g) / len(dev["states"][str(tid)]) for tid, g in zip(tasks, motion))
        n = max(n_motion, n_background or 10000)
        attempts = math.ceil(n / max(0.01, coverage) / len(tasks))
        ceiling = len(design["fresh_candidate_states"])
        curve = planning_curve(
            background, [len(dev["states"][str(t)]) for t in tasks], margin, 12, ceiling
        )
        planned = next((i for i in range(12, ceiling + 1) if curve.get(str(i), 0) >= 0.8), ceiling + 1)
        needed = max(needed, attempts, planned)
        details[model] = {
            "complete_by_task": [len(x) for x in motion],
            "pooled_motion_sd": sd,
            "plannable_motion_n": n_motion,
            "background_sd": colour_sd,
            "observed_background_mean": mu,
            "background_n": n_background,
            "attempts_per_task": attempts,
            "simulated_background_resolution": curve,
            "min_task_coverage": coverage,
        }
    budget = min(len(design["fresh_candidate_states"]), needed)
    rng = np.random.default_rng(20261123)
    return {
        **design,
        "split": "held_out",
        "states": {
            str(task): rng.permutation(design["fresh_candidate_states"])[:budget].tolist()
            for task in tasks
        },
        "planning": details,
        "sample_size_per_task": budget,
        "cap_limited": needed > budget,
        "development_sha256": digest(development),
        "design_sha256": digest(spec),
        "ledger": "Reviewed available E1 protocols. LIBERO-10 IDs 34-49 reserved. Prior IDs 34-39 in libero_object were from another suite.",
        "claim_rule": "Separate prespecified checkpoint-specific estimates. A two-checkpoint claim requires each to pass (intersection-union). No post-hoc task claims. Signed and selectivity comparisons are secondary.",
    }


def unit(vec):
    length = np.linalg.norm(vec)
    return vec / length if length else vec


def sample_vectors(manifest, root, model):
    for task, idx, folders in cases(manifest, root):
        files = [f / f"{model}_features.npz" for f in folders]
        if not all(
            read(f / (manifest.get("source_backend", "fastwam") + "_source.json"))["passed"]
            for f in folders
        ):
            continue
        group = {k: [] for k in PROPERTIES}
        for f in files:
            with np.load(f, allow_pickle=False) as z:
                for k in PROPERTIES:
                    group[k].append(unit(z[k]))
        yield task, idx, {k: unit(np.mean(v, axis=0)) for k, v in group.items()}


def gate(dev_path, held_path, dev_root, held_root, model):
    dev, held = read(dev_path), read(held_path)
    learn = list(sample_vectors(dev, dev_root, model))
    test = list(sample_vectors(held, held_root, model))
    rng = np.random.default_rng(641911)
    if {(a, b) for a, b, _ in learn} & {(a, b) for a, b, _ in test}:
        raise ValueError("Overlap between training and test")
    result = {}
    for prop in PROPERTIES:
        by_task = {}
        labels = []
        for task in held["tasks"]:
            training = [v[prop] for t, _, v in learn if t == task]
            template = unit(np.mean(training, axis=0)) if training else None
            scored = []
            for tid, idx, v in test:
                if tid != task or template is None:
                    continue
                side = int(rng.choice([-1, 1]))
                val = float(template @ (side * v[prop]))
                good = val * side > 0
                scored.append(
                    {"state": idx, "random_orientation": side, "score": val, "correct": bool(good)}
                )
                labels.append((side, val))
            n = len(scored)
            qualified = (
                n
                >= max(
                    held["minimum_complete_states_per_task"],
                    len(held["states"][str(task)]) * held["minimum_task_coverage"],
                )
                and sum(s["correct"] for s in scored) > n / 2
            )
            by_task[str(task)] = {
                "training_states": len(training),
                "scores": scored,
                "qualified": bool(qualified),
            }
        total = len(labels)
        count = sum(side * val > 0 for side, val in labels)
        p = binomtest(count, total, 0.5, alternative="greater").pvalue if total else 1.0
        permuted = rng.permutation([s for s, _ in labels])
        shuffled = sum(float(s) * val > 0 for s, (_, val) in zip(permuted, labels))
        result[prop] = {
            "n": total,
            "correct": count,
            "p": float(p),
            "shuffled_control_correct": shuffled,
            "by_task": by_task,
            "passed": bool(
                total
                and count / total >= 0.75
                and p < 0.05
                and all(d["qualified"] for d in by_task.values())
            ),
        }
    return {
        "backend": model,
        "properties": result,
        "passed": all(p["passed"] for p in result.values()),
        "manifest_sha256": digest(held_path),
    }


def estimates(values, rng):
    if not all(len(g) for g in values):
        return None
    by_task_draws = np.stack([g[rng.integers(0, len(g), (10000, len(g)))].mean(axis=1) for g in values])
    samples = by_task_draws.mean(axis=0)
    return {
        "mean": float(np.mean([g.mean() for g in values])),
        "ci95": np.quantile(samples, [0.025, 0.975]).tolist(),
        "ci90": np.quantile(samples, [0.05, 0.95]).tolist(),
        "by_task": [float(g.mean()) for g in values],
        "by_task_ci95": np.quantile(by_task_draws, [0.025, 0.975], axis=1).T.tolist(),
        "by_task_ci90": np.quantile(by_task_draws, [0.05, 0.95], axis=1).T.tolist(),
        "p90": float(np.quantile(np.concatenate(values), 0.9)),
    }


def summarize(manifest_path, root, model):
    spec = read(manifest_path)
    check = read(root / f"{model}_feature_gate.json")
    if check["manifest_sha256"] != digest(manifest_path):
        raise ValueError("Unmatched held-out manifest")
    rows = collect(spec, root, model)
    totals = {str(t): sum(r["eligible"] for r in rows if r["task"] == t) for t in spec["tasks"]}
    enough = all(
        totals[str(t)]
        >= max(
            spec["minimum_complete_states_per_task"],
            spec["minimum_task_coverage"] * len(spec["states"][str(t)]),
        )
        for t in spec["tasks"]
    )
    rng = np.random.default_rng(20261123)
    stats = {k: estimates(groups(rows, spec["tasks"], k), rng) for k in METRICS}
    motion, background, sel = (
        stats["motion_32_mm"],
        stats["background_rms_max_mm"],
        stats["selectivity_mm"],
    )
    margin = spec["background_rms_margin_mm"]
    arm_ok = check["properties"]["arm"]["passed"]
    tint_ok = check["properties"]["colour"]["passed"]
    generalizes = bool(
        enough and arm_ok and motion and motion["ci95"][0] > 0 and min(motion["by_task"]) > 0
    )
    nuisance = "inconclusive"
    if enough and tint_ok and background:
        if background["ci90"][1] < margin:
            nuisance = "bounded mean response"
        if background["ci90"][0] > margin:
            nuisance = "mean response exceeds margin"
    all_complete = []
    for r in rows:
        if all("metrics" in z for z in r["seeds"]):
            means = {k: float(np.mean([z["metrics"][k] for z in r["seeds"]])) for k in METRICS}
            all_complete.append(dict(r, eligible=True, means=means))
    sensitivity = {
        k: estimates(groups(all_complete, spec["tasks"], k), rng)
        for k in ("motion_32_mm", "background_rms_max_mm", "selectivity_mm")
    }
    all_runs = [trial for row in rows for trial in row["seeds"]]
    shams = [z["metrics"]["sham_drift_mm"] for z in all_runs if "metrics" in z]
    contact_comparisons = []
    for z in all_runs:
        c = z.get("contacts", {})
        if "left_neutral" not in c:
            continue

        def occurred(k, contacts=c):
            return (
                bool(contacts[k].get("contact_substeps", 0))
                if isinstance(contacts[k], dict)
                else bool(contacts[k])
            )

        contact_comparisons.append(
            any(
                occurred(f"{d}_{t}") != occurred(f"{d}_neutral")
                for d in ("left", "right")
                for t in ("warm", "cool")
            )
        )
    quality = {
        "sampled_seeds": len(all_runs),
        "source_pass": sum(s["source_pass"] for s in all_runs),
        "executed_seeds": len(shams),
        "sham_eligible_seeds": sum(s.get("eligible", False) for s in all_runs),
        "largest_sham_drift_mm": max(shams, default=None),
        "changed_contact_status_runs": sum(contact_comparisons),
        "contact_comparable_runs": len(contact_comparisons),
    }
    return {
        "quality": quality,
        "sensitivity_all_complete_starts": sensitivity,
        "model": model,
        "suite": spec["suite"],
        "tasks": spec["tasks"],
        "attempted": len(rows),
        "usable": totals,
        "coverage_pass": enough,
        "estimates": stats,
        "motion_generalizes": generalizes,
        "background_decision": nuisance,
        "signed_endpoint_equivalence": bool(
            enough
            and tint_ok
            and stats["signed_background_half_mm"]
            and max(map(abs, stats["signed_background_half_mm"]["ci90"])) < margin
        ),
        "same_axis_selectivity": bool(enough and arm_ok and tint_ok and sel and sel["ci95"][0] > 0),
        "feature_gate": check,
        "states": rows,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("freeze", "gate", "report"))
    parser.add_argument(
        "--root", type=Path, required=True, help="Directory containing dev.json, dev/, held.json, held/"
    )
    parser.add_argument("--design", type=Path, default=Path("configs/experiments/e1_confirmation.json"))
    parser.add_argument("--backend", choices=MODELS)
    a = parser.parse_args()
    root = a.root
    dev, held = root / "dev.json", root / "held.json"
    if a.stage == "freeze":
        write(held, freeze(a.design, dev, root / "dev"), True)
        calibration = {}
        devspec = read(dev)
        for model in MODELS:
            data = collect(devspec, root / "dev", model)
            own = []
            for task, idx, folders in cases(devspec, root / "dev"):
                for folder in folders:
                    marker = folder / f"{model}_own_source_behavior.json"
                    if marker.is_file():
                        own.append(
                            {"task": task, "state": idx, "seed": folder.name, "behavior": read(marker)}
                        )
                    elif (folder / f"{model}_source.json").exists() and model == "longwam":
                        own.append(
                            {
                                "task": task,
                                "state": idx,
                                "seed": folder.name,
                                "source": read(folder / f"{model}_source.json"),
                            }
                        )
            calibration[model] = {"shared_source": data, "own_source": own}
        write(root / "calibration_summary.json", calibration)
        codefiles = sorted(Path("scripts").glob("e1_confirm*.py")) + sorted(
            Path("src/experiments").glob("confirm*.py")
        )
        codefiles += list(
            map(
                Path,
                [
                    "src/experiments/future_adapter.py",
                    "src/experiments/clip_study.py",
                    "scripts/run_e1_confirmation.sh",
                ],
            )
        )
        write(
            root / "implementation_hashes.json",
            {str(f): digest(f) for f in codefiles if f.exists()},
            True,
        )
    elif a.stage == "gate":
        if not a.backend:
            parser.error("--backend required")
        write(
            root / "held" / f"{a.backend}_feature_gate.json",
            gate(dev, held, root / "dev", root / "held", a.backend),
            True,
        )
    else:
        for model in [a.backend] if a.backend else MODELS:
            write(root / f"{model}_results.json", summarize(held, root / "held", model))
        if not a.backend:
            reports = {m: read(root / f"{m}_results.json") for m in MODELS}
            paired = set.intersection(
                *[
                    {(r["task"], r["state"]) for r in report["states"] if r["eligible"]}
                    for report in reports.values()
                ]
            )
            common = {"ids": sorted(paired), "models": {}}
            for m, report in reports.items():
                rows = [r for r in report["states"] if (r["task"], r["state"]) in paired]
                common["models"][m] = {
                    k: estimates(groups(rows, report["tasks"], k), np.random.default_rng(20261123))
                    for k in ("motion_32_mm", "background_rms_max_mm", "selectivity_mm")
                }
            write(root / "common_starts.json", common)


if __name__ == "__main__":
    main()
