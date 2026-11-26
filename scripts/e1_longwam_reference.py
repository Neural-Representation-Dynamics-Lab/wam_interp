"""Register a Long-WAM reference-source comparison and reuse native controls from matched starts."""

import argparse
import hashlib
from pathlib import Path

import numpy as np
from e1_confirm_analysis import cases, digest, estimates, groups, read, write


def register(shared, dest):
    plan = read(Path("configs/experiments/e1_longwam_reference.json"))
    base = shared / "held.json"
    if digest(base) != plan["parent_heldout_manifest_sha256"]:
        raise ValueError("The comparison requires the registered shared-video study")
    for split in ("dev", "held"):
        spec = read(shared / f"{split}.json")
        spec.update(source_backend="longwam", suite="libero_10", source_study=plan)
        write(dest / f"{split}.json", spec, True)


def copy_native(shared, dest, split):
    spec = read(dest / f"{split}.json")
    rows = []
    for task, idx, folders in cases(spec, dest / split):
        for seed, folder in zip(spec["seeds"], folders):
            folder.mkdir(parents=True, exist_ok=True)
            origin = shared / split / folder.relative_to(dest / split) / "longwam_commands.npz"
            target = folder / "longwam_reference.npy"
            row = {"task": task, "state": idx, "seed": seed, "copied": False}
            if origin.is_file() and any(
                (origin.parent / name).is_file()
                for name in ("longwam_behavior.json", "longwam_commands.json")
            ):
                with np.load(origin, allow_pickle=False) as z:
                    commands = z["native"]
                if commands.shape != (32, 7) or not np.isfinite(commands).all():
                    raise ValueError("Invalid native reference")
                if target.exists():
                    if not np.array_equal(np.load(target), commands):
                        raise ValueError("Previously saved native commands differ")
                else:
                    temp = target.with_suffix(".tmp.npy")
                    np.save(temp, commands)
                    temp.replace(target)
                row.update(
                    copied=True, native_array_sha256=hashlib.sha256(commands.tobytes()).hexdigest()
                )
            rows.append(row)
    write(dest / f"{split}_reference_ledger.json", rows)


def reference_paths(shared, dest, folder):
    """Verify that each rendered source followed the same policy's native physical trajectory."""
    sources = {}
    audit = {}
    for backend, root in (("fastwam", shared), ("longwam", dest)):
        inputs = root / folder
        source = inputs / f"{backend}_source.npz"
        traces = inputs / f"{backend}_paths.npz"
        if not source.is_file():
            continue
        with np.load(source, allow_pickle=False) as archive:
            sources[backend] = archive["source_reference_path"]
        if traces.is_file():
            with np.load(traces, allow_pickle=False) as archive:
                native = archive["native"]
            ref = sources[backend]
            if ref.shape != native.shape:
                raise ValueError(f"Native trajectory length changed at {inputs}")
            error = float(np.max(np.abs(ref - native)))
            audit[backend + "_reference_max_abs_m"] = error
            if error > 1e-7:
                raise ValueError(f"Rendered reference differs from executed native trajectory at {inputs}")
    if len(sources) == 2:
        physical_distance = np.linalg.norm(sources["longwam"] - sources["fastwam"], axis=-1) * 1000
        audit.update(endpoint_separation_mm=float(physical_distance[-1]),
                     trajectory_rms_separation_mm=float(np.sqrt(np.mean(physical_distance[1:] ** 2))))
    return audit


def compare(shared, dest):
    first = read(shared / "longwam_results.json")
    second = read(dest / "longwam_results.json")
    tasks = first["tasks"]
    pairs = []
    states = lambda run: {(r["task"], r["state"]): r for r in run["states"]}
    a, b = states(first), states(second)
    if a.keys() != b.keys():
        raise ValueError("The two studies must have the same attempts")
    for task, idx in a:
        x, y = a[task, idx], b[task, idx]
        rec = {
            "task": task,
            "state": idx,
            "shared_pass": x["eligible"],
            "own_pass": y["eligible"],
            "eligible": x["eligible"] and y["eligible"],
        }
        rec["native_checks"] = []
        rec["reference_trajectory_checks"] = []
        for left, right in zip(x["seeds"], y["seeds"], strict=True):
            if left["seed"] != right["seed"]:
                raise ValueError("Mismatched action seeds")
            f = Path("held") / f"task_{task}/state_{idx}/seed_{left['seed']}"
            rec["reference_trajectory_checks"].append(
                dict(seed=left["seed"], **reference_paths(shared, dest, f))
            )
            if "metrics" not in left or "metrics" not in right:
                continue
            errs = {}
            for name in ("commands", "paths"):
                rel = f / f"longwam_{name}.npz"
                with np.load(shared / rel) as old, np.load(dest / rel) as new:
                    errs[name] = float(np.abs(old["native"] - new["native"]).max())
            rec["native_checks"].append(dict(seed=left["seed"], **errs))
            if errs["commands"] or errs["paths"] > 1e-7:
                raise ValueError(f"Native replay differed at {f}")
        if rec["eligible"]:
            rec["means"] = {k: y["means"][k] - x["means"][k] for k in x["means"]}
        pairs.append(rec)
    results = {}
    for k in ("motion_32_mm", "background_rms_max_mm", "sham_drift_mm", "selectivity_mm"):
        vals = groups(pairs, tasks, k)
        results[k] = {
            "pooled": estimates(vals, np.random.default_rng(20261123)),
            "by_task": {
                str(t): estimates([v], np.random.default_rng(20261123)) for t, v in zip(tasks, vals)
            },
        }
    counts = {
        str(t): {
            key: sum(
                bool(r["shared_pass"]) == a0 and bool(r["own_pass"]) == b0
                for r in pairs
                if r["task"] == t
            )
            for key, a0, b0 in (
                ("both", True, True),
                ("shared_only", True, False),
                ("own_only", False, True),
                ("neither", False, False),
            )
        }
        for t in tasks
    }
    write(
        dest / "paired_source_comparison.json",
        {
            "suite": first["suite"],
            "tasks": tasks,
            "coverage": counts,
            "own_minus_shared": results,
            "states": pairs,
        },
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage", choices=("register", "references", "compare"))
    p.add_argument("--shared", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--split", choices=("dev", "held"))
    a = p.parse_args()
    if a.stage == "register":
        register(a.shared, a.out)
    elif a.stage == "references":
        if not a.split:
            p.error("--split required")
        copy_native(a.shared, a.out, a.split)
    else:
        compare(a.shared, a.out)


if __name__ == "__main__":
    main()
