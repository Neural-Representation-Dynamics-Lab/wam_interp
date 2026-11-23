"""Execute already frozen open-loop commands while the next GPU inference runs."""

import argparse
import json
from pathlib import Path
from time import monotonic, sleep

import numpy as np
from wam_interp.evaluations.LIBERO.environment import initial_states_for_task
from wam_interp.evaluations.LIBERO.runner import _get_suite
from wam_interp.experiments.clip_study import save
from wam_interp.experiments.confirm_replay import replay
from wam_interp.experiments.confirmation import measured_effects


def score_commands(task, state, folder, backend, margin):
    receipt = folder / f"{backend}_behavior.json"
    inputs = folder / f"{backend}_commands.json"
    if receipt.exists():
        return
    meta = json.loads(inputs.read_text())
    if meta.get("eligible") is False:
        save(receipt, meta)
        return
    began = monotonic()
    with np.load(folder / f"{backend}_commands.npz", allow_pickle=False) as z:
        chunks = {k: z[k] for k in meta["order"]}
    paths, contacts = replay(task, state, chunks, meta["anchor_hash"])
    np.savez_compressed(folder / f"{backend}_paths.npz", **paths)
    result = dict(meta, contacts=contacts, seconds=meta["prediction_seconds"] + monotonic() - began)
    if all(len(path) == 33 for path in paths.values()):
        metrics = measured_effects(paths)
        result.update(metrics=metrics, eligible=metrics["sham_drift_mm"] <= margin)
    else:
        result.update(
            eligible=False,
            reason="early_termination",
            lengths={k: len(v) - 1 for k, v in paths.items()},
        )
    save(receipt, result)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--backend", choices=("fastwam", "longwam"), required=True)
    p.add_argument("--wait", action="store_true")
    p.add_argument("--task", type=int, help="Execute just one registered task")
    a = p.parse_args()
    spec = json.loads(a.manifest.read_text())
    suite = _get_suite("libero_10")
    for task in spec["tasks"]:
        if a.task is not None and a.task != task:
            continue
        states = initial_states_for_task(suite, task)
        for idx in spec["states"][str(task)]:
            for seed in spec["seeds"]:
                folder = a.out / f"task_{task}/state_{idx}/seed_{seed}"
                receipt = folder / f"{a.backend}_behavior.json"
                inputs = folder / f"{a.backend}_commands.json"
                if receipt.is_file():
                    continue
                if not json.loads((folder / "fastwam_source.json").read_text())["passed"]:
                    continue
                until = monotonic() + 10800
                while a.wait and not inputs.is_file() and monotonic() < until:
                    sleep(2)
                score_commands(
                    suite.get_task(task), states[idx], folder, a.backend, spec["sham_margin_mm"]
                )
                print("executed", a.backend, task, idx, seed, flush=True)


if __name__ == "__main__":
    main()
