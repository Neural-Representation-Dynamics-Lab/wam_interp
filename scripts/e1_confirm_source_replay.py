"""Replay saved donor commands and check every robot-to-scene contact against the original paths."""

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from wam_interp.evaluations.LIBERO.environment import initial_states_for_task
from wam_interp.evaluations.LIBERO.runner import _get_suite
from wam_interp.experiments.clip_study import save, start
from wam_interp.experiments.confirm_replay import anchor_hash, replay


def audit(job):
    task_id, idx, seed, folder, backend = job
    receipt = folder / f"{backend}_source_audit.json"
    src = json.loads((folder / f"{backend}_source.json").read_text())
    if receipt.exists() or not src["passed"]:
        return
    suite = _get_suite("libero_10")
    task = suite.get_task(task_id)
    state = initial_states_for_task(suite, task_id)[idx]
    with start(None, task, state, idx, task_id=task_id, seed=seed) as (_, __, obs, ___):
        anchor = anchor_hash(obs)
    ref = np.load(folder / f"{backend}_reference.npy")
    bias = src["attempts"][-1]["bias"]
    chunks = {"sham": ref}
    for side, sign in (("left", -1), ("right", 1)):
        c = ref.copy()
        c[:, 0] = np.clip(c[:, 0] + sign * bias, -1.0, 1.0)
        chunks[side] = c
    paths, contacts = replay(task, state, chunks, anchor)
    with np.load(folder / f"{backend}_source.npz") as z:
        deltas = {
            k: float(
                np.max(
                    np.abs(paths[k] - z["source_reference_path" if k == "sham" else f"{k}_source_path"])
                )
            )
            for k in chunks
        }
    passed = not any(contacts.values()) and max(deltas.values()) < 1e-7
    save(
        receipt,
        {
            "passed": passed,
            "source_hash": src["sha256"],
            "contact_substeps": contacts,
            "maximum_path_error_m": deltas,
        },
    )
    if not passed:
        raise ValueError(f"Donor source replay failed in {folder}")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--workers", type=int, default=2)
    a = p.parse_args()
    spec = json.loads(a.manifest.read_text())
    jobs = [
        (t, i, s, a.out / f"task_{t}/state_{i}/seed_{s}", "fastwam")
        for t in spec["tasks"]
        for i in spec["states"][str(t)]
        for s in spec["seeds"]
    ]
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        list(pool.map(audit, jobs))


if __name__ == "__main__":
    main()
