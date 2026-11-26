"""Generate geometrically checked counterfactual clips independently of recipient actions."""

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from wam_interp.evaluations.LIBERO.environment import initial_states_for_task
from wam_interp.evaluations.LIBERO.runner import _get_suite
from wam_interp.experiments.confirm_sources import make_sources


def render(item):
    task_id, idx, seed, folder, backend, stabilize = item
    suite = _get_suite("libero_10")
    state = initial_states_for_task(suite, task_id)[idx]
    make_sources(suite.get_task(task_id), task_id, state, idx, seed, folder, backend, stabilize)
    return (task_id, idx, seed)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--backend", choices=("fastwam", "longwam"), default="fastwam")
    p.add_argument("--workers", type=int, default=2)
    args = p.parse_args()
    spec = json.loads(args.manifest.read_text())
    cases = [
        (
            task,
            idx,
            seed,
            args.out / f"task_{task}/state_{idx}/seed_{seed}",
            args.backend,
            spec["split"] == "development" and idx == spec["states"][str(task)][0],
        )
        for task in spec["tasks"]
        for idx in spec["states"][str(task)]
        for seed in spec["seeds"]
    ]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for case in pool.map(render, cases):
            print("source complete", case, flush=True)


if __name__ == "__main__":
    main()
