"""Compare separated held-out inference/replay to native development branch traces."""

import argparse
import json
import shutil
import tempfile
from pathlib import Path

import numpy as np
from e1_confirm import load
from wam_interp.evaluations.LIBERO.environment import initial_states_for_task
from wam_interp.evaluations.LIBERO.runner import _get_suite
from wam_interp.experiments.clip_study import save
from wam_interp.experiments.confirm_replay import replay
from wam_interp.experiments.confirmation import run_case


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--backend", choices=("fastwam", "longwam"), required=True)
    a = p.parse_args()
    spec = json.loads((a.root / "dev.json").read_text())
    api = load(a.backend)
    suite = _get_suite("libero_10")
    report = []
    for t in spec["tasks"]:
        task = suite.get_task(t)
        idx = spec["states"][str(t)][0]
        state = initial_states_for_task(suite, t)[idx]
        seed = spec["seeds"][0]
        src = a.root / "dev" / f"task_{t}/state_{idx}/seed_{seed}"
        if not (src / f"{a.backend}_paths.npz").exists():
            continue
        with tempfile.TemporaryDirectory(dir=a.root) as tmp:
            folder = Path(tmp)
            for name in ("fastwam_source.json", "fastwam_source.npz"):
                shutil.copyfile(src / name, folder / name)
            run_case(api, "commands", a.backend, task, t, state, idx, seed, folder, spec)
            meta = json.loads((folder / f"{a.backend}_commands.json").read_text())
            with np.load(folder / f"{a.backend}_commands.npz") as z:
                chunks = {k: z[k] for k in meta["order"]}
            with np.load(src / f"{a.backend}_commands.npz") as golden:
                cmd_max = max(float(np.abs(golden[k] - v).max()) for k, v in chunks.items())
            paths, _ = replay(task, state, chunks, meta["anchor_hash"])
            with np.load(src / f"{a.backend}_paths.npz") as golden:
                path_max = max(float(np.abs(golden[k] - v).max()) for k, v in paths.items())
            report.append(
                {
                    "task": t,
                    "state": idx,
                    "seed": seed,
                    "branches": len(chunks),
                    "command_max_abs": cmd_max,
                    "path_max_abs_m": path_max,
                }
            )
            save(a.root / f"{a.backend}_parity.json", report)
            if cmd_max or path_max > 1e-7:
                raise ValueError("Optimized path differs from the development run")
    if not report:
        raise ValueError("No development traces for comparison")


if __name__ == "__main__":
    main()
