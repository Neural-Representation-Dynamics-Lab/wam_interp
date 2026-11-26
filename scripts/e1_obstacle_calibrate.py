"""Independently test unedited replay and visual side encoding before further policy comparisons."""

import argparse
import json
from pathlib import Path

import numpy as np
from e1_confirm import commands, load
from wam_interp.evaluations.LIBERO.environment import initial_states_for_task
from wam_interp.evaluations.LIBERO.runner import _get_suite
from wam_interp.experiments.clip_study import save, start
from wam_interp.experiments.confirm_replay import anchor_hash, omit_unused_rgb
from wam_interp.experiments.obstacle_libero import recipient_clip, rollout


def calibrate(api, task, state, folder, meta):
    task_id, idx, seed = (meta[k] for k in ("task", "state", "seed"))
    out = folder / "calibration.json"
    if not meta["passed"]:
        save(out, {"source_pass": False, "eligible": False})
        return
    with np.load(folder / "sources.npz") as z:
        ref = z["native_path"]
        clips = {k: z[k] for k in ("clear", "minus", "plus")}
    with start(api, task, state, idx, task_id=task_id, seed=seed) as (
        env,
        kw,
        obs,
        req,
    ):
        if anchor_hash(obs) != meta["clear_anchor"]:
            raise ValueError("Recipient differs from registered source")
        opts = {"noise_seed": 20261126 + seed} if api.backend == "longwam" else {}
        donors = {
            k: api.from_clip(req, recipient_clip(v, clips["clear"]), **opts)
            for k, v in clips.items()
        }
        diff = (
            donors["plus"].parts[0].float() - donors["minus"].parts[0].float()
        ).numpy()
        np.save(folder / "side_difference.npy", diff)
        cmd = commands(api, api.predict_with_future(req, donors["clear"]))
        with omit_unused_rgb(env):
            path, _, _, _ = rollout(env, kw, obs, cmd)
    drift = float(1000 * np.linalg.norm(path[-1] - ref[-1]))
    save(
        out,
        {
            "eligible": len(path) == 33 and drift <= 20,
            "source_pass": True,
            "clear_sham_drift_mm": drift,
            "controls": len(path) - 1,
            "side_rms": float(np.sqrt(np.mean(diff**2))),
        },
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--backend", required=True, choices=("fastwam", "longwam"))
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    manifest = json.loads(args.manifest.read_text())
    api = load(args.backend)
    suite = _get_suite("libero_10")
    for t in manifest["tasks"]:
        task, states = suite.get_task(t), initial_states_for_task(suite, t)
        for i in manifest["states"][str(t)]:
            for s in manifest["seeds"]:
                folder = args.out / args.backend / f"task_{t}/state_{i}/seed_{s}"
                if not (folder / "calibration.json").is_file():
                    calibrate(
                        api,
                        task,
                        states[i],
                        folder,
                        json.loads((folder / "sources.json").read_text()),
                    )
                    print(folder, flush=True)


if __name__ == "__main__":
    main()
