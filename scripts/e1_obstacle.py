"""Generate development sources or execute paired E1 obstacle rollouts on LIBERO."""

import argparse
import json
from pathlib import Path

import numpy as np
from wam_interp.evaluations.LIBERO.environment import initial_states_for_task
from wam_interp.evaluations.LIBERO.runner import _get_suite
from wam_interp.experiments.clip_study import save
from wam_interp.experiments.confirm_replay import anchor_hash, omit_unused_rgb
from wam_interp.experiments.obstacle import Barrier, score
from wam_interp.experiments.obstacle_libero import (
    PLACEMENTS,
    READOUT,
    recipient_clip,
    render_sources,
    rollout,
    scene,
)


def behavior(api, task, task_id, state, idx, seed, folder, split="development"):
    from e1_confirm import commands

    meta = json.loads((folder / "sources.json").read_text())
    result = {
        "backend": api.backend,
        "task": task_id,
        "state": idx,
        "seed": seed,
        "design": split,
        "passed": False,
        "readout": READOUT,
        "arms": {},
    }
    if not meta["passed"]:
        result["reason"] = "source geometry or visibility failed"
        save(folder / "behavior.json", result)
        return
    with np.load(folder / "sources.npz", allow_pickle=False) as z:
        clips = {k: z[k] for k in ("clear", *PLACEMENTS)}
        native_path = z["native_path"]
        ref = z["commands"]
    paths, tensors = {}, {}
    # Each live scene supplies its own anchor and native identity replay.
    for live in ("clear", "minus", "plus"):
        box = (
            None if live == "clear" else Barrier(**meta["placements"][live]["geometry"])
        )
        expected = (
            meta["clear_anchor"] if not box else meta["placements"][live]["anchor"]
        )
        labels = ("clear", *PLACEMENTS) if not box else ("clear", live)
        opts = {"noise_seed": seed + 20261126} if api.backend == "longwam" else {}
        with scene(api, task, state, idx, task_id, seed, box) as (_, _, obs, req):
            if anchor_hash(obs) != expected:
                raise ValueError("Source/recipient anchor mismatch")
            prediction, native = api.capture(req)
            original = commands(api, prediction)
            identity = commands(api, api.predict_with_future(req, native))
            error = float(np.abs(original - identity).max())
            if error > 1e-5 or (
                live == "clear" and not np.allclose(original, ref, atol=1e-5, rtol=0)
            ):
                raise ValueError(
                    f"Native replay or reference failed for {live} (error={error})"
                )
            plans = {"native": original}
            for future in labels:
                clip = recipient_clip(clips[future], clips[live])
                donor = api.from_clip(req, clip, **opts)
                if live == "clear" and future in ("plus", "minus"):
                    tensors[future] = donor.parts[0].float().numpy()
                plans[future] = commands(api, api.predict_with_future(req, donor))
        for future, cmd in plans.items():
            key = live + "__" + future
            with scene(None, task, state, idx, task_id, seed, box) as (
                env,
                args,
                obs,
                _,
            ):
                if anchor_hash(obs) != expected:
                    raise ValueError("Reset differed during execution")
                with omit_unused_rgb(env):
                    path, _, _, contact = rollout(
                        env, args, obs, cmd, barrier=bool(box)
                    )
            paths[key] = path
            paths[key + "_commands"] = cmd
            result["arms"][key] = {
                "contact_steps": contact,
                "controls": len(path) - 1,
                "identity_max_command_error": error,
            }
    np.save(folder / "side_difference.npy", tensors["plus"] - tensors["minus"])
    np.savez_compressed(folder / "paths.npz", **paths)
    drift = np.linalg.norm(paths["clear__clear"][-1] - native_path[-1]) * 1000
    result["clear_sham_drift_mm"] = float(drift)
    result["metrics"] = {}
    for side in ("plus", "minus"):
        for label, a, b in (
            ("future_only", "clear__" + side, "clear__clear"),
            ("both_minus_live", side + "__" + side, side + "__clear"),
            ("live_only", side + "__clear", "clear__clear"),
            ("off_route", "clear__off_" + side, "clear__clear"),
        ):
            box_key = "off_" + side if label == "off_route" else side
            if min(len(paths[a]), len(paths[b])) > READOUT:
                result["metrics"][label + "_" + side] = score(
                    paths[a],
                    paths[b],
                    Barrier(**meta["placements"][box_key]["geometry"]),
                    READOUT,
                )
    result["passed"] = bool(drift <= 20 and len(result["metrics"]) == 8)
    result["contact_before_readout"] = [
        k
        for k, v in result["arms"].items()
        if any(c <= READOUT for c in v["contact_steps"])
    ]
    save(folder / "behavior.json", result)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage", choices=("sources", "behavior"))
    p.add_argument(
        "--backend",
        choices=("fastwam", "longwam", "lingbot_va", "openwam"),
        required=True,
    )
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--refs", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--station", type=int, choices=(28, 32), default=28)
    a = p.parse_args()
    if a.stage == "behavior" and a.backend not in ("fastwam", "longwam"):
        p.error("Use e1_obstacle_cache.py for a KV intervention")
    m = json.loads(a.manifest.read_text())
    if m["split"] not in ("development", "confirmation") or m.get("reasons"):
        p.error("Expected a development manifest or an eligible frozen design")
    if m["split"] == "confirmation" and (
        m.get("backend") != a.backend or m.get("source_station") != a.station
    ):
        p.error("Backend or geometry differs from the frozen design")
    suite = _get_suite("libero_10")
    api = None
    for task_id in m["tasks"]:
        task = suite.get_task(task_id)
        states = initial_states_for_task(suite, task_id)
        for idx in m["states"][str(task_id)]:
            for seed in m["seeds"]:
                sub = Path(f"task_{task_id}/state_{idx}/seed_{seed}")
                dest = a.out / a.backend / sub
                marker = dest / (
                    "sources.json" if a.stage == "sources" else "behavior.json"
                )
                if marker.exists():
                    continue
                ref = a.refs / sub / f"{a.backend}_reference.npy"
                if a.stage == "sources":
                    render_sources(
                        task,
                        task_id,
                        states[idx],
                        idx,
                        seed,
                        np.load(ref),
                        dest,
                        station=a.station,
                    )
                else:
                    if api is None:
                        from e1_confirm import load

                        api = load(a.backend)
                    behavior(
                        api, task, task_id, states[idx], idx, seed, dest, m["split"]
                    )
                print(marker, flush=True)


if __name__ == "__main__":
    main()
