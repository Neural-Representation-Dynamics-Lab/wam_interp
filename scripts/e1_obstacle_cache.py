"""Test a shared obstacle fixture at the LingBot/OpenWAM native future-only cache boundary."""

import argparse
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
from wam_interp.evaluations.LIBERO.environment import initial_states_for_task
from wam_interp.evaluations.LIBERO.runner import _get_suite
from wam_interp.experiments.clip_study import save
from wam_interp.experiments.confirm_replay import anchor_hash, omit_unused_rgb
from wam_interp.experiments.future_adapter import FutureAdapter
from wam_interp.experiments.obstacle import Barrier, score
from wam_interp.experiments.obstacle_features import lingbot_clip, openwam_clip
from wam_interp.experiments.obstacle_libero import PLACEMENTS, rollout, scene
from wam_interp.experiments.simulator_video import paired_camera_clip
from wam_interp.types import InferenceRequest


def load(backend):
    if backend == "lingbot_va":
        from wam_interp.models.lingbot_va.libero_adapter import LingBotVALiberoAdapter
        from wam_interp.models.lingbot_va.loader import load_lingbot_va

        runtime = load_lingbot_va(checkpoint_dir="checkpoints/lingbot_va/libero_long")
        return FutureAdapter(LingBotVALiberoAdapter(runtime))
    from wam_interp.models.openwam.libero_adapter import OpenWAMLiberoAdapter
    from wam_interp.models.openwam.loader import load_openwam

    adapter = OpenWAMLiberoAdapter(load_openwam("checkpoints/openwam/libero"))
    return FutureAdapter(
        adapter, stage=adapter.inference_settings["num_inference_steps"] - 1
    )


def make_request(api, obs, seed):
    adapter = api.adapter
    if hasattr(adapter, "begin_episode"):
        adapter.begin_episode(obs)
    defaults = dict(adapter.inference_defaults(), seed=seed)
    horizon = (
        adapter.action_horizon
        if api.backend == "lingbot_va"
        else adapter.inference_settings["action_horizon"]
    )
    return InferenceRequest(
        adapter.prepare_observation(obs), action_horizon=horizon, **defaults
    )


def commands(api, pred):
    import torch

    x = api.adapter.postprocess_actions(pred.actions, environment="libero")
    cmd = torch.as_tensor(x).detach().float().cpu().numpy().reshape(-1, 7)
    if not np.isfinite(cmd).all():
        raise ValueError("Nonfinite actions")
    return cmd


def run(api, source, dest, *, calibrate_only=False):
    import torch

    meta = json.loads((source / "sources.json").read_text())
    if not meta["passed"]:
        raise ValueError("Select a geometry-validated shared source")
    suite = _get_suite("libero_10")
    task_id, idx, seed = meta["task"], meta["state"], meta["seed"]
    task, state = suite.get_task(task_id), initial_states_for_task(suite, task_id)[idx]
    result = {
        "backend": api.backend,
        "layer": api.layer,
        "stage": api.stage,
        "source": str(source),
        "task": task_id,
        "state": idx,
        "seed": seed,
        "instruction": task.language,
        "arms": {},
    }
    dest.mkdir(parents=True, exist_ok=True)
    with np.load(source / "sources.npz") as z:
        actions = z["commands"]
        prefix = meta["depicted_native_prefix_steps"]
        full_path = z["clear_path"]
        frame_bank = {}
        sources = (
            ("clear", "minus", "plus") if calibrate_only else ("clear", *PLACEMENTS)
        )
        for key in sources:
            box = (
                None
                if key == "clear"
                else Barrier(**meta["placements"][key]["geometry"])
            )
            with scene(None, task, state, idx, task_id, seed, box) as (env, kw, obs, _):
                path, _, cams, contact = rollout(
                    env, kw, obs, actions, frames=True, barrier=bool(box), stride=1
                )
                from wam_interp.experiments.obstacle_libero import hold_after

                cams = {k: hold_after(v, prefix) for k, v in cams.items()}
                path = hold_after(path, prefix)
                samples = paired_camera_clip(cams["agentview"][::4], cams["wrist"][::4])
                if (
                    any(c <= prefix for c in contact)
                    or not np.array_equal(samples, z[key])
                    or not np.array_equal(path, full_path)
                ):
                    raise ValueError("Dense reference failed geometry or camera replay")
                frame_bank[key] = [
                    api.adapter.prepare_observation(
                        replace(obs, cameras={c: v[t] for c, v in cams.items()})
                    ).image
                    for t in range(len(path))
                ]
    context = lingbot_clip if api.backend == "lingbot_va" else openwam_clip
    paths = {}
    for live in ("clear",) if calibrate_only else ("clear", "minus", "plus"):
        box = (
            None if live == "clear" else Barrier(**meta["placements"][live]["geometry"])
        )
        with scene(None, task, state, idx, task_id, seed, box) as (_, _, obs, _):
            req = make_request(api, obs, seed)
            base, own = api.capture(req)
            native = commands(api, base)
            repeat = commands(api, api.predict_with_future(req, own))
            if not np.array_equal(native, repeat):
                raise ValueError("Native cache identity replay failed")
            plans = {"native": native}
            side_features = {}
            for future in sources if box is None else ("clear", live):
                video = frame_bank[future].copy()
                video[0] = frame_bank[live][0]
                if api.backend == "lingbot_va":
                    video = video[:13]
                else:
                    if (len(video) - 1) % (req.num_video_frames - 1):
                        raise ValueError("Nonintegral native video stride")
                    video = video[:: (len(video) - 1) // (req.num_video_frames - 1)]
                with context(api, req, video):
                    _, donor = api.capture(req)
                if not calibrate_only or future == "clear":
                    plans[future] = commands(api, api.predict_with_future(req, donor))
                if live == "clear" and future in ("minus", "plus"):
                    side_features[future] = donor.parts
            if side_features:
                diff = [
                    a.float() - b.float()
                    for a, b in zip(side_features["plus"], side_features["minus"])
                ]
                # Save a fixed projection for later across-start decoding; do not fit on actions.
                # Pool adjacent elements to keep anonymous exported feature evidence manageable.
                np.save(
                    dest / "side_difference.npy",
                    torch.cat([x.flatten() for x in diff])
                    .numpy()
                    .reshape(-1, 32)
                    .mean(1),
                )
                result["kv_side_rms"] = [float(x.square().mean().sqrt()) for x in diff]
        for future, cmd in plans.items():
            key = live + "__" + future
            with (
                scene(None, task, state, idx, task_id, seed, box) as (env, kw, obs, _),
                omit_unused_rgb(env),
            ):
                expected = (
                    meta["clear_anchor"]
                    if box is None
                    else meta["placements"][live]["anchor"]
                )
                if anchor_hash(obs) != expected:
                    raise ValueError("Recipient anchor replay failed")
                path, _, _, contacts = rollout(env, kw, obs, cmd, barrier=bool(box))
                paths[key] = path
                paths[key + "_commands"] = cmd
                result["arms"][key] = {"controls": len(path) - 1, "contacts": contacts}
    result["sham_drift_mm"] = float(
        1000 * np.linalg.norm(paths["clear__native"][-1] - paths["clear__clear"][-1])
    )
    result["passed_sham"] = result["sham_drift_mm"] <= 20
    if calibrate_only:
        save(dest / "calibration.json", result)
        return
    result["metrics_at_8"] = {}
    for side in ("minus", "plus"):
        for label, a, b in (
            ("future_only", "clear__" + side, "clear__clear"),
            ("off_route", "clear__off_" + side, "clear__clear"),
            ("both_minus_live", side + "__" + side, side + "__clear"),
            ("live_only", side + "__clear", "clear__clear"),
        ):
            geom = ("off_" if label == "off_route" else "") + side
            result["metrics_at_8"][label + "_" + side] = score(
                paths[a], paths[b], Barrier(**meta["placements"][geom]["geometry"]), 8
            )
    result["passed_sham"] = result["sham_drift_mm"] <= 20
    np.savez_compressed(dest / "paths.npz", **paths)
    save(dest / "behavior.json", result)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--backend", required=True, choices=("lingbot_va", "openwam"))
    p.add_argument("--source", type=Path, action="append", required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--calibrate-only", action="store_true")
    args = p.parse_args()
    api = load(args.backend)
    for source in args.source:
        j = json.loads((source / "sources.json").read_text())
        dest = (
            args.out
            / args.backend
            / f"task_{j['task']}/state_{j['state']}/seed_{j['seed']}"
        )
        marker = "calibration.json" if args.calibrate_only else "behavior.json"
        if not (dest / marker).exists() and not (
            args.calibrate_only and (dest / "behavior.json").exists()
        ):
            run(api, source, dest, calibrate_only=args.calibrate_only)
