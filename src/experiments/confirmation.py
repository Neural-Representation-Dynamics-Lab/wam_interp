"""Blind feature checks and paired physical outcomes for the E1 confirmation study."""

import hashlib
import json
from contextlib import contextmanager
from time import perf_counter

import numpy as np
from wam_interp.evaluations.LIBERO.environment import step_libero

from .clip_study import save, start
from .confirm_sources import MOTION, TINTS


@contextmanager
def reuse_fastwam_video(api):
    """Reuse a pre-intervention denoiser call only when every native input is equal."""
    import torch

    if api.backend != "fastwam":
        yield
        return
    owner = api.owner
    native = owner._denoise_video
    previous = vars(owner).get("_denoise_video")
    cache = {}

    def forward(**kwargs):
        key = float(kwargs["timestep_video"].item())
        if key in cache:
            inputs, prediction = cache[key]
            matched = all(
                torch.equal(v, inputs[k]) if torch.is_tensor(v) else v == inputs[k]
                for k, v in kwargs.items()
            )
            if not matched:
                raise ValueError("A repeated pre-intervention denoising input changed")
            return prediction
        prediction = native(**kwargs)
        cache[key] = (
            {k: v.detach().clone() if torch.is_tensor(v) else v for k, v in kwargs.items()},
            prediction.detach().clone(),
        )
        return prediction

    owner._denoise_video = forward
    try:
        yield
    finally:
        if previous is None:
            del owner._denoise_video
        else:
            owner._denoise_video = previous


def measured_effects(paths):
    base = paths["native"]
    drift = np.linalg.norm(paths["sham"][-1] - base[-1])
    output = {"sham_drift_mm": float(1000 * drift)}
    for step in (10, 32):
        delta = paths["right_neutral"][step] - paths["left_neutral"][step]
        output[f"motion_{step}_mm"] = float(500 * delta[0])
    distances = []
    signed = []
    for direction in MOTION:
        neutral = paths[f"{direction}_neutral"]
        for tint in ("warm", "cool"):
            diff = paths[f"{direction}_{tint}"][1:] - neutral[1:]
            distances.append(float(1000 * np.sqrt(np.mean(np.sum(diff**2, axis=-1)))))
        warm = paths[f"{direction}_warm"][-1, 0]
        cool = paths[f"{direction}_cool"][-1, 0]
        signed.append(float(500 * (warm - cool)))
    output.update(
        background_rms_max_mm=max(distances),
        background_by_arm_mm=distances,
        signed_background_half_mm=float(np.mean(signed)),
        selectivity_mm=output["motion_32_mm"] - float(np.mean(np.abs(signed))),
    )
    if "right_fixed" in paths:
        output["fixed_wrist_motion_mm"] = float(
            500 * (paths["right_fixed"][-1, 0] - paths["left_fixed"][-1, 0])
        )
        output["fixed_wrist_sham_drift_mm"] = float(
            1000 * np.linalg.norm(paths["fixed_sham"][-1] - base[-1])
        )
    return output


def vectors(donors):
    def flat(k):
        return donors[k].parts[0].float().numpy().ravel()

    arm = flat("right_neutral") - flat("left_neutral")
    colour = np.mean([flat(d + "_warm") - flat(d + "_cool") for d in MOTION], axis=0)
    return {"arm": arm, "colour": colour}


def run_case(api, stage, backend, task, task_id, state, idx, seed, folder, manifest, source=None):
    import torch

    primary_source = manifest.get("source_backend", "fastwam")
    source = source or primary_source
    report = json.loads((folder / f"{source}_source.json").read_text())
    if not report["passed"]:
        return
    prefix = backend if source == primary_source else f"{backend}_own_source"
    receipt = folder / f"{prefix}_{stage}.json"
    if receipt.is_file():
        return
    if stage in {"behavior", "commands"} and manifest["split"] == "held_out":
        gate_path = folder.parents[2] / f"{backend}_feature_gate.json"
        gate = json.loads(gate_path.read_text())
        if not any(item["passed"] for item in gate["properties"].values()):
            save(receipt, {"eligible": False, "reason": "neither_property_identifiable"})
            return
    began = perf_counter()
    source_path = folder / f"{source}_source.npz"
    if hashlib.sha256(source_path.read_bytes()).hexdigest() != report["sha256"]:
        raise ValueError("Stimulus hash differs")
    noise_seed = 500000 + 1000 * task_id + 10 * idx + seed
    opts = {"noise_seed": noise_seed} if backend == "longwam" else {}
    keys = [f"{d}_{t}" for d in MOTION for t in TINTS]
    with (
        np.load(source_path, allow_pickle=False) as z,
        start(api, task, state, idx, task_id=task_id, seed=seed) as (_, _, initial, req),
    ):
        if stage in {"behavior", "commands"}:
            keys += ["sham"]
            if "fixed_sham" in z:
                keys += ["fixed_sham", "left_fixed", "right_fixed"]
        donors = {key: api.from_clip(req, z[key], **opts) for key in keys}
        anchor = req.observation
        depicted = float(500 * (z["right_source_path"][-1, 0] - z["left_source_path"][-1, 0]))
    if stage == "features":
        vec = vectors(donors)
        np.savez_compressed(folder / f"{prefix}_features.npz", **vec)
        save(
            receipt,
            {
                "seconds": perf_counter() - began,
                "rms": {k: float(np.sqrt(np.mean(v * v))) for k, v in vec.items()},
                "source_hash": report["sha256"],
                "noise_seed": opts.get("noise_seed"),
            },
        )
        return
    keys.append("native")
    np.random.default_rng(noise_seed + 901).shuffle(keys)
    if stage == "commands":
        from .confirm_replay import anchor_hash

        chunks = {}
        with reuse_fastwam_video(api):
            for key in keys:
                begin = getattr(api.adapter, "begin_episode", None)
                if begin:
                    begin(initial)
                result = (
                    api.adapter.predict_action(req)
                    if key == "native"
                    else api.predict_with_future(req, donors[key])
                )
                actions = api.adapter.postprocess_actions(result.actions, environment="libero")
                cmds = torch.as_tensor(actions).detach().cpu().float().numpy().reshape(-1, 7)
                if cmds.shape != (32, 7) or not np.isfinite(cmds).all():
                    raise ValueError("Invalid control chunk")
                chunks[key] = cmds
        np.savez_compressed(folder / f"{prefix}_commands.npz", **chunks)
        save(
            receipt,
            {
                "order": keys,
                "anchor_hash": anchor_hash(initial),
                "source_hash": report["sha256"],
                "depicted_half_mm": depicted,
                "action_seed": seed,
                "clip_noise_seed": opts.get("noise_seed"),
                "prediction_seconds": perf_counter() - began,
            },
        )
        return
    paths = {}
    command_chunks = {}
    for key in keys:
        with start(api, task, state, idx, task_id=task_id, seed=seed) as (env, meta, obs, req):
            if not (
                torch.equal(req.observation.image, anchor.image)
                and torch.equal(req.observation.proprio, anchor.proprio)
            ):
                raise ValueError("Unpaired recipient reset")
            result = (
                api.adapter.predict_action(req)
                if key == "native"
                else api.predict_with_future(req, donors[key])
            )
            cmds = api.adapter.postprocess_actions(result.actions, environment="libero")
            cmds = torch.as_tensor(cmds).float().cpu().numpy().reshape(-1, 7)
            if cmds.shape != (32, 7) or not np.isfinite(cmds).all():
                raise ValueError("Invalid control chunk")
            path = [obs.state["eef_position"].copy()]
            for cmd in cmds:
                step = step_libero(env, cmd, **meta)
                path.append(step.observation.state["eef_position"].copy())
                if step.done or step.success:
                    break
            if len(path) != 33:
                save(
                    receipt,
                    {
                        "eligible": False,
                        "reason": "early_termination",
                        "arm": key,
                        "length": len(path) - 1,
                    },
                )
                return
            paths[key] = np.stack(path)
            command_chunks[key] = cmds
    metrics = measured_effects(paths)
    np.savez_compressed(folder / f"{prefix}_paths.npz", **paths)
    np.savez_compressed(folder / f"{prefix}_commands.npz", **command_chunks)
    save(
        receipt,
        {
            "eligible": metrics["sham_drift_mm"] <= manifest.get("sham_margin_mm", 20),
            "metrics": metrics,
            "depicted_half_mm": depicted,
            "order": keys,
            "seconds": perf_counter() - began,
            "source_hash": report["sha256"],
            "action_seed": seed,
            "clip_noise_seed": opts.get("noise_seed"),
        },
    )
