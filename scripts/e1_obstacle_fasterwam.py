"""Audit native one-pass noise with real obstacle encodings and measure live-scene steering."""

import argparse
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from e1_confirm import commands
from wam_interp.evaluations.LIBERO.environment import initial_states_for_task
from wam_interp.evaluations.LIBERO.runner import _get_suite
from wam_interp.experiments.clip_study import save
from wam_interp.experiments.confirm_replay import anchor_hash, omit_unused_rgb
from wam_interp.experiments.future_adapter import FutureAdapter
from wam_interp.experiments.obstacle import Barrier, score
from wam_interp.experiments.obstacle_libero import recipient_clip, rollout, scene
from wam_interp.models.fasterwam.libero_adapter import FasterWAMLiberoAdapter
from wam_interp.models.fasterwam.loader import compose_fasterwam_config, load_fasterwam
from wam_interp.types import InferenceRequest


def audit(api, req, clip):
    model = api.owner
    with torch.inference_mode():
        video = (
            torch.as_tensor(clip, device=model.device)
            .permute(3, 0, 1, 2)[None]
            .to(model.torch_dtype)
            * (2 / 255.0)
            - 1.0
        )
        if not torch.equal(video[:, :, 0], req.observation.image):
            raise ValueError("Faster-WAM recipient anchor failed")
        clean = model.vae.model.encode(video, model.vae.scale)[:, :, 1:]
    native = model.video_expert.pre_dit
    receipts = []

    def input_override(*args, **kwargs):
        x, t = kwargs["x"], kwargs["timestep"]
        noisy = model.infer_video_scheduler.add_noise(clean, x[:, :, 1:], t)
        if noisy.shape != x[:, :, 1:].shape or not torch.equal(noisy, x[:, :, 1:]):
            raise ValueError("Native timestep did not remove all clean signal")
        receipts.append(float(t.item()))
        kwargs["x"] = torch.cat((x[:, :, :1], noisy), 2)
        return native(*args, **kwargs)

    with patch.object(model.video_expert, "pre_dit", input_override):
        pred, donor = api.capture(req)
    if len(receipts) != 1:
        raise ValueError("Expected exactly one native video pass")
    return clean.detach().float().cpu(), commands(api, pred), donor, receipts[0]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    j = json.loads((a.source / "sources.json").read_text())
    if not j["passed"]:
        raise ValueError("Expected a matched source")
    cfg = compose_fasterwam_config(overrides=["model.redirect_common_files=false"])
    weights = Path("checkpoints/fasterwam_release/libero")
    model = load_fasterwam(cfg, weights / "step_021700.pt")
    api = FutureAdapter(
        FasterWAMLiberoAdapter(
            model,
            data_config=cfg.data.train,
            dataset_stats=weights / "dataset_stats.json",
        )
    )
    tid, idx, seed = j["task"], j["state"], j["seed"]
    suite = _get_suite("libero_10")
    task = suite.get_task(tid)
    state = initial_states_for_task(suite, tid)[idx]
    checks = {}
    paths = {}
    with np.load(a.source / "sources.npz") as clips:
        for live in ("clear", "minus", "plus"):
            box = (
                None
                if live == "clear"
                else Barrier(**j["placements"][live]["geometry"])
            )
            with scene(None, task, state, idx, tid, seed, box) as (env, meta, obs, _):
                expected = (
                    j["clear_anchor"]
                    if box is None
                    else j["placements"][live]["anchor"]
                )
                if anchor_hash(obs) != expected:
                    raise ValueError("Source/recipient mismatch")
                req = InferenceRequest(
                    api.adapter.prepare_observation(obs),
                    action_horizon=32,
                    **dict(api.adapter.inference_defaults(), seed=seed),
                )
                base, native = api.capture(req)
                cmd = commands(api, base)
                baseclean = None
                for name in (
                    ("clear", "minus", "plus") if box is None else ("clear", live)
                ):
                    clean, other, donor, t = audit(
                        api, req, recipient_clip(clips[name], clips[live])
                    )
                    if baseclean is None:
                        baseclean = clean
                    same = all(
                        torch.equal(x, y) for x, y in zip(native.parts, donor.parts)
                    )
                    equal = bool(np.array_equal(other, cmd))
                    rms = float((clean - baseclean).square().mean().sqrt())
                    checks[live + "__" + name] = {
                        "timestep": t,
                        "clean_edit_rms": rms,
                        "same_future_kv": same,
                        "same_commands": equal,
                    }
                    if not same or not equal or (name != "clear" and rms <= 0):
                        raise ValueError("Native one-pass audit failed")
                with omit_unused_rgb(env):
                    paths[live], _, _, _ = rollout(
                        env, meta, obs, cmd, barrier=bool(box)
                    )
    result = {
        "backend": "fasterwam",
        "condition": "one_pass_future_cache",
        "layer": api.layer,
        "source": str(a.source),
        "task": tid,
        "state": idx,
        "seed": seed,
        "native_checks": checks,
        "live_only_at_8": {
            s: score(
                paths[s], paths["clear"], Barrier(**j["placements"][s]["geometry"]), 8
            )
            for s in ("minus", "plus")
        },
    }
    np.savez_compressed(a.out / "paths.npz", **paths)
    save(a.out / "audit.json", result)


if __name__ == "__main__":
    main()
