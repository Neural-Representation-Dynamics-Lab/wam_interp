"""Run the paired E1 controls on expert-screened RoboTwin scenes with Long-WAM IDM."""

import argparse
import hashlib
import json
from pathlib import Path
from time import monotonic

import numpy as np
from e1_robotwin_clip import api_for, joints, request
from wam_interp.evaluations.ROBOTWIN.config import RoboTwinEvalConfig
from wam_interp.evaluations.ROBOTWIN.environment import open_environment
from wam_interp.experiments.clip_study import save
from wam_interp.experiments.confirm_replay import anchor_hash
from wam_interp.experiments.confirmation import measured_effects, vectors
from wam_interp.experiments.robotwin_motion import CAMERAS, can_skip_images, physical_trace, record


def reset(front, api, row):
    front.close()
    obs = front.begin(row["seed"], 0, row["info"], "unseen")
    api.adapter.begin_episode(obs)
    if obs.instruction != row["instruction"] or str(front.env.arm_tag) != row["active_arm"]:
        raise ValueError("Expert-screened context changed")
    return obs


def source_parity(full, sampled):
    old, old_path, old_log = full
    new, new_path, new_log = sampled
    pixels = {}
    for tint in old:
        if len(old[tint]) != len(new[tint]):
            return {"passed": False, "reason": "different_frame_count"}
        pixels[tint] = {
            cam: max(int(np.max(np.abs(a[cam].astype(int) - b[cam].astype(int))))
                     for a, b in zip(old[tint], new[tint], strict=True))
            for cam in CAMERAS
        }
    error = float(np.abs(old_path - new_path).max()) if old_path.shape == new_path.shape else None
    return {
        "max_pixel_difference": pixels, "max_abs_both_tcp_m": error, "same_audit": old_log == new_log,
        "passed": bool(error is not None and error <= 1e-7 and old_log == new_log
                       and all(max(v.values()) == 0 for v in pixels.values())),
    }


def sources(front, api, row, seed, out, *, development):
    marker = out / "longwam_source.json"
    if marker.is_file():
        return
    obs = reset(front, api, row)
    anchor = anchor_hash(obs)
    req = request(api, obs, seed)
    cmds = joints(api, api.adapter.predict_action(req))
    stride = api.adapter.sample_stride
    side = ("left", "right").index(row["active_arm"])
    audit = {}
    clips = {}
    parity_file = out.parents[3] / "source_render_parity.json"
    sampled_only = parity_file.is_file() and json.loads(parity_file.read_text())["passed"]
    for key, dx in (("sham", 0.0), ("left", -0.03), ("right", 0.03)):
        if key != "sham":
            obs = reset(front, api, row)
            if anchor_hash(obs) != anchor:
                raise ValueError("Donor initial state changed")
        full = record(front, obs, cmds, row["active_arm"], dx, stride, sampled_only=sampled_only)
        if development and key == "left" and not parity_file.is_file() and can_skip_images(front):
            obs = reset(front, api, row)
            if anchor_hash(obs) != anchor:
                raise ValueError("Development source anchor changed before rendering comparison")
            sampled = record(front, obs, cmds, row["active_arm"], dx, stride, sampled_only=True)
            parity = source_parity(full, sampled)
            save(parity_file, dict(parity, scene_seed=row["seed"], action_seed=seed, anchor_hash=anchor))
        variants, path, log = full
        if key != "sham":
            from wam_interp.models.longwam.transforms import robotwin_model_image

            native = []
            for plain, tinted in zip(variants["neutral"][1:], variants["warm"][1:]):

                def prepare(frame):
                    return robotwin_model_image(
                        frame,
                        api.adapter.image_meta,
                        device=api.owner.device,
                        dtype=api.owner.torch_dtype,
                    ).float()

                diff = (prepare(plain) - prepare(tinted)).abs()
                native.append((float(diff.gt(0).any(dim=1).float().mean()), float(diff.mean() * 127.5)))
            mean = np.mean(native, axis=0) if native else (0.0, 0.0)
            log.update(prepared_changed_fraction=float(mean[0]), prepared_pixel_mae=float(mean[1]))
        audit[key] = log
        clips[key + "_path"] = path
        if key == "sham":
            for cam in CAMERAS:
                clips["sham_" + cam] = np.stack([f[cam] for f in variants["neutral"]])
        else:
            for tint, frames in variants.items():
                for cam in CAMERAS:
                    clips[f"{key}_{tint}_{cam}"] = np.stack([f[cam] for f in frames])
    passed = True
    target = []
    for key, sign in (("left", -1), ("right", 1)):
        base = clips["sham_path"]
        difference = clips[key + "_path"][-1] - base[-1]
        audit[key]["both_endpoint_delta_m"] = difference.tolist()
        audit[key]["duration_ratio"] = audit[key]["elapsed_seconds"] / max(
            1e-6, audit["sham"]["elapsed_seconds"]
        )
        target.append(difference[side, 0])
        passed &= (
            0.02 <= sign * difference[side, 0] <= 0.04
            and np.linalg.norm(difference[side, 1:]) <= 0.01
            and np.linalg.norm(difference[1 - side]) <= 0.01
            and audit[key]["changed_fraction"] >= 0.005
            and audit[key]["prepared_changed_fraction"] >= 0.005
            and audit[key]["prepared_pixel_mae"] >= 0.1
            and audit[key]["pixel_mae"] >= 0.1
            and audit[key]["mask_leak_pixels"] == 0
        )
        for cam in CAMERAS:
            passed &= np.array_equal(clips["sham_" + cam][0], clips[key + "_neutral_" + cam][0])
    passed &= all(
        s["controls"] == 32 and s["contact_substeps"] == 0 and s["object_drift_m"] <= 0.001
        for s in audit.values()
    )
    duration = [s["elapsed_seconds"] for s in audit.values()]
    # Control-space comparisons use the same 32 sample indices. Large timing changes confound motion.
    passed &= max(duration) <= 1.15 * min(duration)
    result = {
        "passed": bool(passed),
        "arms": audit,
        "anchor_hash": anchor,
        "active_arm": row["active_arm"],
        "rendered_at_sample_times_only": bool(sampled_only),
        "depicted_half_mm": float(500 * (target[1] - target[0])),
        "instruction": obs.instruction,
    }
    if passed:
        np.savez_compressed(out / "longwam_source.npz", **clips)
        result["sha256"] = hashlib.sha256((out / "longwam_source.npz").read_bytes()).hexdigest()
    save(marker, result)


def verify_physical_readout(front, api, row, seed, out, source):
    """Compare one real development policy chunk in this task before suppressing unused images."""
    obs = reset(front, api, row)
    expected = source["anchor_hash"]
    if anchor_hash(obs) != expected:
        raise ValueError("Development anchor changed before readout calibration")
    cmds = joints(api, api.adapter.predict_action(request(api, obs, seed)))
    runs = []
    for skip in (False, True):
        if anchor_hash(reset(front, api, row)) != expected:
            raise ValueError("Readout comparison requires identical initial observations")
        start = monotonic()
        path, contacts, termination, skipped = physical_trace(front, cmds, skip_images=skip)
        runs.append((path, contacts, termination, skipped, monotonic() - start))
    full, fast = runs
    error = float(np.max(np.abs(full[0] - fast[0]))) if full[0].shape == fast[0].shape else None
    save(out, {
        "task": front.config.task_name, "scene_seed": row["seed"], "action_seed": seed,
        "anchor_hash": expected, "commands_sha256": hashlib.sha256(cmds.tobytes()).hexdigest(),
        "max_abs_both_tcp_m": error, "contacts": [full[1], fast[1]],
        "termination": [full[2], fast[2]], "seconds": [full[4], fast[4]],
        "passed": bool(fast[3] and error is not None and error <= 1e-7
                       and full[1] == fast[1] and full[2] == fast[2]),
    })


def score(front, api, row, seed, out, spec, stage):
    marker = out / f"longwam_{stage}.json"
    if marker.is_file():
        return
    source = json.loads((out / "longwam_source.json").read_text())
    if not source["passed"]:
        return
    if stage == "behavior" and spec["split"] == "held_out":
        g = json.loads((out.parents[2] / "longwam_feature_gate.json").read_text())
        if not any(v["passed"] for v in g["properties"].values()):
            save(marker, {"eligible": False, "reason": "neither_property_identifiable"})
            return
    began = monotonic()
    obs = reset(front, api, row)
    if anchor_hash(obs) != source["anchor_hash"]:
        raise ValueError("New recipient differs")
    req = request(api, obs, seed)
    noise = 10 * row["seed"] + seed
    keys = [f"{d}_{t}" for d in ("left", "right") for t in ("neutral", "warm", "cool")]
    if stage == "behavior":
        keys += ["sham"]
    donors = {}
    source_path = out / "longwam_source.npz"
    if hashlib.sha256(source_path.read_bytes()).hexdigest() != source["sha256"]:
        raise ValueError("Stimulus changed")
    with np.load(source_path, allow_pickle=False) as z:
        for key in keys:
            views = {c: z[f"{key}_{c}"] for c in CAMERAS}
            frames = [{c: views[c][t] for c in CAMERAS} for t in range(len(views["head"]))]
            donors[key] = api.from_camera_frames(req, frames, noise_seed=noise)
    if stage == "features":
        vec = vectors(donors)
        np.savez_compressed(out / "longwam_features.npz", **vec)
        save(
            marker,
            {
                "seconds": monotonic() - began,
                "source_hash": source["sha256"],
                "rms": {k: float(np.sqrt(np.mean(v * v))) for k, v in vec.items()},
            },
        )
        return
    keys.append("native")
    np.random.default_rng(noise + 901).shuffle(keys)
    paths = {}
    contact = {}
    chunks = {}
    parity_file = out.parents[3] / "physical_readout_parity.json"
    if spec["split"] != "held_out" and not parity_file.exists():
        verify_physical_readout(front, api, row, seed, parity_file, source)
    skip_images = parity_file.is_file() and json.loads(parity_file.read_text())["passed"]
    for key in keys:
        obs = reset(front, api, row)
        if anchor_hash(obs) != source["anchor_hash"]:
            raise ValueError("Recipient reset changed")
        req = request(api, obs, seed)
        result = (
            api.adapter.predict_action(req)
            if key == "native"
            else api.predict_with_future(req, donors[key])
        )
        cmds = joints(api, result)
        chunks[key] = cmds
        paths[key], log, _, skipped = physical_trace(front, cmds, skip_images=skip_images)
        contact[key] = log
    np.savez_compressed(out / "longwam_paths.npz", **paths)
    np.savez_compressed(out / "longwam_commands.npz", **chunks)
    result = {
        "source_hash": source["sha256"],
        "order": keys,
        "skipped_unused_images": skipped,
        "depicted_half_mm": source["depicted_half_mm"],
        "contacts": contact,
        "action_seed": seed,
        "clip_noise_seed": noise,
        "seconds": monotonic() - began,
    }
    if all(len(v) == 33 for v in paths.values()):
        side = ("left", "right").index(row["active_arm"])
        metrics = measured_effects({k: v[:, side] for k, v in paths.items()})
        result.update(
            eligible=metrics["sham_drift_mm"] <= spec.get("sham_margin_mm", 20),
            metrics=metrics,
            other_arm=measured_effects({k: v[:, 1 - side] for k, v in paths.items()}),
        )
    else:
        result.update(eligible=False, reason="early_termination")
    save(marker, result)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage", choices=("sources", "features", "behavior", "calibrate"))
    for name in ("root", "weights", "manifest", "out", "screen-dir"):
        p.add_argument("--" + name, type=Path, required=True)
    a = p.parse_args()
    spec = json.loads(a.manifest.read_text())
    root = a.root.resolve()
    dest = a.out.resolve()
    screened = a.screen_dir.resolve()
    api = api_for(a.weights.resolve())
    for task in spec["tasks"]:
        replay = json.loads((screened / f"smoke_{task}/fidelity.json").read_text())
        if not replay.get("passed"):
            raise ValueError(f"Undemonstrated unedited-video fidelity for {task}")
        rows = json.loads((screened / f"{spec['screen_split']}_{task}.json").read_text())
        chosen = {r["seed"]: r for r in rows if r["accepted"]}
        with open_environment(RoboTwinEvalConfig(task_name=task, robotwin_root=root)) as front:
            for idx in spec["states"][task]:
                for seed in spec["seeds"]:
                    folder = dest / f"task_{task}/state_{idx}/seed_{seed}"
                    folder.mkdir(parents=True, exist_ok=True)
                    if a.stage in ("sources", "calibrate"):
                        sources(front, api, chosen[idx], seed, folder, development=spec["split"] == "development")
                    if a.stage != "sources":
                        for stage in ("features", "behavior") if a.stage == "calibrate" else (a.stage,):
                            score(front, api, chosen[idx], seed, folder, spec, stage)
                    print("completed", task, idx, seed, a.stage, flush=True)


if __name__ == "__main__":
    main()
