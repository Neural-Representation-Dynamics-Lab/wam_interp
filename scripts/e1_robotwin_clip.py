"""Check native task success, identity replay, and the unedited video path on RoboTwin."""

import argparse
import json
from pathlib import Path

import numpy as np
from wam_interp.evaluations.ROBOTWIN.config import RoboTwinEvalConfig
from wam_interp.evaluations.ROBOTWIN.environment import open_environment
from wam_interp.experiments.clip_study import save
from wam_interp.experiments.future_adapter import FutureAdapter
from wam_interp.types import InferenceRequest


def api_for(weights):
    from wam_interp.models.longwam.loader import load_longwam
    from wam_interp.models.longwam.robotwin_adapter import LongWAMRobotWinAdapter

    owner = load_longwam(weights / "config.yaml", weights / "model.pt", device="cuda", dtype="bfloat16")
    return FutureAdapter(LongWAMRobotWinAdapter(owner, dataset_stats=weights / "dataset_stats.json"))


def request(api, obs, seed):
    options = dict(api.adapter.inference_defaults())
    options.update(condition="idm", seed=seed)
    return InferenceRequest(api.adapter.prepare_observation(obs), action_horizon=32, **options)


def joints(api, result):
    cmds = api.adapter.postprocess_actions(result.actions, environment="robotwin")
    cmds = np.asarray(cmds)
    if cmds.shape != (32, 14) or not np.isfinite(cmds).all():
        raise ValueError("Expected 32 finite 14-D joint commands")
    return cmds


def tcp(env):
    return np.array([env.robot.get_left_tcp_pose()[:3], env.robot.get_right_tcp_pose()[:3]], dtype=float)


def smoke(api, rt, task, row, out):
    import torch

    seed = row["seed"]
    cfg = RoboTwinEvalConfig(task_name=task, robotwin_root=rt)
    with open_environment(cfg) as front:

        def reset():
            front.close()
            obs = front.begin(seed, 0, row["info"], "unseen")
            api.adapter.begin_episode(obs)
            return obs

        obs = reset()
        req = request(api, obs, seed)
        base = api.adapter.predict_action(req)
        captured, marker = api.capture(req)
        replay = api.predict_with_future(req, marker)
        diagnostics = {
            "seed": seed,
            "task": task,
            "arm": row["active_arm"],
            "instruction": obs.instruction,
            "capture_max_abs": float((base.actions - captured.actions).abs().max()),
            "replay_max_abs": float((base.actions - replay.actions).abs().max()),
            "sample_stride": api.adapter.sample_stride,
        }
        save(out, diagnostics)
        if diagnostics["capture_max_abs"] or diagnostics["replay_max_abs"]:
            raise RuntimeError("RoboTwin native replay differs")
        anchor = req.observation.image.detach().clone()
        anchor_state = req.observation.proprio.detach().clone()
        frames = [{k: v.copy() for k, v in obs.cameras.items()}]
        path = [tcp(front.env)]
        for i, cmd in enumerate(joints(api, base), 1):
            obs, success = front.step(cmd, "qpos")
            path.append(tcp(front.env))
            if i % api.adapter.sample_stride == 0:
                frames.append({k: v.copy() for k, v in obs.cameras.items()})
            if success:
                raise ValueError("Smoke source terminated early")
        obs = reset()
        req = request(api, obs, seed)
        diagnostics.update(
            anchor_rgb_exact=bool(torch.equal(req.observation.image, anchor)),
            anchor_state_exact=bool(torch.equal(req.observation.proprio, anchor_state)),
            anchor_image_max_abs=float((req.observation.image - anchor).abs().max()),
            camera_frames=len(frames),
        )
        save(out, diagnostics)
        if not diagnostics["anchor_rgb_exact"] or not diagnostics["anchor_state_exact"]:
            raise ValueError("Reset does not reproduce the recipient observation")
        donor = api.from_camera_frames(req, frames, noise_seed=seed + 20261123)
        sham_path = [tcp(front.env)]
        for cmd in joints(api, api.predict_with_future(req, donor)):
            _, success = front.step(cmd, "qpos")
            sham_path.append(tcp(front.env))
            if success:
                break
        if len(sham_path) != len(path):
            raise ValueError("Unmatched physical horizon")
        endpoint = 1000 * np.linalg.norm(np.array(sham_path)[-1] - np.array(path)[-1], axis=-1)
        side = ("left", "right").index(row["active_arm"])
        diagnostics.update(sham_drift_mm=endpoint.tolist(), sham_pass=bool(endpoint[side] <= 20))
        np.savez_compressed(out.with_suffix(".npz"), native=path, sham=sham_path)
        save(out, diagnostics)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--weights", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--task", required=True)
    p.add_argument("--screen", type=Path, required=True)
    args = p.parse_args()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    root = args.root.resolve()
    api = api_for(args.weights.resolve())
    rows = [x for x in json.loads(args.screen.read_text()) if x["accepted"]]
    from wam_interp.evaluations.ROBOTWIN.runner import evaluate_robotwin

    # The expert planner can reject a previously accepted seed when it plans again.
    # Preserve each official evaluation file; try only the screened development starts.
    attempts = []
    for row in rows:
        cfg = RoboTwinEvalConfig(
            task_name=args.task,
            robotwin_root=root,
            start_seed=row["seed"],
            replan_steps=24,
            max_seed_attempts=1,
            inference_options={"condition": "idm"},
            output_dir=out,
        )
        try:
            evaluate_robotwin(api.adapter, cfg)
        except RuntimeError as exc:
            if "Seed search exhausted after 1 attempts" not in str(exc):
                raise
            attempts.append({"seed": row["seed"], "status": "expert_rejected_on_repeat"})
            save(out / "native_attempts.json", attempts)
            continue
        attempts.append({"seed": row["seed"], "status": "episode_completed"})
        save(out / "native_attempts.json", attempts)
        break
    else:
        raise RuntimeError("No previously expert-accepted development start passed repeated screening")
    rows = [row] + [candidate for candidate in rows if candidate["seed"] != row["seed"]]
    tested = []
    for i, candidate in enumerate(rows):
        marker = out / ("replay.json" if i == 0 else f"replay_{candidate['seed']}.json")
        smoke(api, root, args.task, candidate, marker)
        result = json.loads(marker.read_text())
        tested.append(result)
        if result["sham_pass"]:
            break
    save(out / "fidelity.json", {"passed": any(x["sham_pass"] for x in tested), "starts": tested})


if __name__ == "__main__":
    main()
