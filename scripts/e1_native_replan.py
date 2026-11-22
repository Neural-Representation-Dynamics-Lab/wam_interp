"""Paired native Optional IDM LIBERO relocation diagnostic on official tasks."""
from dataclasses import asdict
from functools import partial
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess

import numpy as np

from wam_interp.evaluations._libero_rollout import run_libero_episode
from wam_interp.evaluations.LIBERO.config import LiberoEvalConfig
from wam_interp.evaluations.LIBERO.environment import (
    create_libero_environment, initial_states_for_task, reset_to_initial_state, step_libero,
)
from wam_interp.evaluations.LIBERO.runner import _get_suite
from wam_interp.experiments.futures import predict_with_future
from wam_interp.experiments.libero_pilot import digest_file, load_pilot
from wam_interp.experiments.physical_relocation import object_geometry_ids, reset_relocated

TARGETS = {"pick up the book": "black_book_1", "put the black bowl": "akita_black_bowl_1"}
STATES = (0, 1, 2, 3)
ROOT = Path("results/e1_native_replan")


def write(path, data):
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(data, indent=2, allow_nan=False), encoding="utf-8")
    temp.replace(path)


def freeze(path, data):
    if path.exists():
        if json.loads(path.read_text()) != data:
            raise ValueError(f"Study inputs changed at {path}")
    else:
        write(path, data)


def reset_seed():
    random.seed(0)
    np.random.seed(0)


def layouts(suite, task_id, name):
    task = suite.get_task(task_id)
    starts = initial_states_for_task(suite, task_id)
    reset_seed()
    env = create_libero_environment(task, 256, 0)
    try:
        env.reset()
        points = []
        for state in starts[:24]:
            env.set_init_state(state)
            points.append(env.sim.data.body_xpos[env.env.obj_body_id[name], :2].copy())
    finally:
        env.close()
    selections = []
    for state in STATES:
        options = []
        for donor in range(4, len(points)):
            delta = points[donor] - points[state]
            length = float(np.linalg.norm(delta))
            if 0.01 <= length <= 0.04:
                options.append((abs(length - 0.03), donor, delta.tolist()))
        if not options:
            raise ValueError(f"No 1–4 cm official alternate placement for task {task_id}, state {state}")
        _, donor, delta = min(options)
        selections.append(dict(state=state, donor_state=donor, xy_delta=delta,
                               initial_target_xy=points[state].tolist()))
    return selections


class Capture:
    def __init__(self, adapter, directory):
        self.adapter = adapter
        self.directory = directory
        self.call = 0

    def __getattr__(self, name):
        return getattr(self.adapter, name)

    def predict_action(self, request):
        if request.condition == "idm":
            result, future = predict_with_future(self.adapter, request)
            np.save(self.directory / f"future_{self.call:03d}.npy", future["used"].float().numpy())
        else:
            result = self.adapter.predict_action(request)
        self.call += 1
        return result


def episode(adapter, suite, task_id, name, layout, moved, condition, destination):
    import torch
    from PIL import Image
    from wam_interp.models.fastwam.libero_adapter import _rotate_180, _center_crop_resize
    destination.mkdir(parents=True, exist_ok=True)
    reset_seed()
    task = suite.get_task(task_id)
    env = create_libero_environment(task, 256, 0)
    state = initial_states_for_task(suite, task_id)[layout["state"]]
    cfg = LiberoEvalConfig(task_suite_name="libero_10", inference_options={"condition": condition})
    cfg.validate()
    rows = []
    calls = 0
    try:
        body_ids = env.env.obj_body_id
        geom_sets = {obj: set(object_geometry_ids(env.sim.model, body)) for obj, body in body_ids.items()}
        grip = {i for i in range(env.sim.model.ngeom) if "gripper" in (env.sim.model.geom_id2name(i) or "")}
        if not grip:
            raise ValueError("Cannot identify native gripper collision geometry")
        def observe_step(sim, action, **kwargs):
            nonlocal calls
            step = step_libero(sim, action, **kwargs)
            calls += 1
            t = calls - cfg.num_steps_wait
            if t >= 0:
                d = sim.sim.data
                contacted = set()
                for ct in d.contact[:d.ncon]:
                    a, b = int(ct.geom1), int(ct.geom2)
                    other = b if a in grip else a if b in grip else None
                    if other is not None:
                        contacted.update(obj for obj, ids in geom_sets.items() if other in ids)
                rows.append(dict(step=t, eef=step.observation.state["eef_position"].tolist(),
                                 objects={obj: d.body_xpos[i].tolist() for obj,i in body_ids.items()},
                                 contacts=sorted(contacted)))
                if condition == "idm" and t % 10 in (0, 4, 8):
                    # Match the training orientation/crop and camera order.
                    frames = [_center_crop_resize(_rotate_180(step.observation.cameras[cam]), width=224, height=224)
                              for cam in ("agentview", "wrist")]
                    Image.fromarray(np.concatenate(frames, axis=1)).save(destination / f"step_{t:03d}.png")
            return step
        reset = partial(reset_relocated, offsets={name: layout["xy_delta"]}) if moved else reset_to_initial_state
        torch.cuda.reset_peak_memory_stats()
        result = run_libero_episode(model=Capture(adapter, destination), env=env, task=task, task_id=task_id,
                    trial_index=layout["state"], initial_state_index=layout["state"], initial_state=state,
                    config=cfg, seed=0, max_steps=700, reset_fn=reset, step_fn=observe_step)
        if not rows or not result.inference_calls:
            raise RuntimeError("No policy execution")
        intended = np.array(layout["initial_target_xy"]) + (layout["xy_delta"] if moved else np.zeros(2))
        if np.linalg.norm(np.asarray(rows[0]["objects"][name])[:2] - intended) > 0.005:
            raise ValueError("Settled target exceeds 5 mm deviation from frozen position")
        hits = [row["step"] for row in rows if name in row["contacts"]]
        write(destination / "tracks.json", rows)
        report = dict(task_id=task_id, instruction=task.language, target=name, layout=layout, moved=moved,
                      condition=condition, config=asdict(cfg), episode=asdict(result), first_target_contact=min(hits,default=None),
                      relocation=getattr(env, "_relocation_audit", None),
                      peak_gpu_memory_bytes=torch.cuda.max_memory_allocated())
        write(destination / "episode.json", report)
        print(destination, "success", result.success, "controls", result.policy_steps, flush=True)
    finally:
        env.close()


def main():
    import torch
    import mujoco
    from wam_interp.evaluations.provenance import package_versions
    if "A100" not in torch.cuda.get_device_name() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("A100 with native bf16 required")
    ROOT.mkdir(parents=True, exist_ok=True)
    suite = _get_suite("libero_10")
    plan = []
    for task_id in range(10):
        language = suite.get_task(task_id).language
        for prefix, target in TARGETS.items():
            if language.startswith(prefix):
                plan.append(dict(task_id=task_id, instruction=language, target=target, layouts=layouts(suite, task_id, target)))
    if len(plan) != 2:
        raise ValueError(f"Unexpected task mapping: {plan}")
    config = json.loads(Path("configs/experiments/e1_libero.json").read_text())
    freeze(ROOT / "protocol.json", dict(plan=plan, conditions=["first_frame","idm"], layouts=["official","relocated"],
           steps=700, replan_steps=10, warmup=30, seed=0, units="world meters", contact_sampling="control boundaries",
           selection="nearest to 3 cm in [1,4] cm among official states 4 to 23, before outcomes",
           script_sha256=digest_file(__file__), checkpoint_sha256=digest_file(config["checkpoint"]),
           statistics_sha256=digest_file(config["statistics"]), source=subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip(),
           libero_revision=subprocess.check_output(["git","-C","third_party/LIBERO","rev-parse","HEAD"],text=True).strip(),
           gpu=torch.cuda.get_device_name(), mujoco=mujoco.__version__, packages=package_versions()))
    pilot = load_pilot(config)
    adapter = pilot.adapter
    pilot.close()
    for item in plan:
        for layout in item["layouts"]:
            for moved in (False,True):
                for condition in ("first_frame","idm"):
                    dst = ROOT / f"task_{item['task_id']}/state_{layout['state']}/{'relocated' if moved else 'official'}/{condition}"
                    if (dst / "episode.json").exists():
                        continue
                    episode(adapter, suite, item["task_id"], item["target"], layout, moved, condition, dst)


if __name__ == "__main__":
    main()
