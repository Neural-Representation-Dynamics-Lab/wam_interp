"""Closed-loop diagnostic on the four previously studied LIBERO-Object physical swaps."""
from dataclasses import asdict
from functools import partial
import json
from pathlib import Path

import numpy as np

from e1_native_replan import freeze, reset_seed, write
from wam_interp.evaluations._libero_rollout import _max_steps, run_libero_episode
from wam_interp.evaluations.LIBERO.config import LiberoEvalConfig
from wam_interp.evaluations.LIBERO.environment import (
    create_libero_environment, initial_states_for_task, step_libero,
)
from wam_interp.evaluations.LIBERO.runner import _get_suite
from wam_interp.experiments.libero_pilot import digest_file, load_pilot
from wam_interp.experiments.physical_relocation import object_geometry_ids, reset_relocated


def run():
    import torch
    output = Path("results/e1_large_swap_replan")
    output.mkdir(parents=True, exist_ok=True)
    source = Path("configs/experiments/e1_physical_relocation.json")
    plan = json.loads(source.read_text())
    model_config = json.loads(Path("configs/experiments/e1_libero.json").read_text())
    suite = _get_suite(plan["suite"])
    task_id = plan["task_id"]
    task = suite.get_task(task_id)
    if task.language.strip().lower() != plan["instructions"]["soup"]:
        raise ValueError(f"Unexpected official instruction {task.language!r}")
    states = initial_states_for_task(suite, task_id)
    conditions = ("first_frame", "idm")
    freeze(output / "protocol.json", dict(suite=plan["suite"], task=task_id, instruction=task.language,
           states=plan["states"], variants=plan["variants"], conditions=list(conditions), seed=0,
           layout_sha256=digest_file(source), runner_sha256=digest_file(__file__),
           checkpoint_sha256=digest_file(model_config["checkpoint"]),
           statistics_sha256=digest_file(model_config["statistics"]), gpu=torch.cuda.get_device_name(),
           precision=model_config["precision"], stop="Official task predicate or suite time limit",
           scope="Exploratory stress test of native full replanning on previously studied four layouts"))
    pilot = load_pilot(model_config)
    adapter = pilot.adapter
    pilot.close()
    for state in plan["states"]:
        for variant in plan["variants"]:
            for condition in conditions:
                dest = output / f"state_{state}" / variant / condition
                if (dest / "episode.json").exists():
                    continue
                dest.mkdir(parents=True, exist_ok=True)
                reset_seed()
                cfg = LiberoEvalConfig(task_suite_name=plan["suite"], task_ids=(task_id,), num_trials=1,
                                       inference_options={"condition": condition})
                env = create_libero_environment(task, cfg.camera_resolution, 0)
                try:
                    objects = list(plan["instruction_objects"].values())
                    ids = env.env.obj_body_id
                    model = env.sim.model
                    geom_to_object = {int(i): obj for obj in objects for i in object_geometry_ids(model, ids[obj])}
                    gripper = {i for i in range(model.ngeom) if "gripper" in (model.geom_id2name(i) or "")}
                    if not gripper:
                        raise RuntimeError("No gripper geometry")
                    rows = []
                    def step(sim, action, **kwargs):
                        result = step_libero(sim, action, **kwargs)
                        data = sim.sim.data
                        hits = set()
                        for contact in data.contact[:data.ncon]:
                            a, b = int(contact.geom1), int(contact.geom2)
                            other = b if a in gripper else a if b in gripper else None
                            if other in geom_to_object:
                                hits.add(geom_to_object[other])
                        rows.append(dict(step=len(rows)+1-cfg.num_steps_wait, contacts=sorted(hits),
                                         eef=result.observation.state["eef_position"].tolist(),
                                         objects={obj:data.body_xpos[ids[obj]].tolist() for obj in objects}))
                        return result
                    reset = partial(reset_relocated, offsets=plan["layouts"][str(state)],
                                    swap_pair=plan["swap_objects"] if variant == "swapped" else ())
                    torch.cuda.reset_peak_memory_stats()
                    result = run_libero_episode(model=adapter, env=env, task=task, task_id=task_id,
                            trial_index=state, initial_state_index=state, initial_state=states[state],
                            config=cfg, seed=0, max_steps=_max_steps(cfg), reset_fn=reset, step_fn=step)
                    active = [row for row in rows if row["step"]>=0]
                    if not active or result.inference_calls < 1:
                        raise RuntimeError("No policy execution")
                    contact = next((r for r in active if r["contacts"]), None)
                    latency = [r["inference_latency_ms"] for r in result.inference_metadata]
                    report = dict(state=state, variant=variant, condition=condition, episode=asdict(result),
                           first_contact=None if contact is None else [contact["step"], contact["contacts"]],
                           first_target_contact=next((r["step"] for r in active if plan["instruction_objects"]["soup"] in r["contacts"]),None),
                           median_inference_ms=float(np.median(latency)),
                           peak_allocated_bytes=torch.cuda.max_memory_allocated(), relocation=env._relocation_audit)
                    write(dest/"tracks.json", active)
                    write(dest/"episode.json", report)
                    print(dest, "success", result.success, "first contact", report["first_contact"], flush=True)
                finally:
                    env.close()


if __name__ == "__main__":
    run()
