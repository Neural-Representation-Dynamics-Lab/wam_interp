"""Compare opposite depicted robot motions with stationary objects in the book task."""
import json
from pathlib import Path

import numpy as np

from e1_native_replan import freeze, write
from e1_sliding_targets import render
from wam_interp.experiments.libero_pilot import digest_file, load_pilot
from wam_interp.experiments.simulator_libero import RobotContactCapture
from wam_interp.experiments.simulator_video import encode_reference_video


def record_motion(pilot, commands, objects, bias):
    with RobotContactCapture(pilot.env) as contacts:
        def sequence():
            contacts.enabled = True
            for command in commands:
                shifted = command.copy()
                shifted[0] = np.clip(shifted[0] + bias, -1, 1)
                yield shifted
        clip = render(pilot, sequence(), objects)
    clip["contact_substeps"] = len(contacts.events)
    return clip


def run():
    out = Path("results/e1_motion_control")
    out.mkdir(parents=True, exist_ok=True)
    cfg = json.loads(Path("configs/experiments/e1_libero.json").read_text())
    objects = ["black_book_1", "white_yellow_mug_1"]
    biases = [0.025, 0.05, 0.09, 0.15]
    freeze(out/"protocol.json", dict(script_sha256=digest_file(__file__), states=[0,1,2,3], seed=0,
           task=5, objects=objects, candidate_action_biases=biases, units="world meters",
           selection="First bias achieving opposite 2-4 cm EEF displacements; zero robot-scene contacts, at most 1 mm object drift and 1 cm lateral or vertical EEF drift at endpoint. Choose before edited recipient predictions.",
           scope="Secondary development control. Wrist viewpoint and arm pixels move with robot."))
    from wam_interp.evaluations.LIBERO.runner import _get_suite
    from wam_interp.evaluations.LIBERO.environment import (
        create_libero_environment, initial_states_for_task, reset_to_initial_state, step_libero,
    )
    from wam_interp.experiments.libero_pilot import LiberoPilot
    loaded = load_pilot(cfg)
    adapter = loaded.adapter
    loaded.close()
    suite = _get_suite("libero_10")
    task = suite.get_task(5)
    states = initial_states_for_task(suite,5)
    if "book" not in task.language:
        raise ValueError(task.language)
    for state in range(4):
        dst = out/f"state_{state}"
        dst.mkdir(exist_ok=True)
        if (dst/"case.json").is_file():
            continue
        pilot = LiberoPilot(dict(cfg,task_id=5,initial_state_index=state,action_seed=0), adapter,
                            task=task, initial_state=states[state], create_env=create_libero_environment,
                            reset_env=reset_to_initial_state, step_env=step_libero)
        try:
            native = pilot.session.run(None)
            commands = np.asarray(native.commands).reshape(32,7)
            native_robot = pilot.last_rollout["eef_position"].copy()
            sham = record_motion(pilot,commands,objects,0)
            if np.max(np.abs(native_robot-sham["robot"]))>1e-7:
                raise RuntimeError("Native motion replay differs")
            report = dict(state=state, instruction=task.language, probes=[], source_pass=False)
            for bias in biases:
                clips = {"sham":sham}
                checks = []
                for sign in (-1,1):
                    key = str(sign)
                    arm = record_motion(pilot,commands,objects,sign*bias)
                    clips[key] = arm
                    move = arm["robot"][-1]-sham["robot"][-1]
                    drift = float(np.linalg.norm(arm["objects"]-sham["objects"][0],axis=-1).max())
                    checks.append(dict(x_m=float(sign*move[0]), off_axis_m=float(np.linalg.norm(move[1:])),
                                       max_object_drift_m=drift, contacts=arm["contact_substeps"],
                                       current_exact=np.array_equal(arm["clip"][0],sham["clip"][0])))
                passed = sham["contact_substeps"]==0 and all(
                    .02<=r["x_m"]<=.04 and r["off_axis_m"]<=.01 and r["max_object_drift_m"]<=.001
                    and r["contacts"]==0 and r["current_exact"] for r in checks)
                report["probes"].append(dict(bias=bias, arms=checks, passed=passed))
                if passed:
                    report["source_pass"]=True
                    break
            if report["source_pass"]:
                paths = {"native":native_robot}
                np.savez_compressed(dst/"source_clips.npz",**{k:v["clip"] for k,v in clips.items()})
                for k in ("sham","-1","1"):
                    donor,_ = encode_reference_video(adapter.model,clips[k]["clip"],sham["clip"][0],native.future)
                    pilot.restore(pilot.saved_rng)
                    pilot.session.run(donor["future"])
                    paths[k]=pilot.last_rollout["eef_position"].copy()
                    if k=="sham":
                        report["sham_drift_m"]=float(np.linalg.norm(paths[k][-1]-native_robot[-1]))
                        report["sham_pass"]=report["sham_drift_m"]<=.02
                        if not report["sham_pass"]:
                            break
                if "1" in paths:
                    report["motion_half_contrast_m"]=float((paths["1"][-1,0]-paths["-1"][-1,0])/2)
                    report["depicted_half_contrast_m"]=float((clips["1"]["robot"][-1,0]-clips["-1"]["robot"][-1,0])/2)
                np.savez_compressed(dst/"execution.npz",**paths)
            write(dst/"case.json",report)
            print(dst, "source",report["source_pass"],"motion",report.get("motion_half_contrast_m"),flush=True)
        finally:
            pilot.close()

if __name__=="__main__":
    run()
