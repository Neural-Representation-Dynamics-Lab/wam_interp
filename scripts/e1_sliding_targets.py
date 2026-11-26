"""Fixed-motion film test using native physics after a common observed frame.

The sole scene intervention is an initial planar velocity applied to a free
object after frame zero. Render all donors under identical native robot commands.
"""
from dataclasses import replace
import json
from pathlib import Path
import numpy as np

from analyze_native_replan import summarize
from e1_native_replan import ROOT, write, freeze
from wam_interp.experiments.libero_pilot import LiberoPilot, load_pilot, digest_file
from wam_interp.experiments.simulator_video import paired_camera_clip, encode_reference_video
from wam_interp.experiments.physical_relocation import NativeObjectCapture
from wam_interp.evaluations.LIBERO.environment import (
    create_libero_environment, initial_states_for_task, reset_to_initial_state, step_libero,
)
from wam_interp.evaluations.LIBERO.runner import _get_suite

OUT = Path("results/e1_sliding_targets")
DISTRACTORS = {3:"wine_bottle_1",5:"white_yellow_mug_1"}
ALTERNATE = {"wine_bottle_1":"pick up the wine bottle","white_yellow_mug_1":"pick up the yellow and white mug"}


def alternative_contact(pilot,prompt,objects):
    from wam_interp.experiments.physical_relocation import object_geometry_ids
    pilot.restore(pilot.saved_rng)
    env=pilot.env
    sim=env.sim
    bodies={k:set(object_geometry_ids(sim.model,env.env.obj_body_id[k])) for k in objects}
    grip={i for i in range(sim.model.ngeom) if "gripper" in (sim.model.geom_id2name(i) or "")}
    obs=pilot.observation
    for chunk in range(16):
        prepared=replace(pilot.adapter.prepare_observation(obs),prompt=prompt)
        request=replace(pilot.request,observation=prepared,seed=pilot.config["action_seed"]+chunk)
        result=pilot.adapter.predict_action(request)
        commands=np.asarray(pilot.adapter.postprocess_actions(result.actions,environment="libero")).reshape(-1,7)
        for i,c in enumerate(commands[:10],1):
            step=step_libero(env,c,instruction=prompt,metadata={})
            obs=step.observation
            touched=set()
            for contact in sim.data.contact[:sim.data.ncon]:
                a,b=int(contact.geom1),int(contact.geom2)
                other=b if a in grip else a if b in grip else None
                touched.update(name for name,ids in bodies.items() if other in ids)
            if touched or step.done:
                return {"first_objects":sorted(touched),"step":10*chunk+i,"valid":touched=={objects[1]}}
    return {"first_objects":[],"step":160,"valid":False}


def render(pilot, commands, objects, changed=None, velocity=0.):
    pilot.restore(pilot.saved_rng)
    env = pilot.env
    sim = env.sim
    obs = pilot.observation
    frames = {c:[obs.cameras[c].copy()] for c in ("agentview","wrist")}
    ids = [env.env.obj_body_id[k] for k in objects]
    positions = [sim.data.body_xpos[ids].copy()]
    orientations = [sim.data.body_xquat[ids].copy()]
    robot = [obs.state["eef_position"].copy()]
    robot_joints = np.asarray(env.env.robots[0]._ref_joint_pos_indexes)
    joints = [sim.data.qpos[robot_joints].copy()]
    pixels = []
    if changed:
        joint = np.flatnonzero(sim.model.jnt_bodyid == env.env.obj_body_id[changed])
        if len(joint)!=1 or sim.model.jnt_type[joint[0]] != 0:
            raise ValueError("A free object is required")
        address = sim.model.jnt_dofadr[joint[0]]
        sim.data.qvel[address] += velocity
        sim.forward()
    with NativeObjectCapture(env,objects) as sensor:
        for step, command in enumerate(commands,1):
            sensor.enabled = step % 4 == 0
            result = step_libero(env,command,instruction=pilot.task.language,metadata={})
            if result.done or result.success:
                raise ValueError("Donor episode terminated inside observation window")
            obs = result.observation
            robot.append(obs.state["eef_position"].copy())
            joints.append(sim.data.qpos[robot_joints].copy())
            positions.append(sim.data.body_xpos[ids].copy())
            orientations.append(sim.data.body_xquat[ids].copy())
            if sensor.enabled:
                for cam in frames:
                    frames[cam].append(obs.cameras[cam].copy())
                pixels.append(sensor.latest["agentview"]["visible_pixels"].copy())
    return {"clip":paired_camera_clip(frames["agentview"],frames["wrist"]),
            "robot":np.array(robot),"joints":np.array(joints),"objects":np.array(positions),
            "orientations":np.array(orientations),"pixels":pixels}


def prescribed_cases(pilot,commands,objects,native_robot):
    from wam_interp.experiments.translated_views import TranslatedViews
    pilot.restore(pilot.saved_rng)
    keys=["sham",*[f"{obj}_{sign}" for obj in objects for sign in (-1,1)]]
    frames={k:{cam:[pilot.observation.cameras[cam].copy()] for cam in ("agentview","wrist")} for k in keys}
    joints=[pilot.observation.state["eef_position"].copy()]
    with TranslatedViews(pilot.env,objects) as views:
        for step, command in enumerate(commands,1):
            views.step=step
            views.capture=(step%4==0)
            outcome=step_libero(pilot.env,command,instruction=pilot.task.language,metadata={})
            joints.append(outcome.observation.state["eef_position"].copy())
            if views.capture:
                for key in keys:
                    for cam,sensor in (("agentview","agentview"),("wrist","robot0_eye_in_hand")):
                        frames[key][cam].append(views.views[key,sensor].copy())
    drift=float(np.max(np.abs(np.asarray(joints)-native_robot)))
    cases={k:{"clip":paired_camera_clip(v["agentview"],v["wrist"])} for k,v in frames.items()}
    checks=[]
    for key in keys[1:]:
        delta=np.abs(cases[key]["clip"].astype(float)-cases["sham"]["clip"])
        mass=float(delta[1:].mean())
        checks.append(dict(variant=key,current_exact=not bool(delta[0].any()),robot_max_m=drift,
                           mean_pixel_delta=mass,signed_axis_motion_m=.03,
                           minimum_sampled_clearance_m=views.min_clearance[key],visible=views.visibility[key],
                           passed=bool(not delta[0].any() and drift<=1e-7 and mass>0 and
                                       views.min_clearance[key]>=-.0002 and views.visibility[key]>=16)))
    return cases,checks


def state_case(pilot, state, seed, target, distractor, dest, speeds, prescribed):
    dest.mkdir(parents=True,exist_ok=True)
    native = pilot.session.run(None)
    commands = np.asarray(native.commands).reshape(-1,7)
    if commands.shape != (32,7):
        raise ValueError("Expected native 32-control source")
    native_robot = pilot.last_rollout["eef_position"].copy()
    objects = [target,distractor]
    base = render(pilot,commands,objects)
    if np.max(np.abs(base["robot"]-native_robot))>1e-7:
        raise ValueError("Reference replay drift")
    cases = {"sham":base}
    checks = []
    for i, obj in enumerate([] if prescribed else objects):
        for sign in (-1,1):
            name = f"{obj}_{sign}"
            edited = render(pilot,commands,objects,obj,sign*speeds[obj])
            cases[name] = edited
            diff = edited["objects"]-base["objects"]
            extent = float(sign*diff[-1,i,0])
            dot = np.abs((edited["orientations"][:,i]*base["orientations"][:,i]).sum(axis=-1))
            angle = float(np.rad2deg(2*np.arccos(np.clip(dot,0,1))).max())
            mass = float(np.abs(edited["clip"][1:].astype(float)-base["clip"][1:]).mean())
            check = dict(variant=name, signed_axis_motion_m=extent,
                         current_exact=np.array_equal(base["clip"][0],edited["clip"][0]),
                         robot_max_m=float(np.abs(edited["robot"]-base["robot"]).max()),
                         joints_max_rad=float(np.abs(edited["joints"]-base["joints"]).max()),
                         rotation_max_deg=angle, vertical_max_m=float(np.abs(diff[:,i,2]).max()),
                         other_max_m=float(np.abs(diff[:,1-i]).max()),
                         visible=min(x[obj] for x in edited["pixels"]), mean_pixel_delta=mass)
            check["passed"] = (check["current_exact"] and .012 <= extent <= .065 and
                               check["robot_max_m"]<=.0005 and check["joints_max_rad"]<=.002 and
                               check["other_max_m"]<=.001 and check["vertical_max_m"]<=.003 and angle<=8
                               and check["visible"]>=16 and mass>0)
            checks.append(check)
    if prescribed:
        cases,checks=prescribed_cases(pilot,commands,objects,native_robot)
    ranges = [x["signed_axis_motion_m"] for x in checks]
    masses = [x["mean_pixel_delta"] for x in checks]
    matched = min(ranges)>0 and max(ranges)/min(ranges)<=2.5 and min(masses)>0 and max(masses)/min(masses)<=3
    report = dict(state=state, seed=seed, target=target,distractor=distractor,checks=checks,
                  matched_physical_and_visual_edits=bool(matched), human_review="pending",
                  source_pass=bool(matched and all(x["passed"] for x in checks)))
    np.savez_compressed(dest/"source_clips.npz",**{k:v["clip"] for k,v in cases.items()})
    if report["source_pass"]:
        measured = {}
        donors = {}
        for key,case in cases.items():
            codec, meta = encode_reference_video(pilot.adapter.model,case["clip"],base["clip"][0],native.future)
            donors[key] = codec["future"]
            pilot.restore(pilot.saved_rng)
            result = pilot.session.run(codec["future"])
            measured[key] = pilot.last_rollout["eef_position"].copy()
            report.setdefault("codec",{})[key]=meta
            if key == "sham":
                report["sham_native_endpoint_m"] = float(np.linalg.norm(measured[key][-1]-native_robot[-1]))
                report["sham_pass"] = report["sham_native_endpoint_m"]<=.02
                if not report["sham_pass"]:
                    break
        np.savez_compressed(dest/"execution.npz",native=native_robot,**measured)
        if len(measured)==5:
            relevant = float((measured[target+"_1"][-1,0]-measured[target+"_-1"][-1,0])/2)
            nuisance = float((measured[distractor+"_1"][-1,0]-measured[distractor+"_-1"][-1,0])/2)
            report.update(relevant_half_contrast_m=relevant,nuisance_half_contrast_m=nuisance,
                          selective_half_contrast_m=relevant-nuisance)
            prompt=ALTERNATE[distractor]
            rec=alternative_contact(pilot,prompt,objects)
            report["alternate_instruction"]={"prompt":prompt,"native_first_contact":rec}
            if rec["valid"]:
                session=pilot.session
                old=session.request
                try:
                    session.request=replace(old,observation=replace(old.observation,prompt=prompt))
                    paths={}
                    for key,donor in [("native",None),*donors.items()]:
                        pilot.restore(pilot.saved_rng)
                        session.run(donor)
                        paths[key]=pilot.last_rollout["eef_position"].copy()
                        if key=="sham" and np.linalg.norm(paths[key][-1]-paths["native"][-1])>.02:
                            break
                    rec2=report["alternate_instruction"]
                    rec2["sham_native_endpoint_m"]=float(np.linalg.norm(paths["sham"][-1]-paths["native"][-1]))
                    if len(paths)==6:
                        ra=float((paths[target+"_1"][-1,0]-paths[target+"_-1"][-1,0])/2)
                        rb=float((paths[distractor+"_1"][-1,0]-paths[distractor+"_-1"][-1,0])/2)
                        rec2["instruction_by_property_interaction_m"]=(relevant-nuisance)-(ra-rb)
                    np.savez_compressed(dest/"alternate_execution.npz",**paths)
                finally:
                    session.request=old
    write(dest/"case.json",report)
    print(dest, "source",report["source_pass"],"sham",report.get("sham_pass"),flush=True)
    return report


def run(prescribed=False):
    p0 = summarize(ROOT)
    if not p0["complete"]:
        raise RuntimeError("Native calibration is incomplete")
    out=OUT.with_name("e1_prescribed_slide") if prescribed else OUT
    out.mkdir(exist_ok=True)
    native_tasks = [int(k) for k,v in p0["task_gates"].items() if v["eligible"]]
    calibration = json.loads((ROOT/"stimulus_calibration.json").read_text())
    speeds = calibration["chosen"]
    plan = json.loads((ROOT/"protocol.json").read_text())["plan"]
    tasks = [task["task_id"] for task in plan if prescribed or
             (task["task_id"] in native_tasks and speeds[task["target"]] is not None
              and speeds[DISTRACTORS[task["task_id"]]] is not None)]
    freeze(out/"protocol.json",dict(script_sha256=digest_file(__file__),native_sha256=digest_file(ROOT/"protocol.json"),
           native_eligible_tasks=native_tasks,eligible_tasks=tasks,task_gates=p0["task_gates"],geometry_calibrated_speeds_mps=speeds,source_axis="world x",
           dev=[0,1,2,3],held_out=list(range(24,34)),seeds=[0,1],primary="relevant half-contrast minus nuisance half-contrast at 32 controls",
           units="world meters",same_instruction=True,source="prescribed 3 cm quintic render translation on one shared robot trajectory" if prescribed else "native 32-action open-loop replay; exogenous initial object velocity",
           qualification="Prescribed-render development cases include failed native controls to diagnose stimulus feasibility. Only tasks passing both native and source/sham development checks can reach held-out confirmation. Task relevance varies by object."))
    if (out/"status.json").exists():
        return
    if not tasks:
        write(out/"status.json",dict(status="stopped_stimulus_calibration" if native_tasks else "stopped_native_target_gate",
                                     tasks=tasks,native_eligible_tasks=native_tasks,speeds=speeds))
        return
    cfg = json.loads(Path("configs/experiments/e1_libero.json").read_text())
    loaded = load_pilot(cfg)
    adapter=loaded.adapter
    loaded.close()
    suite = _get_suite("libero_10")
    dev_ok = []
    for stage in ("development","held_out"):
        if stage == "held_out" and not dev_ok:
            break
        for task_id in (tasks if stage=="development" else dev_ok):
            task=suite.get_task(task_id)
            target=next(t["target"] for t in json.loads((ROOT/"protocol.json").read_text())["plan"] if t["task_id"]==task_id)
            successes=[]
            for state in (range(4) if stage=="development" else range(24,34)):
                for seed in ([0] if stage=="development" else [0,1]):
                    dest=out/f"{stage}/task_{task_id}/state_{state}/seed_{seed}"
                    if (dest/"case.json").exists():
                        report=json.loads((dest/"case.json").read_text())
                    else:
                        c=dict(cfg,task_id=task_id,initial_state_index=state,action_seed=seed)
                        pilot=LiberoPilot(c,adapter,task=task,initial_state=initial_states_for_task(suite,task_id)[state],
                                    create_env=create_libero_environment,reset_env=reset_to_initial_state,step_env=step_libero)
                        try:
                            report=state_case(pilot,state,seed,target,DISTRACTORS[task_id],dest,speeds,prescribed)
                        finally:
                            pilot.close()
                    successes.append(report["source_pass"] and report.get("sham_pass",False))
            if stage=="development" and task_id in native_tasks and sum(successes)>=3:
                dev_ok.append(task_id)
    write(out/"status.json", dict(status="held_out_evaluated" if dev_ok else "stopped_development_gates",
                                 qualifying_tasks=dev_ok, cases=len(list(out.glob("*/*/*/*/case.json")))))

if __name__ == "__main__":
    run()
    run(prescribed=True)
