"""First-decision native feature transplants from physically relocated scenes."""
import json
from pathlib import Path

import numpy as np
from wam_interp.evaluations._libero_rollout import _request_for_call, LIBERO_DUMMY_ACTION
from wam_interp.evaluations.LIBERO.config import LiberoEvalConfig
from wam_interp.evaluations.LIBERO.environment import (
    create_libero_environment, initial_states_for_task, reset_to_initial_state, step_libero,
)
from wam_interp.evaluations.LIBERO.runner import _get_suite
from wam_interp.experiments.physical_relocation import reset_relocated


def matched_layout(suite, task, state, targets, config, delta):
    env = create_libero_environment(suite.get_task(task),config.camera_resolution,0)
    try:
        states = initial_states_for_task(suite,task)
        env.reset()
        points=[]
        for start in states[:24]:
            env.set_init_state(start)
            points.append(env.sim.data.body_xpos[env.env.obj_body_id[targets[1]],:2].copy())
        vectors=[p-points[state] for p in points]
        ix=min(range(4,len(points)),key=lambda i: np.linalg.norm(vectors[i]-delta))
        return vectors[ix],ix
    finally:
        env.close()


def collect(audit, suite, task_id, state_id, config, offsets, *, donors=None):
    import random
    import torch
    random.seed(0)
    np.random.seed(0)
    adapter = audit.adapter
    task = suite.get_task(task_id)
    env = create_libero_environment(task,config.camera_resolution,0)
    try:
        start=initial_states_for_task(suite,task_id)[state_id]
        kwargs=dict(instruction=task.language, metadata={"task_id":task_id, "initial_state_index":state_id})
        obs=reset_relocated(env,start,offsets=offsets,**kwargs) if offsets else reset_to_initial_state(env,start,**kwargs)
        for _ in range(config.num_steps_wait):
            result=step_libero(env,LIBERO_DUMMY_ACTION,**kwargs)
            if result.done or result.success:
                raise ValueError("Terminated during paired reset")
            obs=result.observation
        anchor={k:x.copy() for k,x in obs.cameras.items()}
        proprio={k:np.asarray(x).copy() for k,x in obs.state.items()}
        if callable(getattr(adapter,"begin_episode",None)):
            adapter.begin_episode(obs)
        request=_request_for_call(config,adapter,adapter.prepare_observation(obs),0,0)
        prediction, features = audit.futures.capture(request, donor=donors)
        commands=adapter.postprocess_actions(prediction.actions,environment="libero")
        commands=torch.as_tensor(commands).detach().float().cpu().numpy().reshape(-1,7)
        if not np.isfinite(commands).all():
            raise ValueError("Nonfinite command")
        eef=[obs.state["eef_position"].tolist()]
        for action in commands[:min(config.replan_steps,10)]:
            step=step_libero(env,action,**kwargs)
            eef.append(step.observation.state["eef_position"].tolist())
            if step.done or step.success:
                break
        return dict(anchor=anchor,proprio=proprio,features=features,actions=commands,eef=np.asarray(eef))
    finally:
        env.close()


def run_scene_probe(audit, protocol, config, out):
    import torch
    out=Path(out)
    out.mkdir(parents=True,exist_ok=True)
    suite=_get_suite("libero_10")
    p=json.loads(Path(protocol).read_text())
    rows=[]
    controls={3:"wine_bottle_1",5:"white_yellow_mug_1"}
    for task in p["plan"]:
        task_id=task["task_id"]
        objects=[task["target"],controls[task_id]]
        for layout in task["layouts"]:
            state=layout["state"]
            delta=np.asarray(layout["xy_delta"])
            nuisance,donor=matched_layout(suite,task_id,state,objects,config,delta)
            discrepancy=float(np.linalg.norm(delta-nuisance))
            rec=dict(task=task_id,state=state,condition=config.inference_options["condition"],layer=audit.layer,
                     stage=0,target=objects[0],irrelevant=objects[1],target_shift_m=delta.tolist(),
                     nuisance_shift_m=nuisance.tolist(),nuisance_official_state=donor,
                     movement_mismatch_m=discrepancy)
            rows.append(rec)
            if discrepancy>.015 or np.max(np.abs(nuisance))>.05:
                rec["status"]="unmatched_physical_moves"
            else:
                base=collect(audit,suite,task_id,state,config,{})
                outputs={"native_eef":base["eef"]}
                image_mae=[]
                labels=["target","irrelevant"]
                for key,obj,move in zip(labels,objects,(delta,nuisance)):
                    shifted=collect(audit,suite,task_id,state,config,{obj:move.tolist()})
                    patched=collect(audit,suite,task_id,state,config,{},donors=shifted["features"])
                    if not all(np.array_equal(base["anchor"][k],a) for k,a in patched["anchor"].items()):
                        raise RuntimeError("Recipient observation differs")
                    if any(not np.allclose(base["proprio"][k],v,atol=1e-9,rtol=0) for k,v in shifted["proprio"].items()):
                        raise RuntimeError("Donor robot start differs")
                    current_mae=float(np.abs(base["anchor"]["agentview"].astype(float)-shifted["anchor"]["agentview"]).mean())
                    image_mae.append(current_mae)
                    a,b=base["features"].parts,shifted["features"].parts
                    d_feature=float(np.sqrt(np.mean([float((v.float()-w.float()).square().mean()) for v,w in zip(a,b)])))
                    for name,case in (("native_shifted",shifted),("future_only",patched)):
                        if case["eef"].shape!=base["eef"].shape:
                            raise RuntimeError("Unmatched executed horizon")
                        outputs[f"{key}_{name}_eef"]=case["eef"]
                        movement=case["eef"][-1]-base["eef"][-1]
                        rec[f"{key}_{name}_eef_change_m"]=movement.tolist()
                        rec[f"{key}_{name}_projected_m"]=float(movement[:2]@delta/np.linalg.norm(delta))
                    rec[f"{key}_donor_feature_rms"]=d_feature
                rec["agentview_pixel_mae"]=image_mae
                rec["status"]="sampled" if min(image_mae)>0 and max(image_mae)/min(image_mae)<=3 else "unmatched_visual_magnitude"
                rec["selective_difference_m"]=rec["target_future_only_projected_m"]-rec["irrelevant_future_only_projected_m"]
                np.savez_compressed(out/f"task_{task_id}_state_{state}.npz",**outputs)
            tmp=out/"results.tmp"
            tmp.write_text(json.dumps(dict(rows=rows,backend=audit.backend,units="world meters",
                 scope="Unchanged recipient, same prompt and seed, first ten controls. Native donor future slots may encode the donor current scene. Comparative scene-feature probe, without fixed depicted motion."),indent=2,allow_nan=False))
            tmp.replace(out/"results.json")
            print("Scene features",audit.backend,task_id,state,rec["status"],flush=True)
