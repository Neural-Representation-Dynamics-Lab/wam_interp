"""Calibrate simulated sliding stimuli using only object/gripper geometry on development resets."""
from pathlib import Path
import json
import numpy as np
from e1_native_replan import ROOT, write, reset_seed
from wam_interp.evaluations._libero_rollout import LIBERO_DUMMY_ACTION
from wam_interp.evaluations.LIBERO.environment import (
    create_libero_environment, initial_states_for_task, reset_to_initial_state, step_libero)
from wam_interp.evaluations.LIBERO.runner import _get_suite

def main():
    suite=_get_suite("libero_10")
    rows=[]
    objects={3:["akita_black_bowl_1","wine_bottle_1"],5:["black_book_1","white_yellow_mug_1"]}
    for task_id,names in objects.items():
        task=suite.get_task(task_id)
        state=initial_states_for_task(suite,task_id)[0]
        reset_seed()
        env=create_libero_environment(task,256,0)
        try:
            kwargs=dict(instruction=task.language,metadata={})
            for obj in names:
                for speed in (.25,.4,.55,.65,.8,1.0):
                    for sign in (1,-1):
                        reset_to_initial_state(env,state,**kwargs)
                        for _ in range(30):
                            step_libero(env,LIBERO_DUMMY_ACTION,**kwargs)
                        sim=env.sim
                        body=env.env.obj_body_id[obj]
                        jt=np.flatnonzero(sim.model.jnt_bodyid==body)[0]
                        xyz=sim.data.body_xpos[body].copy()
                        quat=sim.data.body_xquat[body].copy()
                        sim.data.qvel[sim.model.jnt_dofadr[jt]]+=speed*sign
                        sim.forward()
                        rot=[]; height=[]
                        for _ in range(32):
                            step_libero(env,LIBERO_DUMMY_ACTION,**kwargs)
                            rot.append(float(np.rad2deg(2*np.arccos(np.clip(abs(sim.data.body_xquat[body]@quat),0,1)))))
                            height.append(abs(float(sim.data.body_xpos[body,2]-xyz[2])))
                        delta=(sim.data.body_xpos[body]-xyz)*sign
                        rows.append(dict(task=task_id,obj=obj,speed=speed,sign=sign,dx=float(delta[0]),
                                         max_rot=max(rot),max_height=max(height)))
                print(obj,flush=True)
        finally:
            env.close()
    chosen={}
    for names in objects.values():
        for name in names:
            options=[]
            for v in (.25,.4,.55,.65,.8,1.0):
                both=[r for r in rows if r["obj"]==name and r["speed"]==v]
                if all(.012<=r["dx"]<=.065 and r["max_height"]<=.003 and r["max_rot"]<=8 for r in both):
                    options.append((max(abs(r["dx"]-.03) for r in both),v))
            chosen[name]=min(options)[1] if options else None
    write(ROOT/"stimulus_calibration.json",dict(rows=rows,chosen=chosen,
          criterion="One speed per object, minimize maximum signed 3 cm error across both directions, subject to preregistered translation/rotation/height bounds. No policy outputs used."))
    print(chosen,flush=True)

if __name__=="__main__":
    main()
