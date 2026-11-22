"""Score visible local motion at each native replan in the external camera.

The simulator projects 3D positions. Local optical flow estimates motion near
those points in the decoded forecast. This yields exploratory pixel metrics.
"""
import argparse
import json
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
from e1_native_replan import ROOT, write
from wam_interp.evaluations.LIBERO.environment import create_libero_environment, initial_states_for_task
from wam_interp.evaluations.LIBERO.runner import _get_suite
from wam_interp.experiments.physical_relocation import object_geometry_ids
from wam_interp.experiments.libero_pilot import digest_file


def external_camera(suite, task_id, target):
    from robosuite.utils.camera_utils import get_camera_transform_matrix
    from robosuite import macros
    import mujoco
    if macros.IMAGE_CONVENTION != "opengl":
        raise ValueError("Expected the benchmark OpenGL camera convention")
    task=suite.get_task(task_id)
    env=create_libero_environment(task,256,0)
    try:
        env.reset()
        raw=env.set_init_state(initial_states_for_task(suite,task_id)[0])
        matrix=get_camera_transform_matrix(env.sim,"agentview",256,256)
        native=env.sim.render(camera_name="agentview", height=256,width=256,segmentation=True)
        # Saved model-facing image is rotated 180 degrees from raw MuJoCo.
        seg=native[::-1,::-1]
        body=env.env.obj_body_id[target]
        mask=(seg[...,0]==int(mujoco.mjtObj.mjOBJ_GEOM)) & np.isin(seg[...,1],object_geometry_ids(env.sim.model,body))
        y,x=np.where(mask)
        center=project(matrix,env.sim.data.body_xpos[body])
        residual=np.sqrt((x*224/256-center[0])**2+(y*224/256-center[1])**2)
        if not len(x) or residual.min()>4:
            raise ValueError(f"Projection/segmentation mismatch for {target}")
        return matrix,dict(mask_pixels=len(x),distance_to_target_mask_px=float(residual.min()),
                            eef_anchor_projection=project(matrix,raw["robot0_eef_pos"]).tolist())
    finally:
        env.close()


def project(matrix,xyz):
    homogeneous=matrix@np.r_[xyz,1.]
    if homogeneous[2]<=0:
        raise ValueError("Point behind external camera")
    x,y=homogeneous[:2]/homogeneous[2]
    return (np.array([255-x,y])+.5)*(224/256)-.5


def lk(a,b,pts):
    opts=dict(winSize=(15,15),maxLevel=2,
              criteria=(cv2.TERM_CRITERIA_EPS|cv2.TERM_CRITERIA_COUNT,30,.01))
    q,good,_=cv2.calcOpticalFlowPyrLK(a,b,pts,None,**opts)
    back,reverse,_=cv2.calcOpticalFlowPyrLK(b,a,q,None,**opts)
    ok=good.ravel().astype(bool)&reverse.ravel().astype(bool)&(np.linalg.norm(back-pts,axis=2).ravel()<=1)
    ok &= ((q>=0)&(q<224)).all(axis=(1,2))
    return q,ok


def measure(actual,predicted,geometry,prior):
    mask=np.zeros(actual[0].shape,np.uint8)
    cv2.circle(mask,tuple(np.rint(geometry[0]).astype(int)),9,255,-1)
    corners=cv2.goodFeaturesToTrack(actual[0],maxCorners=18,qualityLevel=.03,minDistance=2,mask=mask)
    if corners is None or len(corners)<2:
        return [None]*(len(actual)-1)
    pred0,anchor_ok=lk(actual[0],predicted[0],corners)
    anchor_ok &= np.linalg.norm(pred0-corners,axis=2).ravel()<=3
    output=[]
    for i in range(1,len(actual)):
        truth=geometry[i]-geometry[0]
        real,ok1=lk(actual[0],actual[i],corners)
        pred,ok2=lk(predicted[0],predicted[i],pred0)
        real_delta=(real-corners)[:,0]
        # Reject features that demonstrably belong to the stationary background or an occluder.
        valid=anchor_ok & ok1 & ok2 & (np.linalg.norm(real_delta-truth,axis=1)<=3)
        if valid.sum()<2:
            output.append(None)
            continue
        delta=np.median((pred-pred0)[valid,0],axis=0)
        output.append(dict(tracked_features=int(valid.sum()),forecast_delta_px=delta.tolist(),executed_delta_px=truth.tolist(),
                           error_px=float(np.linalg.norm(delta-truth)),stationary_error_px=float(np.linalg.norm(truth)),
                           constant_velocity_error_px=None if prior is None else float(np.linalg.norm(truth-i*prior))))
    return output


def score(vae, folder,matrix):
    import torch
    meta=json.loads((folder/"episode.json").read_text())
    tracks=json.loads((folder/"tracks.json").read_text())
    target=meta["target"]
    pil=lambda path: np.asarray(Image.open(path).convert("RGB"))[:,:224].copy()
    gray=lambda img: cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
    rows=[]
    for j in range(meta["episode"]["inference_calls"]):
        start=j*10
        offsets=[x for x in (0,4,8) if start+x<len(tracks)]
        row={"decision":j,"start_control":start,"eligible_offsets":offsets}
        if len(offsets)>1:
            z=torch.from_numpy(np.load(folder/f"future_{j:03d}.npy")).to("cuda",torch.bfloat16)
            with torch.inference_mode():
                rgb=vae.decode(z,device="cuda",tiled=False).squeeze(0).float().clamp(-1,1)
                rgb=((rgb+1)*127.5).byte().permute(1,2,3,0).cpu().numpy()
            if rgb.shape!=(9,224,448,3):
                raise ValueError(f"Unexpected decoded layout {rgb.shape}")
            pred=[gray(rgb[i,:,:224]) for i in range(len(offsets))]
            actual=[gray(pil(folder/f"step_{start+x:03d}.png")) for x in offsets]
            for name in ("eef",target):
                def point(t):
                    v=tracks[t]["eef"] if name=="eef" else tracks[t]["objects"][target]
                    return project(matrix,v)
                geometry=[point(start+x) for x in offsets]
                velocity=geometry[0]-point(start-4) if start>=4 else None
                row[name]=measure(actual,pred,geometry,velocity)
            if j in (0,3,8):
                tile=np.vstack([np.hstack([cv2.cvtColor(im,cv2.COLOR_GRAY2RGB) for im in seq]) for seq in (actual,pred)])
                Image.fromarray(tile).save(folder/f"paired_forecast_{j:03d}.jpg")
        rows.append(row)
    write(folder/"alignment.json",dict(target=target,rows=rows,units="external-camera pixels of displacement",
          annotation="Exploratory local feature motion near projected landmark; independent human review pending",
          sampling="Offsets 4 and 8 of the actual <=10 controls. Baseline uses last 4 controls."))
    print("scored", folder, len(rows),flush=True)


def main():
    import time
    import torch
    from wam_interp.models.fastwam.model_code.helpers.loader import _load_registered_model
    torch.cuda.set_per_process_memory_fraction(.20)
    path="checkpoints/Wan-AI/Wan2.2-TI2V-5B/Wan2.2_VAE.pth"
    vae=_load_registered_model(path,"wan_video_vae",torch_dtype=torch.bfloat16,device="cuda").eval()
    suite=_get_suite("libero_10")
    cameras={}
    for t in json.loads((ROOT/"protocol.json").read_text())["plan"]:
        matrix,audit=external_camera(suite,t["task_id"],t["target"])
        cameras[t["task_id"]]=matrix
        write(ROOT/f"projection_{t['task_id']}.json",dict(matrix=matrix.tolist(),**audit))
    while True:
        completed=list(ROOT.glob("task_*/state_*/*/idm/episode.json"))
        for p in sorted(completed):
            if not (p.parent/"alignment.json").exists():
                score(vae,p.parent,cameras[json.loads(p.read_text())["task_id"]])
        if len(completed)==16 or not args.follow:
            break
        time.sleep(15)

if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--follow",action="store_true")
    args=parser.parse_args()
    main()
