"""Build fixed, physically rendered approach/appearance pairs without a WAM."""
import argparse
from datetime import datetime,timezone
import hashlib,json,random,subprocess,zipfile
from pathlib import Path
import numpy as np
from wam_interp.experiments.simulator_video import approach_command,appearance_metrics,approach_metrics,valid_approach,paired_camera_clip
from wam_interp.experiments.simulator_libero import AppearanceCapture,RobotContactCapture


def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2,allow_nan=False),encoding='utf-8')


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def sheet(clip,path):
    from PIL import Image,ImageDraw
    out=Image.new('RGB',(1344,750),'white');draw=ImageDraw.Draw(out)
    for i,frame in enumerate(clip):
        x,y=(i%3)*448,(i//3)*250
        draw.text((x+3,y+3),f'{path.parent.name} {path.stem} frame {i}',fill='black');out.paste(Image.fromarray(frame),(x,y+25))
    out.save(path,quality=97)


def observation_arrays(obs,env):
    d=env.sim.data;ids=list(env.env.obj_body_id.values())
    return {**{k:v.copy() for k,v in obs.cameras.items()},**{k:v.copy() for k,v in obs.state.items()},
        'qpos':d.qpos.copy(),'qvel':d.qvel.copy(),'ctrl':d.ctrl.copy(),
        'object_position':d.body_xpos[ids].copy(),'object_quaternion':d.body_xquat[ids].copy()}


def run_reference(config,state,label,*,commands=None,capture=True,capture_factory=None,reset_fn=None):
    from wam_interp.evaluations.LIBERO.runner import _get_suite
    from wam_interp.evaluations.LIBERO.environment import create_libero_environment,initial_states_for_task,reset_to_initial_state,step_libero,LIBERO_DUMMY_ACTION
    import mujoco
    suite=_get_suite(config['suite']);task=suite.get_task(config['task_id']);env=None
    frames=[];records=[];actions=[];validation=[]
    try:
        random.seed(config['environment_seed']);np.random.seed(config['environment_seed'])
        env=create_libero_environment(task,config['camera_resolution'],config['environment_seed'])
        obs=(reset_to_initial_state if reset_fn is None else reset_fn)(env,initial_states_for_task(suite,config['task_id'])[state],instruction=task.language,metadata={})
        appearance=(AppearanceCapture(env,material_name=config['appearance']['material'],factors=config['appearance']['factors'],coverage_scale=config.get('coverage_scale',4))
                    if capture_factory is None else capture_factory(env))
        with appearance,RobotContactCapture(env) as contacts:
            for i in range(config['warmup_steps']):
                appearance.enabled=capture and i==config['warmup_steps']-1
                result=step_libero(env,LIBERO_DUMMY_ACTION,instruction=task.language,metadata={})
                if result.done or result.success:raise RuntimeError('Warmup terminated')
                obs=result.observation
            initial=observation_arrays(obs,env);records.append(initial)
            names=list(env.env.obj_body_id);target_index=names.index(config['targets'][label]);other_label=next(n for n in config['targets'] if n!=label)
            other_index=names.index(config['targets'][other_label]);goal=initial['object_position'][target_index].copy();goal[2]+=config['hover_above_object_center_m']
            def collect(index):
                row={}
                for key,cam in [('agentview','agentview'),('wrist','robot0_eye_in_hand')]:
                    sample=appearance.latest[cam]
                    if not np.array_equal(sample['rgb'],obs.cameras[key]):raise RuntimeError(f'Capture differs from native cached camera at {index}/{key}')
                    row[key]=sample
                    for variant in config['appearance']['factors']:
                        metrics=appearance_metrics(sample['rgb'],sample['variants'][variant],sample['mask'],sample['depth'],sample['variant_depths'][variant],boundary_pixels=config['boundary_pixels'])
                        metrics.update(sample.get('variant_diagnostics',{}).get(variant,{}))
                        validation.append({'control_step':index,'camera':key,'variant':variant,**metrics})
                frames.append(row)
            if capture:collect(0)
            contacts.enabled=True
            for step in range(1,33):
                appearance.enabled=capture and step in config['frame_steps']
                command=approach_command(initial['eef_position'],obs.state['eef_position'],goal,step,**config['controller']) if commands is None else commands[step-1].copy()
                result=step_libero(env,command,instruction=task.language,metadata={});obs=result.observation
                if result.done or result.success:raise RuntimeError('Reference terminated before fixed horizon')
                actions.append(command);records.append(observation_arrays(obs,env))
                if appearance.enabled:collect(step)
            arrays={k:np.stack([r[k] for r in records]) for k in records[0]}
            arrays['commands']=np.asarray(actions);arrays['control_steps']=np.arange(33)
            metrics=approach_metrics(arrays['eef_position'],arrays['object_position'],target_index,other_index,goal,len(contacts.events))
            meta={'state':state,'label':label,'instruction':task.language,'object_names':names,'target_index':target_index,'other_index':other_index,'goal':goal.tolist(),'approach':metrics,'appearance_validation':validation,'contact_events':contacts.events,'integrator_substeps_checked':contacts.substeps,'render_geometry_ids':appearance.geom_ids.tolist(),'material_id':int(appearance.material_id),'camera_convention':int(appearance.convention),'camera_sampling':'Captured inside the native sensor call; matched against the observation returned by env.step, not re-rendered after step.','reference_kind':'scripted region approach, no learned donor prompt','versions':{'mujoco':mujoco.__version__}}
            if capture:
                for key in ('agentview','wrist'):
                    for field in ('rgb','depth','mask','segmentation','sensor_time'):
                        arrays[f'video_{key}_{field}']=np.stack([f[key][field] for f in frames])
                    for field in ('robot_mask','gripper_mask','robot_support_mask','gripper_support_mask'):
                        if field in frames[0][key]:arrays[f'video_{key}_{field}']=np.stack([f[key][field] for f in frames])
                    for field in ('body_xpos','body_xquat','cam_xpos','cam_xmat','qpos','qvel'):
                        arrays[f'sensor_{key}_{field}']=np.stack([f[key]['sensor_state'][field] for f in frames])
                    for variant in config['appearance']['factors']:
                        arrays[f'video_{key}_{variant}']=np.stack([f[key]['variants'][variant] for f in frames])
                        if config.get('save_variant_depths',False):
                            arrays[f'video_{key}_{variant}_depth']=np.stack([f[key]['variant_depths'][variant] for f in frames])
                arrays['clip_original']=paired_camera_clip(arrays['video_agentview_rgb'],arrays['video_wrist_rgb'])
                for variant in config['appearance']['factors']:
                    arrays[f'clip_{variant}']=paired_camera_clip(arrays[f'video_agentview_{variant}'],arrays[f'video_wrist_{variant}'])
            return arrays,meta
    finally:
        if env is not None:env.close()


def build(root,config):
    if root.exists():raise FileExistsError('Use a new result folder; no silent restart')
    if config['control_steps']!=32 or config['frame_steps']!=list(range(0,33,4)) or len(config['targets'])!=2:raise ValueError('Initial builder supports two targets and nine frames over 32 controls')
    root.mkdir(parents=True);write(root/'protocol.json',config)
    with zipfile.ZipFile(root/'source.zip','w',zipfile.ZIP_DEFLATED) as archive:
        for folder in ('src','scripts','configs'):
            for path in sorted(Path(folder).rglob('*')):
                if path.is_file() and path.suffix in ('.py','.json','.yaml'):archive.write(path,path.as_posix())
        for name in ('__init__.py','pyproject.toml'):archive.write(name,name)
    write(root/'freeze.json',{'at':datetime.now(timezone.utc).isoformat(),'source_sha256':sha(root/'source.zip'),'protocol_sha256':sha(root/'protocol.json'),'git_revision':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),'libero_revision':subprocess.check_output(['git','-C','third_party/LIBERO','rev-parse','HEAD'],text=True).strip()})
    rows=[]
    try:
        write(root/'status.json',{'status':'building','policy_branches':0})
        for state in config['initial_states']:
            initial=None
            for label in config['targets']:
                arrays,meta=run_reference(config,state,label)
                repeat,_=run_reference(config,state,label,commands=arrays['commands'],capture=False)
                if any(not np.array_equal(arrays[k],repeat[k]) for k in repeat):raise RuntimeError('Fresh unedited replay differs from rendered reference')
                with np.load(Path(config['previous_results'])/f'state_{state}/seed_0/baseline/execution.npz',allow_pickle=False) as old:
                    for k in ('agentview','wrist','eef_position','eef_quaternion'):
                        if not np.array_equal(arrays[k][0],old[k][0]):raise RuntimeError('Initial observation differs from historical native baseline')
                if initial is not None and any(not np.array_equal(initial[k],arrays[k][0]) for k in initial):raise RuntimeError('Target approaches start from different states')
                initial={k:arrays[k][0].copy() for k in ('agentview','wrist','eef_position','qpos','qvel','object_position')}
                reasons=[]
                if not valid_approach(meta['approach'],config['approach_limits']):reasons.append('approach_geometry_or_contacts')
                for v in meta['appearance_validation']:
                    if not v['depth_exact'] or v['changed_outside_boundary_support']!=0:reasons.append('appearance_leakage_or_depth')
                    if v['changed_pixels']/(config['camera_resolution']**2)<config['minimum_changed_fraction_per_camera']:reasons.append('appearance_not_visible')
                meta.update(accepted=not reasons,rejection_reasons=sorted(set(reasons)),fresh_unedited_replay_exact=True,historical_initial_observation_exact=True)
                folder=root/f'state_{state}'/label;folder.mkdir(parents=True)
                np.savez_compressed(folder/'reference.npz',**arrays);meta['reference_sha256']=sha(folder/'reference.npz')
                for name in ['original',*config['appearance']['factors']]:sheet(arrays[f'clip_{name}'],folder/f'{name}_sheet.jpg')
                write(folder/'metadata.json',meta);rows.append({'state':state,'label':label,'accepted':meta['accepted'],'rejection_reasons':meta['rejection_reasons'],'approach':meta['approach']})
                write(root/'selection.json',rows);print(json.dumps(rows[-1]),flush=True)
        write(root/'status.json',{'status':'awaiting_visual_codec_review' if all(x['accepted'] for x in rows) else 'stopped_reference_gate','reference_trajectories':len(rows),'policy_branches':0,'human_review':'pending','at':datetime.now(timezone.utc).isoformat()})
    except Exception as exc:
        write(root/'status.json',{'status':'failed_build','error':str(exc),'policy_branches':0});raise


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--config',type=Path,default=Path('configs/experiments/e1_simulator_pairs.json'));ap.add_argument('--output',type=Path,required=True)
    args=ap.parse_args();build(args.output,json.loads(args.config.read_text()))
