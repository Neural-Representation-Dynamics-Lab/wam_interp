"""Frozen native physical-relocation positive control. No future transplantation."""
import argparse
from dataclasses import replace
from functools import partial
from pathlib import Path
import json, random, zipfile
import numpy as np
from e1_replication import write,now
from e1_visual_replication import verify,verify_manifest,attach_physics,model_identity
from e1_simulator_pairs import observation_arrays
from wam_interp.experiments.physical_relocation import reset_relocated,NativeObjectCapture,paired_robot_start
from wam_interp.experiments.simulator_libero import RobotContactCapture
from wam_interp.experiments.libero_pilot import LiberoPilot,load_pilot,digest_file,digest_array
from wam_interp.experiments.object_identity import identity_reversal
from wam_interp.experiments.controlled_replication import state_bootstrap
from wam_interp.experiments.e1 import _validate_branch

read=lambda p:json.loads(Path(p).read_text(encoding='utf-8'))
def status(r,s,**kw):write(r/'status.json',{'status':s,'at':now(),**kw})
def require(r,s):
    if read(r/'status.json')['status']!=s: raise ValueError('Unexpected stage; use a fresh run, no silent restart')

def config(p,state):
    c={**read(p['pilot_config']),'suite':p['suite'],'task_id':p['task_id'],'initial_state_index':state,'action_seed':0}
    c.pop('donor_prompts',None)
    if c['warmup_steps']!=30 or c['execute_steps']!=32: raise ValueError('Frozen horizon/warm-up changed')
    return c

def resetter(p,state,variant):
    if variant not in p['variants']: raise ValueError('Unknown physical arrangement')
    return partial(reset_relocated,offsets=p['layouts'][str(state)],swap_pair=p['swap_objects'] if variant=='swapped' else ())

def prepare(r,p):
    if r.exists():raise FileExistsError('Fresh output required')
    if set(p['layouts'])!={str(s) for s in p['states']} or len(set(p['states']))!=len(p['states']):raise ValueError('Explicit unique layouts required')
    if set(p['instructions'])!={'soup','tomato','cream'} or p['variants']!=['original','swapped']:raise ValueError('Full factorial required')
    r.mkdir(parents=True);write(r/'protocol.json',p)
    with zipfile.ZipFile(r/'source.zip','x',zipfile.ZIP_DEFLATED) as z:
        for folder in ('src','scripts','configs','tests'):
            for f in sorted(Path(folder).rglob('*')):
                if f.is_file() and f.suffix in ('.py','.json','.yaml','.sh'):z.write(f,f.as_posix())
        for f in ('__init__.py','pyproject.toml','docs/e1_physical_relocation.md'):z.write(f,f)
    write(r/'freeze.json',{'at':now(),'protocol_sha256':digest_file(r/'protocol.json'),'source_sha256':digest_file(r/'source.zip')})
    status(r,'prepared')

def scene(p,state,variant,capture=False):
    from wam_interp.evaluations.LIBERO.runner import _get_suite
    from wam_interp.evaluations.LIBERO.environment import create_libero_environment,initial_states_for_task,step_libero,LIBERO_DUMMY_ACTION
    c=config(p,state);suite=_get_suite(p['suite']);task=suite.get_task(p['task_id']);env=None
    try:
        random.seed(c['environment_seed']);np.random.seed(c['environment_seed'])
        env=create_libero_environment(task,c['camera_resolution'],c['environment_seed'])
        resetter(p,state,variant)(env,initial_states_for_task(suite,p['task_id'])[state],instruction=task.language,metadata={})
        initial=env.sim.data.body_xpos[list(env.env.obj_body_id.values())].copy()
        with NativeObjectCapture(env,p['instruction_objects'].values()) as sensor,RobotContactCapture(env) as contacts:
            contacts.enabled=True
            for i in range(c['warmup_steps']):
                sensor.enabled=capture and i==c['warmup_steps']-1
                step=step_libero(env,LIBERO_DUMMY_ACTION,instruction=task.language,metadata={})
                if step.done or step.success:raise ValueError('Warm-up terminated')
            obs=step.observation
        arrays=observation_arrays(obs,env);body_to_object={v:k for k,v in env.env.obj_body_id.items()};cross=[]
        for ct in env.sim.data.contact[:env.sim.data.ncon]:
            a,b=[body_to_object.get(int(env.sim.model.geom_bodyid[int(g)])) for g in (ct.geom1,ct.geom2)]
            if a is not None and b is not None and a!=b:cross.append({'objects':[a,b],'distance_m':float(ct.dist)})
        pixels={}
        if capture:
            for key,cam in [('agentview','agentview'),('wrist','robot0_eye_in_hand')]:
                sample=sensor.latest[cam]
                if not np.array_equal(sample['rgb'],obs.cameras[key]):raise ValueError('Camera audit timestamp mismatch')
                arrays[key+'_segmentation']=sample['segmentation'];arrays[key+'_depth']=sample['depth'];pixels[key]=sample['visible_pixels']
        meta={'state':state,'variant':variant,'relocation':env._relocation_audit,'object_names':list(env.env.obj_body_id),
              'warmup_robot_contacts':contacts.events,'cross_object_contacts':cross,'visible_pixels':pixels,
              'max_warmup_xy_motion_m':float(np.linalg.norm(arrays['object_position'][:,:2]-initial[:,:2],axis=-1).max()),
              'initial_after_relocation_positions':initial.tolist()}
        return env,obs,arrays,meta
    except BaseException:
        if env is not None:env.close()
        raise

def sources(r):
    require(r,'prepared');verify(r);p=read(r/'protocol.json');status(r,'sources_running');rows=[]
    from PIL import Image,ImageDraw
    from wam_interp.models.fastwam.libero_adapter import _rotate_180,_center_crop_resize
    for state in p['states']:
        cases={};metas={};out=Image.new('RGB',(896,510),'white');draw=ImageDraw.Draw(out)
        for row,variant in enumerate(p['variants']):
            env=None
            try:
                env,obs,a,m=scene(p,state,variant,capture=True);env.close();env=None
                env,_,b,_=scene(p,state,variant)
                if any(not np.array_equal(a[k],v) for k,v in b.items()):raise ValueError('Fresh source replay differs')
                reasons=[]
                if m['warmup_robot_contacts']:reasons.append('warmup_robot_contact')
                if m['cross_object_contacts']:reasons.append('cross_object_contact')
                if m['max_warmup_xy_motion_m']>p['source_limits']['maximum_settling_xy_m']:reasons.append('excess_settling')
                if min(m['visible_pixels']['agentview'].values())<p['source_limits']['minimum_agentview_pixels_per_target']:reasons.append('target_not_visible')
                m.update(accepted=not reasons,reasons=reasons,fresh_reset_exact=True)
                dest=r/f'state_{state}/{variant}';dest.mkdir(parents=True)
                np.savez_compressed(dest/'source.npz',**a);write(dest/'source.json',m);cases[variant]=a;metas[variant]=m
                for j,cam in enumerate(('agentview','wrist')):
                    frame=_center_crop_resize(_rotate_180(a[cam]),width=224,height=224)
                    draw.text((j*448+5,row*255+5),f'Layout {state}, {variant}, {cam}',fill='black')
                    out.paste(Image.fromarray(frame).resize((448,224)),(j*448,row*255+28))
                rows.append({'state':state,'variant':variant,'accepted':m['accepted'],'reasons':reasons})
                print(json.dumps(rows[-1]),flush=True)
            finally:
                if env is not None:env.close()
        out.save(r/f'state_{state}/source.jpg',quality=96)
        a,b=cases['original'],cases['swapped'];names=metas['original']['object_names']
        indices=[names.index(n) for n in p['swap_objects']]
        robot=paired_robot_start(a,b,p['source_limits']['paired_robot_absolute_tolerance'])
        slot_error=float(np.linalg.norm(a['object_position'][indices,:2][::-1]-b['object_position'][indices,:2],axis=-1).max())
        unchanged=[i for i,n in enumerate(names) if n not in p['swap_objects']]
        distractor_error=float(np.linalg.norm(a['object_position'][unchanged]-b['object_position'][unchanged],axis=-1).max())
        pair={'state':state,'robot_proprio_ctrl_exact':robot['exact'],'robot_start_audit':robot,'max_swap_slot_error_m':slot_error,'max_unmoved_object_difference_m':distractor_error,
              'pass':robot['pass'] and slot_error<=p['source_limits']['maximum_slot_error_m'] and distractor_error<=1e-8}
        write(r/f'state_{state}/source_pair.json',pair)
        if not pair['pass']:rows.append({'state':state,'variant':'pair','accepted':False,'reasons':['paired_scene_mismatch']})
    write(r/'source_selection.json',rows)
    write(r/'source_manifest.json',{f.relative_to(r).as_posix():digest_file(f) for f in r.glob('state_*/**/*') if f.is_file()})
    write(r/'source_review.json',{'method':None,'before_policy_outcomes':False,'human_review':'pending','states':{str(s):None for s in p['states']}})
    verify(r);status(r,'sources_ready' if all(x['accepted'] for x in rows) else 'stopped_source_gate')

def make_pilot(p,state,variant,loaded=None):
    from wam_interp.evaluations.LIBERO.runner import _get_suite
    from wam_interp.evaluations.LIBERO.environment import create_libero_environment,initial_states_for_task,step_libero
    c=config(p,state)
    if loaded is None:loaded=load_pilot(c)
    loaded.close();suite=_get_suite(p['suite'])
    obj=LiberoPilot(c,loaded.adapter,task=suite.get_task(p['task_id']),initial_state=initial_states_for_task(suite,p['task_id'])[state],
        create_env=create_libero_environment,reset_env=resetter(p,state,variant),step_env=step_libero)
    obj.resolved_model_config=loaded.resolved_model_config;obj.verify_replay();attach_physics(obj);return obj

def run_case(obj,dest,context,baseline=None):
    import torch
    obj.restore(obj.saved_rng);torch.cuda.reset_peak_memory_stats();b=obj.session.run(None);_validate_branch(b)
    e={k:v.copy() for k,v in obj.last_rollout.items()}
    if b.actions.shape!=(32,7) or len(b.executed_commands)!=32:raise ValueError('Incomplete 32-control chunk')
    if baseline is not None:
        old_b,old_e=baseline
        if not np.array_equal(b.actions,old_b.actions) or not np.array_equal(b.future,old_b.future) or any(not np.array_equal(v,e[k]) for k,v in old_e.items()):
            raise ValueError('Native repeat differs')
    dest.mkdir(parents=True)
    np.savez_compressed(dest/'policy.npz',future=b.future,actions=b.actions,commands=b.commands,executed_commands=b.executed_commands)
    np.savez_compressed(dest/'execution.npz',**e)
    write(dest/'record.json',{**context,'at':now(),'metadata':b.metadata,'peak_gpu_allocated_bytes':torch.cuda.max_memory_allocated(),
        'input_image_sha256':digest_array(obj.request.observation.image),'proprio_sha256':digest_array(obj.request.observation.proprio),
        'policy_sha256':digest_file(dest/'policy.npz'),'execution_sha256':digest_file(dest/'execution.npz'),
        'repeat_exact':baseline is not None,'benchmark_success_scope':'Original soup BDDL goal only; not success for counterfactual instructions.'})
    return b,e

def run(r):
    require(r,'sources_ready');verify(r);verify_manifest(r,r/'source_manifest.json');p=read(r/'protocol.json');review=read(r/'source_review.json')
    if review['method'] not in ('AI_assisted','human') or not review['before_policy_outcomes'] or set(review['states'])!={str(s) for s in p['states']} or any(not x or not x.get('accepted') or not x.get('notes') for x in review['states'].values()):
        raise ValueError('Actually review all native source pairs before inference')
    write(r/'run_freeze.json',{'at':now(),'review_sha256':digest_file(r/'source_review.json'),'source_manifest_sha256':digest_file(r/'source_manifest.json')})
    status(r,'running');obj=None;n=0
    try:
        for state in p['states']:
            for variant in p['variants']:
                obj=make_pilot(p,state,variant,obj);native=obj.request
                with np.load(r/f'state_{state}/{variant}/source.npz',allow_pickle=False) as z:
                    current=observation_arrays(obj.observation,obj.env)
                    if any(not np.array_equal(v,z[k]) for k,v in current.items()):raise ValueError('Policy source differs from frozen scene')
                if not (r/'model_identity.json').exists():write(r/'model_identity.json',model_identity(obj))
                for language,prompt in p['instructions'].items():
                    for seed in p['action_seeds']:
                        obj.request=replace(native,observation=replace(native.observation,prompt=prompt),seed=seed);obj.session.request=obj.request;obj.config['action_seed']=seed
                        context={'state':state,'variant':variant,'language':language,'seed':seed,'instruction':prompt,'intervention':'coherent_physical_xy_swap' if variant=='swapped' else 'native_layout'}
                        dest=r/f'state_{state}/{variant}/{language}/seed_{seed}'
                        baseline=run_case(obj,dest/'native',context);n+=1
                        if variant=='original' and seed==0:run_case(obj,dest/'repeat',context,baseline);n+=1
                        write(r/'progress.json',{'branches':n,**context});print(json.dumps({'branches':n,**context}),flush=True)
        verify(r);verify_manifest(r,r/'source_manifest.json');status(r,'policy_complete',branches=n)
    finally:
        if obj is not None:obj.close()

def analyze(r):
    require(r,'policy_complete');verify(r);p=read(r/'protocol.json');rows=[];runtime=[]
    for state in p['states']:
        meta=read(r/f'state_{state}/original/source.json');names=meta['object_names']
        with np.load(r/f'state_{state}/original/source.npz',allow_pickle=False) as z:
            axis=z['object_position'][names.index(p['swap_objects'][1])]-z['object_position'][names.index(p['swap_objects'][0])];axis[2]=0
        for seed in p['action_seeds']:
            endpoints={};distances=[];safe=True
            for variant in p['variants']:
                for language in p['instructions']:
                    dest=r/f'state_{state}/{variant}/{language}/seed_{seed}/native';record=read(dest/'record.json')
                    if digest_file(dest/'execution.npz')!=record['execution_sha256'] or digest_file(dest/'policy.npz')!=record['policy_sha256']:raise ValueError('Outcome drift')
                    with np.load(dest/'execution.npz',allow_pickle=False) as z:
                        endpoint=z['eef_position'][-1].copy();endpoints[language,variant]=endpoint
                        movement=float(np.linalg.norm(z['object_position']-z['object_position'][:1],axis=-1).max())
                        target=names.index(p['instruction_objects'][language]);target_distance=float(np.linalg.norm(endpoint[:2]-z['object_position'][0,target,:2]))
                        distances.append({'language':language,'variant':variant,'endpoint_target_xy_distance_m':target_distance,
                            'initial_target_xy_distance_m':float(np.linalg.norm(z['eef_position'][0,:2]-z['object_position'][0,target,:2]))})
                    contacts=len(record['metadata']['rollout']['contact_events']);safe &= contacts==0 and movement<=p['maximum_object_motion_m']
                    runtime.append({'state':state,'seed':seed,'variant':variant,'language':language,'contacts':contacts,'object_motion_m':movement,
                        'inference_seconds':record['metadata']['inference_with_capture_seconds'],'peak_gpu_allocated_bytes':record['peak_gpu_allocated_bytes']})
            row=identity_reversal(endpoints,axis,**p['identity_limits']);row.update(task=p['task_id'],state=state,seed=seed,
                contact_and_object_gate=safe,target_distances=distances,relocation_interaction_m=(row['soup_swap_shift_m']-row['tomato_swap_shift_m'])/2)
            row['pass'] &= safe;rows.append(row)
    passed=[s for s in p['states'] if all(x['pass'] for x in rows if x['state']==s)]
    result={'at':now(),'rows':rows,'runtime':runtime,'passing_layouts':passed,'all_layouts':p['states'],'native_branches':len(runtime),
        'repeat_branches':len(p['states'])*len(p['instructions']),'pass':len(passed)>=p['minimum_passing_layouts'],
        'interaction':state_bootstrap(rows,'relocation_interaction_m'),'human_review':'pending',
        'interpretation':'Native current-scene target following only. Does not isolate future-video semantics or establish task success. Four designed arrangements on one table; seeds are paired repeats.'}
    write(r/'analysis.json',result);status(r,'analysis_complete',branches=read(r/'status.json')['branches'],passing_layouts=passed,validated=result['pass'])
    print(json.dumps({'passing_layouts':passed,'interaction':result['interaction']}),flush=True)

def replay(r):
    require(r,'analysis_complete');verify(r);p=read(r/'protocol.json');rows=[]
    from wam_interp.evaluations.LIBERO.environment import step_libero
    for state in p['states']:
        for variant in p['variants']:
            for language in p['instructions']:
                dest=r/f'state_{state}/{variant}/{language}/seed_0/native';env=None
                try:
                    with np.load(dest/'policy.npz',allow_pickle=False) as z:commands=z['executed_commands'].copy()
                    env,obs,a,_=scene(p,state,variant);records=[a]
                    with RobotContactCapture(env) as contacts:
                        contacts.enabled=True
                        for command in commands:
                            step=step_libero(env,command,instruction=p['instructions'][language],metadata={});records.append(observation_arrays(step.observation,env))
                    arrays={k:np.stack([x[k] for x in records]) for k in records[0]};arrays['control_steps']=np.arange(len(records))
                    with np.load(dest/'execution.npz',allow_pickle=False) as z:
                        if any(not np.array_equal(z[k],arrays[k]) for k in z.files):raise ValueError('Independent replay differs')
                    old=read(dest/'record.json')['metadata']['rollout']
                    if old['contact_events']!=contacts.events or old['integrator_substeps_checked']!=contacts.substeps:raise ValueError('Replay contact mismatch')
                    row={'state':state,'variant':variant,'language':language,'seed':0,'all_arrays_and_contacts_exact':True};rows.append(row)
                    write(dest/'independent_replay.json',row);write(r/'replay_progress.json',{'branches':len(rows),'expected':24})
                finally:
                    if env is not None:env.close()
    write(r/'replay_audit.json',{'at':now(),'rows':rows,'all_exact':True,'branches':len(rows),'selection':'Every layout, physical arrangement and instruction, seed zero; fixed before outcomes.'})
    status(r,'complete',branches=60,validated=read(r/'analysis.json')['pass'])

if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('stage',choices=['prepare','sources','run','analyze','replay']);ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--protocol',type=Path,default=Path('configs/experiments/e1_physical_relocation.json'));args=ap.parse_args()
    try:
        if args.stage=='prepare':prepare(args.output,read(args.protocol))
        else:globals()[args.stage](args.output)
    except BaseException as exc:
        if args.stage!='prepare':write(args.output/'failure.json',{'at':now(),'stage':args.stage,'error':repr(exc)});status(args.output,'failed',stage=args.stage,error=repr(exc))
        raise
