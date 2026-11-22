"""Frozen multi-task simulator-video replication, with outcome-free calibration."""
import argparse
from dataclasses import replace
from pathlib import Path
import json,zipfile
import numpy as np
from e1_replication import new_pilot,write,now,load_future
from e1_simulator_pairs import run_reference,observation_arrays,sheet
from e1_video_bottleneck import execute,images
from wam_interp.experiments.controlled_replication import validate_protocol,choose_appearance,rms_difference
from wam_interp.experiments.simulator_video import encode_reference_video,valid_approach
from wam_interp.experiments.simulator_libero import RobotContactCapture
from wam_interp.experiments.libero_pilot import provenance,digest_file,digest_array
from wam_interp.experiments.fastwam_session import _array
from wam_interp.experiments.video_bottleneck import roundtrip_future,trajectory_difference,baseline_gate


def read(path):return json.loads(Path(path).read_text())
def case_id(task,state):return f'task_{task}/state_{state}'
def status(root,value,**extra):write(root/'status.json',{'status':value,'at':now(),**extra})
def require(root,value):
    if read(root/'status.json')['status']!=value:raise ValueError(f'Expected stage {value}; no silent restart')


def prepare(root,protocol):
    validate_protocol(protocol)
    if root.exists():raise FileExistsError('Fresh result directory required')
    root.mkdir(parents=True);write(root/'protocol.json',protocol)
    with zipfile.ZipFile(root/'source.zip','w',zipfile.ZIP_DEFLATED) as z:
        for folder in ('src','scripts','configs'):
            for p in sorted(Path(folder).rglob('*')):
                if p.is_file() and p.suffix in ('.py','.json','.yaml'):z.write(p,p.as_posix())
        for name in ('__init__.py','pyproject.toml'):z.write(name,name)
    write(root/'freeze.json',{'at':now(),'protocol_sha256':digest_file(root/'protocol.json'),'source_sha256':digest_file(root/'source.zip')})
    status(root,'prepared')


def verify(root):
    frozen=read(root/'freeze.json')
    for name in ('protocol','source'):
        path=root/('protocol.json' if name=='protocol' else 'source.zip')
        if digest_file(path)!=frozen[f'{name}_sha256']:raise ValueError(f'Frozen {name} changed')
    with zipfile.ZipFile(root/'source.zip') as z:
        for name in z.namelist():
            if Path(name).read_bytes()!=z.read(name):raise ValueError(f'Inference source changed: {name}')


def task_for(protocol,task_id):return next(t for t in protocol['tasks'] if t['task_id']==task_id)
def reference_config(protocol,task,partition,root):
    config=read(protocol['reference_config'])
    factors=protocol['appearance_candidates']
    if partition=='test':
        selected=read(root/'selection.json')['selected'];factors={selected:factors[selected]}
    return {**config,'suite':protocol['suite'],'task_id':task['task_id'],'targets':task['targets'],
            'appearance':{**config['appearance'],'factors':factors},'save_variant_depths':True}


def references(root,partition):
    verify(root);protocol=read(root/'protocol.json')
    require(root,'prepared' if partition=='calibration' else 'calibrated')
    rows=[];status(root,f'{partition}_references_running')
    for task in protocol['tasks']:
        config=reference_config(protocol,task,partition,root)
        for state in protocol[f'{partition}_states']:
            folder=root/partition/case_id(task['task_id'],state);folder.mkdir(parents=True)
            write(folder/'reference_config.json',config);pair=[];initial=None
            for target in task['targets']:
                dest=folder/'references'/target;dest.mkdir(parents=True)
                try:
                    arrays,meta=run_reference(config,state,target)
                    repeat,_=run_reference(config,state,target,commands=arrays['commands'],capture=False)
                    if any(not np.array_equal(arrays[k],repeat[k]) for k in repeat):raise ValueError('Fresh reference replay differs')
                    keys=('agentview','wrist','eef_position','qpos','qvel','object_position')
                    if initial is not None and any(not np.array_equal(initial[k],arrays[k][0]) for k in keys):raise ValueError('Reference pair starts differ')
                    initial={k:arrays[k][0].copy() for k in keys}
                    reasons=[]
                    if not valid_approach(meta['approach'],config['approach_limits']):reasons.append('approach_geometry_or_contacts')
                    for metric in meta['appearance_validation']:
                        if not metric['depth_exact'] or metric['changed_outside_boundary_support']!=0:reasons.append('appearance_support_or_depth')
                        if metric['changed_pixels']/config['camera_resolution']**2<config['minimum_changed_fraction_per_camera']:reasons.append('appearance_not_visible')
                    np.savez_compressed(dest/'reference.npz',**arrays)
                    meta.update(accepted=not reasons,rejection_reasons=sorted(set(reasons)),fresh_unedited_replay_exact=True,
                                reference_sha256=digest_file(dest/'reference.npz'))
                    for app in ('original',*config['appearance']['factors']):sheet(arrays[f'clip_{app}'],dest/f'{app}_sheet.jpg')
                except Exception as exc:
                    meta={'accepted':False,'rejection_reasons':['reference_error'],'error':repr(exc)}
                write(dest/'metadata.json',meta);pair.append(meta)
            row={'task':task['task_id'],'state':state,'case':case_id(task['task_id'],state),'accepted':all(x['accepted'] for x in pair),
                 'reference_rejections':{target:meta['rejection_reasons'] for target,meta in zip(task['targets'],pair)}}
            rows.append(row);write(root/partition/'reference_selection.json',rows);print(json.dumps(row),flush=True)
    counts={str(t['task_id']):sum(r['accepted'] and r['task']==t['task_id'] for r in rows) for t in protocol['tasks']}
    minimum=len(protocol['calibration_states']) if partition=='calibration' else protocol['minimum_test_states_per_task']
    manifest={p.relative_to(root).as_posix():digest_file(p) for p in (root/partition).rglob('*') if p.is_file()}
    write(root/partition/'reference_manifest.json',manifest);verify(root)
    status(root,f'{partition}_references_complete' if all(n>=minimum for n in counts.values()) else f'stopped_{partition}_references',
           accepted_states=counts,candidate_states_per_task=len(protocol[f'{partition}_states']))


def verify_manifest(root,path):
    for name,h in read(path).items():
        if digest_file(root/name)!=h:raise ValueError(f'Changed input: {name}')


def accepted_cases(root,partition):return [r for r in read(root/partition/'reference_selection.json') if r['accepted']]
def load_references(root,partition,case):
    out={}
    for target in ('a','b'):
        path=root/partition/case/'references'/target/'reference.npz'
        with np.load(path,allow_pickle=False) as a:out[target]={k:a[k].copy() for k in a.files if k.startswith('clip_')}
    for app in out['a']:
        if not np.array_equal(out['a'][app][0],out['b'][app][0]):raise ValueError('Current reference frames differ')
    return out


def make_pilot(protocol,row,loaded):
    config={**read(protocol['pilot_config']),'suite':protocol['suite'],'task_id':row['task'],'initial_state_index':row['state'],'action_seed':0}
    config.pop('donor_prompts',None)
    p=new_pilot(config,loaded);p.verify_replay();attach_physics(p);return p


def attach_physics(p):
    """Record contacts at native integration steps and object/proprio state live."""
    def rollout(commands):
        initial=observation_arrays(p.observation,p.env)
        keys=[k for k in initial if k not in ('agentview','wrist','eef_position','eef_quaternion')]
        records=[initial];old_step=p.step_env
        def tracked_step(*args,**kwargs):
            result=old_step(*args,**kwargs)
            records.append(observation_arrays(result.observation,p.env));return result
        try:
            with RobotContactCapture(p.env) as contacts:
                contacts.enabled=True;p.step_env=tracked_step;result=p.rollout(commands)
            p.last_rollout.update({k:np.stack([r[k] for r in records]) for k in keys})
            result['metadata'].update(contact_events=contacts.events,integrator_substeps_checked=contacts.substeps,
                object_names=list(p.env.env.obj_body_id),task_id=p.config['task_id'])
            return result
        finally:p.step_env=old_step
    p.session.rollout_fn=rollout


def model_identity(p):
    full=provenance(p)
    return {k:full[k] for k in ('repo_revision','source_sha256','backend','checkpoint','statistics','resolved_config','precision','base_assets','versions')}


def set_request(p,native,reference,appearance,seed):
    import torch
    rgb=reference['a'][f'clip_{appearance}'][0]
    image=torch.as_tensor(rgb.copy(),device=p.adapter.model.device).permute(2,0,1).unsqueeze(0).to(p.adapter.model.torch_dtype)
    image=image*(2.0/255.0)-1.0
    if appearance=='original' and not torch.equal(image,native.observation.image):raise ValueError('Native preprocessing differs')
    obs=replace(native.observation,image=image,metadata={**native.observation.metadata,'observed_appearance':appearance})
    p.request=replace(native,observation=obs,seed=seed);p.session.request=p.request;p.config['action_seed']=seed


def encode_clips(root,partition,row,p,native,reference,observed,future_appearances,manifest):
    set_request(p,native,reference,observed,0);_,capture,_=p.infer();anchor=_array(capture['used'])
    for target in ('a','b'):
        for app in future_appearances:
            folder=root/partition/row['case']/'codecs'/observed/f'{target}_{app}';folder.mkdir(parents=True)
            arrays,metrics=encode_reference_video(p.adapter.model,reference[target][f'clip_{app}'],reference['a'][f'clip_{observed}'][0],anchor)
            np.savez_compressed(folder/'codec.npz',**arrays);write(folder/'metrics.json',metrics)
            sheet(arrays['source_rgb'],folder/'source_sheet.jpg');images(arrays['decoded_rgb'],folder/'decoded')
            for name in ('codec.npz','metrics.json','source_sheet.jpg','decoded_sheet.jpg'):
                manifest[(folder/name).relative_to(root).as_posix()]=digest_file(folder/name)


def review_template(root,partition,manifest):
    clips=sorted({str(Path(k).parent).replace('\\','/') for k in manifest})
    write(root/partition/'codec_manifest.json',manifest)
    write(root/partition/'codec_review.json',{'method':None,'human_review':'pending','before_donor_outcomes':False,
        'clips':{k:None for k in clips}})


def calibration_codecs(root):
    require(root,'calibration_references_complete');verify(root)
    verify_manifest(root,root/'calibration/reference_manifest.json')
    protocol=read(root/'protocol.json');p=None;manifest={};rows=[]
    try:
        status(root,'calibration_codecs_running')
        for row in accepted_cases(root,'calibration'):
            p=make_pilot(protocol,row,p)
            if not (root/'model_identity.json').exists():write(root/'model_identity.json',model_identity(p))
            ref=load_references(root,'calibration',row['case'])
            encode_clips(root,'calibration',row,p,p.request,ref,'original',['original',*protocol['appearance_candidates']],manifest)
            folder=root/'calibration'/row['case']/'codecs/original'
            futures={f'{t}_{a}':load_future(folder/f'{t}_{a}/codec.npz')[:,:,1:] for t in ('a','b') for a in ('original',*protocol['appearance_candidates'])}
            motion=rms_difference(futures['a_original'],futures['b_original'])
            for app in protocol['appearance_candidates']:
                names=[f'{t}_{a}' for t in ('a','b') for a in ('original',app)]
                rows.append({'task':row['task'],'state':row['state'],'appearance':app,'motion_rms':motion,
                    'appearance_rms':float(np.mean([rms_difference(futures[f'{t}_original'],futures[f'{t}_{app}']) for t in ('a','b')])),
                    'valid':all(read(folder/n/'metrics.json')['future_rgb_mae_255']<=protocol['codec_max_future_mae_255'] for n in names)})
            write(root/'calibration/distance_rows.json',rows)
        review_template(root,'calibration',manifest);verify(root);status(root,'calibration_codecs_ready',policy_execution_branches=0)
    finally:
        if p is not None:p.close()


def read_review(root,partition):
    review=read(root/partition/'codec_review.json');verify_manifest(root,root/partition/'codec_manifest.json')
    if review['method'] not in ('AI_assisted','human') or not review['before_donor_outcomes']:raise ValueError('Actual visual review required')
    expected={str(Path(k).parent).replace('\\','/') for k in read(root/partition/'codec_manifest.json')}
    if set(review['clips'])!=expected:raise ValueError('Review omitted or added a codec')
    for name,item in review['clips'].items():
        if not isinstance(item,dict) or type(item.get('accepted')) is not bool or not item.get('notes'):raise ValueError(f'Missing review: {name}')
    return review


def select(root):
    require(root,'calibration_codecs_ready');verify(root);protocol=read(root/'protocol.json');review=read_review(root,'calibration')
    rows=read(root/'calibration/distance_rows.json')
    for row in rows:
        prefix='calibration/'+case_id(row['task'],row['state'])+'/codecs/original/'
        row['valid'] &= all(review['clips'][prefix+f'{t}_{a}']['accepted'] for t in ('a','b') for a in ('original',row['appearance']))
    selection=choose_appearance(rows,list(protocol['appearance_candidates']))
    selection.update(at=now(),review_sha256=digest_file(root/'calibration/codec_review.json'),distance_rows_sha256=digest_file(root/'calibration/distance_rows.json'),
        model_identity_sha256=digest_file(root/'model_identity.json'),test_states_unopened=True)
    write(root/'selection.json',selection);write(root/'selection_freeze.json',{'sha256':digest_file(root/'selection.json')})
    status(root,'calibrated',selected=selection['selected'])


def verify_selection(root):
    selection=read(root/'selection.json')
    if digest_file(root/'selection.json')!=read(root/'selection_freeze.json')['sha256']:raise ValueError('Calibration selection changed')
    for path,key in [('calibration/codec_review.json','review_sha256'),('calibration/distance_rows.json','distance_rows_sha256'),('model_identity.json','model_identity_sha256')]:
        if digest_file(root/path)!=selection[key]:raise ValueError('Calibration provenance changed')


def controls(root):
    require(root,'test_references_complete');verify(root);verify_selection(root);verify_manifest(root,root/'test/reference_manifest.json')
    protocol=read(root/'protocol.json');appearances=['original',read(root/'selection.json')['selected']]
    p=None;manifest={};gates=[];branches=0
    try:
        status(root,'test_controls_running');checked_model=False
        for row in accepted_cases(root,'test'):
            p=make_pilot(protocol,row,p);native=p.request;ref=load_references(root,'test',row['case'])
            if not checked_model:
                if model_identity(p)!=read(root/'model_identity.json'):raise ValueError('Calibration/test model identity differs')
                checked_model=True
            for observed in appearances:
                encode_clips(root,'test',row,p,native,ref,observed,appearances,manifest)
                for seed in protocol['action_seeds']:
                    set_request(p,native,ref,observed,seed);result,capture,_=p.infer();baseline=_array(capture['used'])
                    case=root/'test'/row['case']/f'seed_{seed}'/observed
                    write(case/'context.json',{'task':row['task'],'state':row['state'],'seed':seed,'observed':observed,
                        'prompt':p.request.observation.prompt,'image_sha256':digest_array(p.request.observation.image),
                        'proprio_sha256':digest_array(p.request.observation.proprio),'initial_state_sha256':digest_array(p.initial_state)})
                    b,e=execute(p,case,'baseline',None,baseline)
                    if not np.array_equal(b.future,baseline) or not np.array_equal(b.actions,_array(result.actions)):raise ValueError('Native repeat differs')
                    execute(p,case,'identity',baseline,baseline,e,b.actions)
                    rt,metrics=roundtrip_future(p.adapter.model,baseline)
                    np.savez_compressed(case/'roundtrip_input.npz',future=rt['future']);write(case/'roundtrip_metrics.json',metrics)
                    _,trace=execute(p,case,'roundtrip_baseline',rt['future'],baseline)
                    delta=trajectory_difference(e['eef_position'],trace['eef_position']);gate={**row,'seed':seed,'observed':observed,**delta,
                        'pass':baseline_gate(delta,protocol['baseline_limits']),'identity_exact':True}
                    gates.append(gate);branches+=3;write(case/'baseline_gate.json',gate);write(root/'test/baseline_gates.json',gates)
                    write(root/'progress.json',{'stage':'controls','branches':branches,'case':row['case'],'seed':seed,'observed':observed})
        review_template(root,'test',manifest);verify(root)
        status(root,'test_controls_ready' if all(g['pass'] for g in gates) else 'stopped_baseline_gate',branches=branches,failed_gates=sum(not g['pass'] for g in gates))
    finally:
        if p is not None:p.close()


def donors(root):
    require(root,'test_controls_ready');verify(root);verify_selection(root);verify_manifest(root,root/'test/reference_manifest.json')
    protocol=read(root/'protocol.json');review=read_review(root,'test');appearances=['original',read(root/'selection.json')['selected']]
    cases=[];excluded=[]
    for row in accepted_cases(root,'test'):
        prefix='test/'+row['case']+'/codecs/'
        valid=all(item['accepted'] and read(root/name/'metrics.json')['future_rgb_mae_255']<=protocol['codec_max_future_mae_255'] for name,item in review['clips'].items() if name.startswith(prefix))
        (cases if valid else excluded).append(row)
    counts={str(t['task_id']):sum(r['task']==t['task_id'] for r in cases) for t in protocol['tasks']}
    write(root/'donor_freeze.json',{'at':now(),'review_sha256':digest_file(root/'test/codec_review.json'),
        'codec_manifest_sha256':digest_file(root/'test/codec_manifest.json'),'cases':cases,'codec_excluded':excluded})
    if any(n<protocol['minimum_test_states_per_task'] for n in counts.values()):
        status(root,'stopped_test_codec_gate',included_per_task=counts);return
    p=None;branches=read(root/'status.json')['branches'];checked_model=False
    try:
        status(root,'test_donors_running',branches=branches)
        for row in cases:
            p=make_pilot(protocol,row,p);native=p.request;ref=load_references(root,'test',row['case'])
            if not checked_model:
                if model_identity(p)!=read(root/'model_identity.json'):raise ValueError('Model identity differs')
                checked_model=True
            for seed in protocol['action_seeds']:
                for observed in appearances:
                    set_request(p,native,ref,observed,seed);case=root/'test'/row['case']/f'seed_{seed}'/observed;context=read(case/'context.json')
                    if context['prompt']!=p.request.observation.prompt or context['image_sha256']!=digest_array(p.request.observation.image) or context['proprio_sha256']!=digest_array(p.request.observation.proprio) or context['initial_state_sha256']!=digest_array(p.initial_state):raise ValueError('Recipient context changed')
                    baseline=load_future(case/'baseline/policy.npz')
                    for target in ('a','b'):
                        for app in appearances:
                            name=f'{target}_{app}';donor=load_future(root/'test'/row['case']/'codecs'/observed/name/'codec.npz')
                            execute(p,case,name,donor,baseline);branches+=1
                            write(root/'progress.json',{'stage':'donors','branches':branches,'case':row['case'],'seed':seed,'observed':observed,'arm':name})
        verify(root);status(root,'complete',branches=branches,included_per_task=counts,human_review='pending')
    finally:
        if p is not None:p.close()


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=['prepare','calibration-references','calibration-codecs','select','test-references','controls','donors'])
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--protocol',type=Path,default=Path('configs/experiments/e1_visual_replication.json'))
    args=parser.parse_args()
    previous_status=read(args.output/'status.json') if (args.output/'status.json').exists() else None
    try:
        if args.stage=='prepare':prepare(args.output,read(args.protocol))
        elif args.stage.endswith('-references'):references(args.output,args.stage.split('-')[0])
        elif args.stage=='calibration-codecs':calibration_codecs(args.output)
        elif args.stage=='select':select(args.output)
        elif args.stage=='controls':controls(args.output)
        elif args.stage=='donors':donors(args.output)
    except Exception as exc:
        if args.stage!='prepare' and (args.output/'status.json').exists():
            previous=read(args.output/'status.json');write(args.output/'failure.json',{'at':now(),'stage':args.stage,'previous_status':previous,'error':repr(exc)})
            if previous!=previous_status:status(args.output,'failed',stage=args.stage,error=repr(exc))
        raise
