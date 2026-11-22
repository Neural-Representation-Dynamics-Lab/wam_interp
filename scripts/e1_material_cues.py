"""Frozen fixed-motion object-material by recipient-instruction pilot."""
import argparse
from dataclasses import replace
from pathlib import Path
import json,zipfile
import numpy as np
from e1_replication import new_pilot,write,now,load_future
from e1_simulator_pairs import run_reference,sheet
from e1_video_bottleneck import execute,images
from e1_visual_replication import verify,attach_physics,model_identity,verify_manifest
from wam_interp.experiments.simulator_materials import ObjectMaterialCapture
from wam_interp.experiments.simulator_video import valid_approach,encode_reference_video
from wam_interp.experiments.material_cues import masked_registration,cue_interaction
from wam_interp.experiments.material_cues import robot_pixel_audit
from wam_interp.experiments.libero_pilot import digest_file,digest_array
from wam_interp.experiments.fastwam_session import _array
from wam_interp.experiments.video_bottleneck import roundtrip_future,trajectory_difference,baseline_gate
from wam_interp.experiments.controlled_replication import state_bootstrap

read=lambda p:json.loads(Path(p).read_text())

def status(root,value,**extra):write(root/'status.json',{'status':value,'at':now(),**extra})
def require(root,value):
    if read(root/'status.json')['status']!=value:raise ValueError('Wrong stage; no silent restart')

def prepare(root,p):
    if root.exists():raise FileExistsError('Fresh output directory required')
    if len(set(p['states']))!=len(p['states']) or set(p['instructions'])!={'soup','tomato','cream'} or p['variants']!=['original','swapped'] or p['motions'] not in (['a'],['a','b']):raise ValueError('Unsupported or unbalanced design')
    root.mkdir(parents=True);write(root/'protocol.json',p)
    with zipfile.ZipFile(root/'source.zip','w',zipfile.ZIP_DEFLATED) as z:
        for folder in ('src','scripts','configs'):
            for f in sorted(Path(folder).rglob('*')):
                if f.is_file() and f.suffix in ('.py','.json','.yaml'):z.write(f,f.as_posix())
        for name in ('__init__.py','pyproject.toml'):z.write(name,name)
    write(root/'freeze.json',{'at':now(),'source_sha256':digest_file(root/'source.zip'),'protocol_sha256':digest_file(root/'protocol.json')})
    status(root,'prepared')

def references(root):
    require(root,'prepared');verify(root);p=read(root/'protocol.json');status(root,'references_running');rows=[]
    base=read(p['reference_config']);base.pop('previous_results',None)
    config={**base,'suite':p['suite'],'task_id':p['task_id'],'initial_states':p['states'],'targets':p['targets'],
        'appearance':{'material':'living_room_table','factors':{'swapped':[1,1,1]}},'save_variant_depths':True,
        'intervention':'Native geom material-ID exchange; appearance factors are placeholders for the generic capture interface.',
        'purpose':'Fixed robot motion, object-label/material cues, recipient instruction factorial'}
    factory=lambda env:ObjectMaterialCapture(env,list(p['targets'].values()),coverage_scale=4)
    for state in p['states']:
        folder=root/f'state_{state}';write(folder/'reference_config.json',config);pair=[];initial=None
        for motion in p['motions']:
            dest=folder/'references'/motion;dest.mkdir(parents=True)
            try:
                arrays,meta=run_reference(config,state,motion,capture_factory=factory)
                repeat,_=run_reference(config,state,motion,commands=arrays['commands'],capture=False,capture_factory=factory)
                if any(not np.array_equal(arrays[k],repeat[k]) for k in repeat):raise ValueError('Fresh unedited replay differs')
                current={k:arrays[k][0] for k in ('qpos','qvel','object_position','agentview','wrist')}
                if initial is not None and any(not np.array_equal(v,initial[k]) for k,v in current.items()):raise ValueError('Motion variants have different starts')
                initial=current;reasons=[]
                if not valid_approach(meta['approach'],config['approach_limits']):reasons.append('approach_geometry_or_contacts')
                pixel_audits=[]
                for camera in ('agentview','wrist'):
                    metrics=[m for m in meta['appearance_validation'] if m['camera']==camera]
                    if any(not m['depth_exact'] or m['changed_outside_boundary_support'] for m in metrics):reasons.append('render_support_or_depth')
                    if sum(m['changed_pixels']>=p['minimum_changed_pixels'] for m in metrics)<p['minimum_visible_frames_per_camera']:reasons.append('insufficient_cue_visibility')
                    mask=arrays[f'video_{camera}_robot_mask'];orig=arrays[f'video_{camera}_rgb'];edit=arrays[f'video_{camera}_swapped']
                    if not np.array_equal(orig[mask],edit[mask]):reasons.append('source_robot_pixels_changed')
                    for frame in range(9):
                        _,audit=robot_pixel_audit(orig[frame],edit[frame],arrays[f'video_{camera}_robot_support_mask'][frame],arrays[f'video_{camera}_mask'][frame])
                        pixel_audits.append({'camera':camera,'frame':frame,**audit})
                np.savez_compressed(dest/'reference.npz',**arrays)
                meta.update(accepted=not reasons,rejection_reasons=sorted(set(reasons)),reference_sha256=digest_file(dest/'reference.npz'),
                    source_robot_interior_pixels_exact=True,robot_pixel_audits=pixel_audits,fresh_unedited_replay_exact=True,material_objects=list(p['targets'].values()),
                    identity_scope='Native material cues only; geometry and UV coordinates unchanged; full perceived identity requires separate validation.')
                for variant in p['variants']:sheet(arrays[f'clip_{variant}'],dest/f'{variant}_sheet.jpg')
            except Exception as exc:meta={'accepted':False,'rejection_reasons':['reference_error'],'error':repr(exc)}
            write(dest/'metadata.json',meta);pair.append(meta)
        rows.append({'state':state,'accepted':all(x['accepted'] for x in pair),'reasons':{m:x['rejection_reasons'] for m,x in zip(p['motions'],pair)}})
        write(root/'reference_selection.json',rows);print(json.dumps(rows[-1]),flush=True)
    manifest={f.relative_to(root).as_posix():digest_file(f) for f in root.glob('state_*/**/*') if f.is_file()}
    write(root/'reference_manifest.json',manifest);verify(root)
    status(root,'references_ready' if sum(r['accepted'] for r in rows)>=p['minimum_states'] else 'stopped_reference_gate',accepted_states=sum(r['accepted'] for r in rows))

def load_refs(root,state):
    out={}
    for m in read(root/'protocol.json')['motions']:
        with np.load(root/f'state_{state}/references/{m}/reference.npz',allow_pickle=False) as z:out[m]={k:z[k].copy() for k in z.files}
    for v in ('original','swapped'):
        if any(not np.array_equal(out['a'][f'clip_{v}'][0],ref[f'clip_{v}'][0]) for ref in out.values()):raise ValueError('Initial cue images differ across motions')
    return out

def pilot(p,state,loaded):
    c={**read(p['pilot_config']),'suite':p['suite'],'task_id':p['task_id'],'initial_state_index':state,'action_seed':0}
    c.pop('donor_prompts',None);obj=new_pilot(c,loaded);obj.verify_replay();attach_physics(obj);return obj

def request(obj,native,refs,current,seed,prompt):
    import torch
    rgb=refs['a'][f'clip_{current}'][0]
    image=torch.as_tensor(rgb.copy(),device=obj.adapter.model.device).permute(2,0,1).unsqueeze(0).to(obj.adapter.model.torch_dtype)*(2/255)-1
    if current=='original' and not torch.equal(image,native.observation.image):raise ValueError('Native preprocessing differs')
    obs=replace(native.observation,image=image,prompt=prompt,metadata={**native.observation.metadata,'material_cues':current})
    obj.request=replace(native,observation=obs,seed=seed);obj.session.request=obj.request;obj.config['action_seed']=seed

def run_branch(obj,case,name,donor,baseline,*args):
    b,e=execute(obj,case,name,donor,baseline,*args)
    path=case/name/'record.json';record=read(path)
    record['simulation_task_instruction']=obj.task.language
    record['recipient_instruction']=obj.request.observation.prompt
    record['benchmark_success_interpretation']='Diagnostic input intervention; physical identities original; not benchmark performance.'
    write(path,record);return b,e

def montage_mask(ref,field):
    from PIL import Image
    out=[]
    for i in range(9):
        cams=[]
        for cam in ('agentview','wrist'):
            a=ref[f'video_{cam}_{field}'][i][::-1,::-1].astype('uint8')*255
            cams.append(np.asarray(Image.fromarray(a).resize((224,224),Image.Resampling.NEAREST))>0)
        out.append(np.concatenate(cams,axis=1))
    return np.stack(out)

def controls(root,*,pilot_factory=None):
    require(root,'references_ready');verify(root);verify_manifest(root,root/'reference_manifest.json')
    p=read(root/'protocol.json');obj=None;manifest={};gates=[];codec_gates=[];branches=0
    status(root,'controls_running')
    try:
        for row in read(root/'reference_selection.json'):
            if not row['accepted']:continue
            state=row['state'];obj=(pilot if pilot_factory is None else pilot_factory)(p,state,obj);native=obj.request;refs=load_refs(root,state)
            if not (root/'model_identity.json').exists():write(root/'model_identity.json',model_identity(obj))
            for current in p['variants']:
                request(obj,native,refs,current,0,p['instructions']['soup']);_,capture,_=obj.infer();anchor=_array(capture['used'])
                for motion in p['motions']:
                    encoded={}
                    for future in p['variants']:
                        dest=root/f'state_{state}/codecs/{current}/{motion}_{future}';dest.mkdir(parents=True)
                        arrays,metrics=encode_reference_video(obj.adapter.model,refs[motion][f'clip_{future}'],refs['a'][f'clip_{current}'][0],anchor)
                        np.savez_compressed(dest/'codec.npz',**arrays);write(dest/'metrics.json',metrics)
                        sheet(arrays['source_rgb'],dest/'source_sheet.jpg');images(arrays['decoded_rgb'],dest/'decoded')
                        encoded[future]=arrays
                        for n in ('codec.npz','metrics.json','source_sheet.jpg','decoded_sheet.jpg'):manifest[(dest/n).relative_to(root).as_posix()]=digest_file(dest/n)
                    mask=montage_mask(refs[motion],'gripper_mask');registration=[]
                    for frame in range(1,9):
                        a,b=[(encoded[v]['decoded_rgb'][0,:,frame].transpose(1,2,0)+1)*127.5 for v in p['variants']]
                        for camera,sl in [('agentview',slice(0,224)),('wrist',slice(224,448))]:
                            m=masked_registration(a[:,sl],b[:,sl],mask[frame,:,sl]);registration.append({'frame':frame,'camera':camera,**m})
                    measurable=[m for m in registration if m['measurable']]
                    passed=len(measurable)>=p['minimum_measurable_gripper_frames_total'] and all(m['mae_255']<=p['codec_max_gripper_mae_255'] and m['shift_pixels']<=p['codec_max_gripper_shift_pixels'] for m in measurable)
                    gate={'state':state,'current':current,'motion':motion,'pass':bool(passed),'registration':registration,
                        'scope':'Source motion exact; decoded image registration is a finite-resolution diagnostic, not equality of internal robot features.'}
                    codec_gates.append(gate);write(root/'gripper_codec_gates.json',codec_gates)
                for language,prompt in p['instructions'].items():
                    for seed in p['action_seeds']:
                        request(obj,native,refs,current,seed,prompt);result,capture,_=obj.infer();baseline=_array(capture['used'])
                        if not np.array_equal(baseline[:,:,:1],anchor[:,:,:1]):raise ValueError('Observed anchor depends on recipient instruction/seed')
                        case=root/f'state_{state}/seed_{seed}/{current}/{language}'
                        write(case/'context.json',{'state':state,'seed':seed,'current':current,'language':language,'prompt':prompt,
                            'image_sha256':digest_array(obj.request.observation.image),'proprio_sha256':digest_array(obj.request.observation.proprio),'state_sha256':digest_array(obj.initial_state)})
                        b,e=run_branch(obj,case,'baseline',None,baseline)
                        if not np.array_equal(b.future,baseline) or not np.array_equal(b.actions,_array(result.actions)):raise ValueError('Native repeat differs')
                        run_branch(obj,case,'identity',baseline,baseline,e,b.actions)
                        rt,metrics=roundtrip_future(obj.adapter.model,baseline);np.savez_compressed(case/'roundtrip_input.npz',future=rt['future']);write(case/'roundtrip_metrics.json',metrics)
                        _,trace=run_branch(obj,case,'roundtrip_baseline',rt['future'],baseline)
                        delta=trajectory_difference(e['eef_position'],trace['eef_position']);gate={'state':state,'seed':seed,'current':current,'language':language,**delta,'pass':baseline_gate(delta,p['baseline_limits'])}
                        gates.append(gate);write(root/'baseline_gates.json',gates);branches+=3
                        write(root/'progress.json',{'stage':'controls','branches':branches,**{k:gate[k] for k in ('state','seed','current','language')}})
        write(root/'codec_manifest.json',manifest)
        write(root/'codec_review.json',{'method':None,'before_donor_outcomes':False,'human_review':'pending','clips':{str(Path(k).parent).replace('\\','/'):None for k in manifest}})
        verify(root);status(root,'controls_ready' if all(g['pass'] for g in gates) else 'stopped_baseline_gate',branches=branches,failed_gripper_pairs=sum(not x['pass'] for x in codec_gates))
    finally:
        if obj is not None:obj.close()

def donors(root,*,pilot_factory=None):
    require(root,'controls_ready');verify(root);verify_manifest(root,root/'reference_manifest.json');verify_manifest(root,root/'codec_manifest.json')
    p=read(root/'protocol.json');review=read(root/'codec_review.json');expected={str(Path(k).parent).replace('\\','/') for k in read(root/'codec_manifest.json')}
    if review['method'] not in ('AI_assisted','human') or not review['before_donor_outcomes'] or set(review['clips'])!=expected:raise ValueError('Actual complete pre-outcome review required')
    if any(not isinstance(x,dict) or type(x.get('accepted')) is not bool or not x.get('notes') for x in review['clips'].values()):raise ValueError('Missing visual decisions')
    selected=[];excluded=[]
    for row in read(root/'reference_selection.json'):
        if not row['accepted']:continue
        state=row['state'];names=[n for n in expected if n.startswith(f'state_{state}/')]
        valid=all(review['clips'][n]['accepted'] and read(root/n/'metrics.json')['future_rgb_mae_255']<=p['codec_max_future_mae_255'] for n in names)
        valid &= all(g['pass'] for g in read(root/'gripper_codec_gates.json') if g['state']==state)
        (selected if valid else excluded).append(state)
    write(root/'donor_freeze.json',{'at':now(),'states':selected,'excluded_states':excluded,'review_sha256':digest_file(root/'codec_review.json'),'codec_manifest_sha256':digest_file(root/'codec_manifest.json')})
    if len(selected)<p['minimum_states']:status(root,'stopped_codec_gate',included_states=selected);return
    obj=None;branches=read(root/'status.json')['branches'];status(root,'donors_running',branches=branches)
    try:
        for state in selected:
            obj=(pilot if pilot_factory is None else pilot_factory)(p,state,obj);native=obj.request;refs=load_refs(root,state)
            if state==selected[0] and model_identity(obj)!=read(root/'model_identity.json'):raise ValueError('Model identity differs')
            for current in p['variants']:
                for language,prompt in p['instructions'].items():
                    for seed in p['action_seeds']:
                        request(obj,native,refs,current,seed,prompt);case=root/f'state_{state}/seed_{seed}/{current}/{language}';context=read(case/'context.json')
                        if context['prompt']!=obj.request.observation.prompt or context['image_sha256']!=digest_array(obj.request.observation.image) or context['proprio_sha256']!=digest_array(obj.request.observation.proprio) or context['state_sha256']!=digest_array(obj.initial_state):raise ValueError('Context drift')
                        baseline=load_future(case/'baseline/policy.npz')
                        for motion in p['motions']:
                            for future in p['variants']:
                                name=f'{motion}_{future}';donor=load_future(root/f'state_{state}/codecs/{current}/{name}/codec.npz')
                                run_branch(obj,case,name,donor,baseline);branches+=1
                                write(root/'progress.json',{'stage':'donors','branches':branches,'state':state,'seed':seed,'current':current,'language':language,'arm':name})
        verify(root);status(root,'complete',branches=branches,states=selected,human_review='pending')
    finally:
        if obj is not None:obj.close()

if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('stage',choices=['prepare','references','controls','donors'])
    ap.add_argument('--output',required=True,type=Path);ap.add_argument('--protocol',type=Path,default=Path('configs/experiments/e1_material_cues.json'));args=ap.parse_args()
    try:
        if args.stage=='prepare':prepare(args.output,read(args.protocol))
        else:globals()[args.stage](args.output)
    except Exception as exc:
        if args.stage!='prepare':write(args.output/'failure.json',{'stage':args.stage,'at':now(),'error':repr(exc)});status(args.output,'failed',stage=args.stage,error=repr(exc))
        raise
