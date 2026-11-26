"""Controlled simulator RGB futures, crossed with current-image appearance.

The `controls` stage prepares reference latents and baseline checks. The `run`
stage requires a frozen visual review and all baseline gates before donors.
"""
import argparse
from dataclasses import replace
from datetime import datetime,timezone
import json,zipfile
from pathlib import Path
import numpy as np
from e1_replication import new_pilot,write,now,load_future
from e1_video_bottleneck import execute,images
from e1_simulator_pairs import sheet
from wam_interp.experiments.libero_pilot import provenance,digest_file,digest_array
from wam_interp.experiments.fastwam_session import _array
from wam_interp.experiments.simulator_video import encode_reference_video
from wam_interp.experiments.video_bottleneck import roundtrip_future,trajectory_difference,baseline_gate


def read(p):return json.loads(Path(p).read_text())


def freeze(root,protocol):
    if root.exists():raise FileExistsError('Use a fresh output directory')
    ref=Path(protocol['references'])
    if read(ref/'status.json')['status']!='awaiting_visual_codec_review':raise ValueError('Reference gates have not passed')
    for state in protocol['initial_states']:
        for target in protocol['targets']:
            meta=read(ref/f'state_{state}/{target}/metadata.json')
            if meta['accepted'] is not True or digest_file(ref/f'state_{state}/{target}/reference.npz')!=meta['reference_sha256']:
                raise ValueError('Invalid or changed reference')
    root.mkdir(parents=True);write(root/'protocol.json',protocol)
    with zipfile.ZipFile(root/'source.zip','w',zipfile.ZIP_DEFLATED) as z:
        for folder in ('src','scripts','configs'):
            for p in sorted(Path(folder).rglob('*')):
                if p.is_file() and p.suffix in ('.py','.json','.yaml'):z.write(p,p.as_posix())
        for n in ('__init__.py','pyproject.toml'):z.write(n,n)
    inputs={str(p):digest_file(p) for p in ref.rglob('*') if p.is_file() and p.suffix in ('.json','.npz','.zip')}
    write(root/'freeze.json',{'at':now(),'source_sha256':digest_file(root/'source.zip'),'protocol_sha256':digest_file(root/'protocol.json'),'reference_inputs':inputs})


def verify(root):
    f=read(root/'freeze.json')
    if digest_file(root/'source.zip')!=f['source_sha256'] or digest_file(root/'protocol.json')!=f['protocol_sha256']:raise ValueError('Source/protocol changed')
    for path,h in f['reference_inputs'].items():
        if digest_file(path)!=h:raise ValueError(f'Reference changed: {path}')
    with zipfile.ZipFile(root/'source.zip') as z:
        for name in z.namelist():
            if Path(name).read_bytes()!=z.read(name):raise ValueError(f'Inference source changed: {name}')


def reference_arrays(protocol,state):
    arrays={}
    for target in protocol['targets']:
        path=Path(protocol['references'])/f'state_{state}/{target}/reference.npz'
        with np.load(path,allow_pickle=False) as a:arrays[target]={k:a[k].copy() for k in a.files if k.startswith('clip_') or k=='object_position'}
    for appearance in protocol['observed_appearances']:
        if not np.array_equal(arrays['soup'][f'clip_{appearance}'][0],arrays['tomato'][f'clip_{appearance}'][0]):raise ValueError('Reference current images differ')
    return arrays


def set_request(p,native,reference,appearance,seed):
    import torch
    rgb=reference['soup'][f'clip_{appearance}'][0]
    image=torch.as_tensor(rgb.copy(),device=p.adapter.model.device).permute(2,0,1).unsqueeze(0).to(p.adapter.model.torch_dtype)
    image=image*(2.0/255.0)-1.0
    if appearance=='original' and not torch.equal(image,native.observation.image):raise ValueError('Reference preprocessing differs from native observation')
    obs=replace(native.observation,image=image,metadata={**native.observation.metadata,'observed_appearance':appearance,'appearance_intervention':'prepared image only; identical physical simulator'})
    p.request=replace(native,observation=obs,seed=seed);p.session.request=p.request;p.config['action_seed']=seed


def controls(root,protocol):
    freeze(root,protocol);p=None;gates=[];codecs={}
    try:
        write(root/'status.json',{'status':'controls_running','at':now()})
        for state in protocol['initial_states']:
            cfg={**read(protocol['pilot_config']),'initial_state_index':state,'action_seed':0}
            p=new_pilot(cfg,p);p.verify_replay();native=p.request;reference=reference_arrays(protocol,state)
            state_root=root/f'state_{state}';write(state_root/'common_context.json',provenance(p))
            for appearance in protocol['observed_appearances']:
                set_request(p,native,reference,appearance,0)
                _,captured,_=p.infer();anchor=_array(captured['used'])
                for target in protocol['targets']:
                    for future_appearance in protocol['future_appearances']:
                        folder=state_root/'codecs'/appearance/f'{target}_{future_appearance}';folder.mkdir(parents=True)
                        arrays,metrics=encode_reference_video(p.adapter.model,reference[target][f'clip_{future_appearance}'],reference['soup'][f'clip_{appearance}'][0],anchor)
                        np.savez_compressed(folder/'codec.npz',**arrays);write(folder/'metrics.json',metrics)
                        sheet(arrays['source_rgb'],folder/'source_sheet.jpg');images(arrays['decoded_rgb'],folder/'decoded')
                        codecs[folder.relative_to(root).as_posix()]={n:digest_file(folder/n) for n in ('codec.npz','metrics.json','source_sheet.jpg','decoded_sheet.jpg')}
                for seed in protocol['action_seeds']:
                    set_request(p,native,reference,appearance,seed)
                    result,captured,_=p.infer();baseline=_array(captured['used'])
                    case=state_root/f'seed_{seed}'/appearance
                    write(case/'context.json',{'state':state,'seed':seed,'observed_appearance':appearance,'image_sha256':digest_array(p.request.observation.image),'proprio_sha256':digest_array(p.request.observation.proprio),'recipient_instruction':p.request.observation.prompt,'common_context_sha256':digest_file(state_root/'common_context.json')})
                    b,e=execute(p,case,'baseline',None,baseline)
                    if not np.array_equal(b.future,baseline) or not np.array_equal(b.actions,_array(result.actions)):raise ValueError('Fresh native replay differs')
                    if appearance=='original':
                        old=Path(protocol['previous_results'])/f'state_{state}/seed_{seed}/baseline'
                        with np.load(old/'policy.npz') as a:
                            if not np.array_equal(b.actions,a['actions']) or not np.array_equal(b.future,a['future']):raise ValueError('Historical native policy differs')
                        with np.load(old/'execution.npz') as a:
                            if any(not np.array_equal(e[k],a[k]) for k in e):raise ValueError('Historical native execution differs')
                    execute(p,case,'identity',baseline,baseline,e,b.actions)
                    rt,rtmeta=roundtrip_future(p.adapter.model,baseline)
                    np.savez_compressed(case/'roundtrip_input.npz',future=rt['future']);write(case/'roundtrip_metrics.json',rtmeta)
                    _,execution=execute(p,case,'roundtrip_baseline',rt['future'],baseline)
                    delta=trajectory_difference(e['eef_position'],execution['eef_position'])
                    gate={'state':state,'seed':seed,'observed_appearance':appearance,**delta,'pass':baseline_gate(delta,protocol['baseline_limits']),'identity_exact':True,'historical_native_exact':appearance=='original'}
                    gates.append(gate);write(case/'baseline_gate.json',gate);write(root/'baseline_gates.json',gates)
        verify(root);write(root/'codec_manifest.json',codecs)
        write(root/'codec_review.json',{'method':None,'human_review':'pending','reviewed_before_donors':False,'accepted':False,'clips':{k:None for k in codecs}})
        write(root/'status.json',{'status':'awaiting_codec_review' if all(g['pass'] for g in gates) else 'stopped_baseline_gate','branches':36,'at':now()})
    except Exception as exc:
        write(root/'status.json',{'status':'failed_controls','error':str(exc),'at':now()});raise
    finally:
        if p is not None:p.close()


def run(root):
    verify(root);protocol=read(root/'protocol.json')
    if read(root/'status.json')['status']!='awaiting_codec_review':raise ValueError('Baseline gates must all pass')
    review=read(root/'codec_review.json');manifest=read(root/'codec_manifest.json')
    if review['method'] not in ('AI_assisted','human') or not review['accepted'] or not review['reviewed_before_donors']:raise ValueError('Explicit visual review required')
    for name,files in manifest.items():
        if not isinstance(review['clips'].get(name),dict) or review['clips'][name].get('accepted') is not True or not review['clips'][name].get('notes'):raise ValueError(f'Missing review: {name}')
        for n,h in files.items():
            if digest_file(root/name/n)!=h:raise ValueError('Codec changed')
    write(root/'donor_freeze.json',{'at':now(),'review_sha256':digest_file(root/'codec_review.json'),'manifest_sha256':digest_file(root/'codec_manifest.json')})
    p=None;rows=[]
    try:
        write(root/'status.json',{'status':'donors_running','at':now()})
        for state in protocol['initial_states']:
            cfg={**read(protocol['pilot_config']),'initial_state_index':state,'action_seed':0}
            p=new_pilot(cfg,p);p.verify_replay();native=p.request;reference=reference_arrays(protocol,state)
            if provenance(p)!=read(root/f'state_{state}/common_context.json'):raise ValueError('Model/source context changed')
            meta=read(Path(protocol['references'])/f'state_{state}/soup/metadata.json')
            objects=reference['soup']['object_position'][0];axis=objects[meta['other_index']]-objects[meta['target_index']];axis[2]=0;axis=axis/np.linalg.norm(axis)
            for seed in protocol['action_seeds']:
                traces={}
                for appearance in protocol['observed_appearances']:
                    set_request(p,native,reference,appearance,seed);case=root/f'state_{state}/seed_{seed}'/appearance
                    ctx=read(case/'context.json')
                    if digest_array(p.request.observation.image)!=ctx['image_sha256'] or digest_array(p.request.observation.proprio)!=ctx['proprio_sha256']:raise ValueError('Recipient input changed')
                    baseline=load_future(case/'baseline/policy.npz')
                    for target in protocol['targets']:
                        for future_appearance in protocol['future_appearances']:
                            name=f'{target}_{future_appearance}'
                            donor=load_future(root/f'state_{state}/codecs'/appearance/name/'codec.npz')
                            _,e=execute(p,case,name,donor,baseline)
                            traces[(appearance,target,future_appearance)]=e['eef_position']
                row={'state':state,'seed':seed,'soup_to_tomato_axis':axis.tolist(),'motion_effects':{},'appearance_effects':{}}
                for appearance in protocol['observed_appearances']:
                    for future_appearance in protocol['future_appearances']:
                        a=traces[(appearance,'soup',future_appearance)];b=traces[(appearance,'tomato',future_appearance)]
                        row['motion_effects'][f'{appearance}/{future_appearance}']={**trajectory_difference(a,b),'target_axis_endpoint_m':float(np.dot(b[-1]-a[-1],axis))}
                    for target in protocol['targets']:
                        a=traces[(appearance,target,'original')];b=traces[(appearance,target,'tint_10')]
                        row['appearance_effects'][f'{appearance}/{target}']={**trajectory_difference(a,b),'target_axis_endpoint_m':float(np.dot(b[-1]-a[-1],axis))}
                rows.append(row);write(root/'comparisons.json',rows)
        verify(root);write(root/'status.json',{'status':'complete','branches':84,'at':now(),'human_review':'pending'})
    except Exception as exc:
        write(root/'status.json',{'status':'failed_donors','error':str(exc),'at':now()});raise
    finally:
        if p is not None:p.close()


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('stage',choices=['controls','run']);ap.add_argument('--protocol',type=Path,default=Path('configs/experiments/e1_simulator_policy.json'));ap.add_argument('--output',type=Path,required=True)
    args=ap.parse_args()
    if args.stage=='controls':controls(args.output,read(args.protocol))
    else:run(args.output)
