"""Bounded seven-arm visual-bottleneck pilot with baseline-first gating."""
import argparse
import json
from pathlib import Path
import shutil
import zipfile
import numpy as np

from e1_replication import now,write,new_pilot,load_future
from e1_equal_norm import check_context
from wam_interp.experiments.libero_pilot import digest_file,provenance
from wam_interp.experiments.video_bottleneck import roundtrip_future,trajectory_difference,baseline_gate
from wam_interp.experiments.e1 import _validate_branch


def read(p):return json.loads(Path(p).read_text())


def prepare(root,protocol):
    if root.exists():raise FileExistsError('Fresh output required')
    source=Path(protocol['source_results']);old=read(source/'freeze.json')
    selected={}
    for state in protocol['initial_states']:
        for seed in protocol['action_seeds']:
            case=f'state_{state}/seed_{seed}'
            for name in ('baseline.npz','same.npz','alternative.npz','previous_context.json','previous_baseline.npz','previous_execution.npz'):
                rel=f'{case}/inputs/{name}';p=source/rel
                if digest_file(p)!=old['inputs'][rel]['sha256']:raise ValueError(f'Old input changed: {p}')
                selected[rel]=p
    root.mkdir(parents=True);write(root/'protocol.json',protocol)
    frozen={'at':now(),'inputs':{},'external_inputs':{},'protocol_sha256':digest_file(root/'protocol.json')}
    for rel,p in selected.items():
        if p.name=='previous_execution.npz':
            frozen['external_inputs'][rel]={'path':str(p),'sha256':digest_file(p)};continue
        out=root/rel;out.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,out)
        frozen['inputs'][rel]={'source':str(p),'sha256':digest_file(out)}
    with zipfile.ZipFile(root/'source.zip','w',zipfile.ZIP_DEFLATED) as z:
        for folder in ('src','scripts','configs'):
            for p in sorted(Path(folder).rglob('*')):
                if p.is_file() and p.suffix in ('.py','.json','.yaml','.sh'):z.write(p,p.as_posix())
        for n in ('__init__.py','pyproject.toml'):z.write(n,n)
    frozen['source_sha256']=digest_file(root/'source.zip');write(root/'freeze.json',frozen)
    write(root/'status.json',{'status':'prepared','at':now()})
    print(json.dumps({'prepared':True,'maximum_branches':protocol['maximum_branches']}),flush=True)


def verify(root):
    f=read(root/'freeze.json')
    if digest_file(root/'protocol.json')!=f['protocol_sha256'] or digest_file(root/'source.zip')!=f['source_sha256']:
        raise ValueError('Protocol/source archive changed')
    for name,r in f['inputs'].items():
        if digest_file(root/name)!=r['sha256']:raise ValueError(f'Input changed: {name}')
    for name,r in f['external_inputs'].items():
        if digest_file(r['path'])!=r['sha256']:raise ValueError(f'Historical input changed: {name}')
    with zipfile.ZipFile(root/'source.zip') as z:
        for name in z.namelist():
            if Path(name).read_bytes()!=z.read(name):raise ValueError(f'Source changed: {name}')


def images(rgb,folder):
    from PIL import Image,ImageDraw
    folder.mkdir()
    sheet=Image.new('RGB',(1344,750),'white');draw=ImageDraw.Draw(sheet)
    if rgb.shape!=(1,3,9,224,448):raise ValueError('Unexpected paired camera RGB shape')
    for i in range(9):
        frame=Image.fromarray(np.clip((rgb[0,:,i].transpose(1,2,0)+1)*127.5,0,255).astype(np.uint8))
        frame.save(folder/f'{i:03}.png');x,y=(i%3)*448,(i//3)*250
        draw.text((x+3,y+4),f'{folder.parent.parent.name} {folder.parent.name} {folder.name} frame {i}',fill='black')
        sheet.paste(frame,(x,y+25))
    sheet.save(folder.parent/f'{folder.name}_sheet.jpg',quality=97)


def execute(p,case,name,donor,baseline,baseline_execution=None,baseline_actions=None):
    import torch
    p.restore(p.saved_rng);torch.cuda.reset_peak_memory_stats()
    b=p.session.run(donor);_validate_branch(b)
    e={k:v.copy() for k,v in p.last_rollout.items()}
    if b.actions.shape!=(32,7) or len(b.executed_commands)!=32:raise ValueError('Expected a complete 32-control chunk')
    if not np.array_equal(b.future[:,:,:1],baseline[:,:,:1]):raise ValueError('Observed anchor changed')
    if donor is not None:
        expected=torch.as_tensor(donor).to(torch.bfloat16).float().numpy()
        if not np.array_equal(b.future[:,:,1:],expected[:,:,1:]):raise ValueError('Incorrect future insertion')
    if name=='identity':
        if not np.array_equal(b.actions,baseline_actions) or any(not np.array_equal(v,e[k]) for k,v in baseline_execution.items()):
            raise ValueError('Identity failed')
    folder=case/name;folder.mkdir()
    np.savez_compressed(folder/'policy.npz',future=b.future,actions=b.actions,commands=b.commands,executed_commands=b.executed_commands)
    np.savez_compressed(folder/'execution.npz',**e)
    np.savez_compressed(folder/'kinematics.npz',**{k:v for k,v in e.items() if k not in ('agentview','wrist')})
    record={'state':p.config['initial_state_index'],'seed':p.config['action_seed'],'arm':name,'at':now(),
        'metadata':b.metadata,'peak_gpu_allocated_bytes':torch.cuda.max_memory_allocated(),
        'recipient_instruction':p.task.language,'anchor_exact':True,'insertion_exact':True,
        'policy_sha256':digest_file(folder/'policy.npz'),'execution_sha256':digest_file(folder/'execution.npz')}
    write(folder/'record.json',record)
    print(json.dumps({'state':record['state'],'seed':record['seed'],'arm':name,'complete':True}),flush=True)
    return b,e


def make_pilot(root,protocol,state,seed,previous_pilot):
    case=root/f'state_{state}'/f'seed_{seed}';previous=read(case/'inputs/previous_context.json')
    config=read(protocol['pilot_config'])
    p=new_pilot({**config,'initial_state_index':state,'action_seed':seed,
        'candidate_seeds':previous['protocol']['donor_generation']['seeds']},previous_pilot)
    p.verify_replay();current=provenance(p);check_context(previous,current)
    if (case/'context.json').exists():
        if current!=read(case/'context.json'):raise ValueError('Control/donor context changed')
    else:write(case/'context.json',current)
    return p,case


def controls(root):
    verify(root)
    if read(root/'status.json')['status']!='prepared':raise ValueError('No silent restart')
    protocol=read(root/'protocol.json');frozen=read(root/'freeze.json');p=None;gates=[];codec_manifest={}
    try:
        write(root/'status.json',{'status':'controls_running','at':now()})
        for state in protocol['initial_states']:
            for seed in protocol['action_seeds']:
                p,case=make_pilot(root,protocol,state,seed,p);bank=case/'inputs'
                baseline=load_future(bank/'baseline.npz')
                # Decode/encode the fixed inputs before observing this case's behavior.
                for name in ('baseline','same','alternative'):
                    folder=case/f'codec_{name}';folder.mkdir()
                    arrays,metrics=roundtrip_future(p.adapter.model,load_future(bank/f'{name}.npz'))
                    np.savez_compressed(folder/'codec.npz',**arrays);write(folder/'metrics.json',metrics)
                    images(arrays['decoded_rgb'],folder/'original');images(arrays['roundtrip_rgb'],folder/'roundtrip')
                    for n in ('codec.npz','metrics.json','original_sheet.jpg','roundtrip_sheet.jpg'):
                        codec_manifest[(folder/n).relative_to(root).as_posix()]=digest_file(folder/n)
                    print(json.dumps({'state':state,'seed':seed,'codec':name,'complete':True}),flush=True)
                b,e=execute(p,case,'baseline',None,baseline)
                if not np.array_equal(b.future,baseline):raise ValueError('Historical baseline future differs')
                with np.load(bank/'previous_baseline.npz',allow_pickle=False) as d:
                    if not np.array_equal(d['actions'],b.actions):raise ValueError('Historical baseline action differs')
                key=f'state_{state}/seed_{seed}/inputs/previous_execution.npz'
                with np.load(frozen['external_inputs'][key]['path'],allow_pickle=False) as d:
                    if set(d.files)!=set(e) or any(not np.array_equal(d[k],e[k]) for k in d.files):
                        raise ValueError('Historical baseline execution differs')
                execute(p,case,'identity',baseline,baseline,e,b.actions)
                _,rt=execute(p,case,'roundtrip_baseline',load_future(case/'codec_baseline/codec.npz'),baseline)
                delta=trajectory_difference(e['eef_position'],rt['eef_position'])
                gate={'state':state,'seed':seed,**delta,'pass':baseline_gate(delta,protocol['baseline_feasibility_limits']),
                    'historical_baseline_exact':True,'identity_all_arrays_exact':True}
                gates.append(gate);write(case/'baseline_gate.json',gate);write(root/'baseline_gates.json',gates)
        verify(root);write(root/'codec_manifest.json',codec_manifest)
        write(root/'codec_review.json',{'method':None,'human_review':'pending','reviewed_before_donor_execution':False,
            'all_cases_reviewed':False,'accepted':False,'notes':'','cases':{f"state_{s}/seed_{n}":None for s in protocol['initial_states'] for n in protocol['action_seeds']}})
        status='awaiting_codec_review' if all(g['pass'] for g in gates) else 'stopped_baseline_gate'
        write(root/'status.json',{'status':status,'branches':18,'at':now(),'failed_gates':sum(not g['pass'] for g in gates)})
    except Exception as exc:
        write(root/'status.json',{'status':'failed_controls','error':str(exc),'at':now()});raise
    finally:
        if p is not None:p.close()


def donors(root):
    verify(root)
    if read(root/'status.json')['status']!='awaiting_codec_review':raise ValueError('Baseline gates must all pass')
    review=read(root/'codec_review.json')
    if review['method'] not in ('AI_assisted','human') or not all(review[k] is True for k in ('reviewed_before_donor_execution','all_cases_reviewed','accepted')):
        raise ValueError('Explicit visual codec review required')
    protocol=read(root/'protocol.json')
    for s in protocol['initial_states']:
        for n in protocol['action_seeds']:
            r=review['cases'][f'state_{s}/seed_{n}']
            if not isinstance(r,dict) or r.get('accepted') is not True or not r.get('notes'):raise ValueError('Incomplete case review')
    for rel,digest in read(root/'codec_manifest.json').items():
        if digest_file(root/rel)!=digest:raise ValueError('Codec artifacts changed')
    write(root/'donor_execution_freeze.json',{'at':now(),'review_sha256':digest_file(root/'codec_review.json'),
        'codec_manifest_sha256':digest_file(root/'codec_manifest.json')})
    p=None;rows=[]
    try:
        write(root/'status.json',{'status':'donors_running','branches':18,'at':now()})
        for state in protocol['initial_states']:
            for seed in protocol['action_seeds']:
                p,case=make_pilot(root,protocol,state,seed,p);baseline=load_future(case/'inputs/baseline.npz');traces={}
                for name,path in [('raw_same','inputs/same.npz'),('raw_alternative','inputs/alternative.npz'),
                    ('roundtrip_same','codec_same/codec.npz'),('roundtrip_alternative','codec_alternative/codec.npz')]:
                    b,e=execute(p,case,name,load_future(case/path),baseline);traces[name]=e['eef_position']
                    if name.startswith('raw_'):
                        old=Path(protocol['source_results'])/f'state_{state}'/f'seed_{seed}'/name.replace('raw_','native_')/'policy.npz'
                        with np.load(old,allow_pickle=False) as d:
                            if not np.array_equal(d['actions'],b.actions):raise ValueError('Historical native donor action changed')
                result={'state':state,'seed':seed,
                    'raw':trajectory_difference(traces['raw_same'],traces['raw_alternative']),
                    'roundtrip':trajectory_difference(traces['roundtrip_same'],traces['roundtrip_alternative'])}
                result['change_in_signed_contrast_m']=result['roundtrip']['signed_world_y_m']-result['raw']['signed_world_y_m']
                rows.append(result);write(case/'comparison.json',result);write(root/'comparisons.json',rows)
        verify(root);write(root/'status.json',{'status':'complete','branches':42,'at':now(),'human_review':'pending'})
    except Exception as exc:
        write(root/'status.json',{'status':'failed_donors','error':str(exc),'at':now()});raise
    finally:
        if p is not None:p.close()


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('stage',choices=['prepare','controls','donors'])
    ap.add_argument('--protocol',type=Path,default=Path('configs/experiments/e1_video_bottleneck.json'));ap.add_argument('--output',type=Path,required=True)
    args=ap.parse_args()
    if args.stage=='prepare':prepare(args.output,read(args.protocol))
    elif args.stage=='controls':controls(args.output)
    else:donors(args.output)
