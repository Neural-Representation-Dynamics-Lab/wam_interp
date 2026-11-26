"""Acquire native donor banks, freeze reviewed matching, then run only eligible pairs."""
import argparse
from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
import shutil
import zipfile

import numpy as np

import e1_libero as pilot_cli
from e1_replication import new_pilot,load_future,write,now,physical_state
from wam_interp.experiments.e1 import context_id
from wam_interp.experiments.fastwam_session import _array
from wam_interp.experiments.libero_pilot import provenance,digest_file,save_frames
from wam_interp.experiments.matching import latent_delta_norm
from wam_interp.experiments.native_pairs import rank_pairs,select_pair,target_direction


def config_for(protocol):
    config=json.loads(Path(protocol['pilot_config']).read_text())
    return {**config,'candidate_seeds':protocol['candidate_seeds'],'donor_prompts':protocol['donor_prompts'],
            **({'replay_prefix':protocol['replay_prefix']} if 'replay_prefix' in protocol else {})}


def snapshot_source(root):
    with zipfile.ZipFile(root/'source.zip','w',zipfile.ZIP_DEFLATED) as z:
        for directory in ('src','configs','scripts'):
            for p in sorted(Path(directory).rglob('*')):
                if p.is_file() and p.suffix in ('.py','.json','.yaml','.sh'):z.write(p,p.as_posix())
        for name in ('__init__.py','pyproject.toml'):z.write(name,name)
    write(root/'protocol_freeze.json',{'at':now(),'protocol_sha256':digest_file(root/'protocol.json'),
        'source_sha256':digest_file(root/'source.zip'),'no_recipient_effects_observed':True})


def verify_source(root):
    freeze=json.loads((root/'protocol_freeze.json').read_text())
    if digest_file(root/'protocol.json')!=freeze['protocol_sha256'] or digest_file(root/'source.zip')!=freeze['source_sha256']:
        raise ValueError('Protocol or source archive changed')
    with zipfile.ZipFile(root/'source.zip') as z:
        for name in z.namelist():
            if Path(name).read_bytes()!=z.read(name):raise ValueError(f'Inference source changed: {name}')


def make_sheets(case,shortlist,index):
    from PIL import Image,ImageDraw
    bank=case/'bank';aliases={c['id']:f'c{i:03}' for i,c in enumerate(index)}
    write(case/'candidate_aliases.json',aliases)
    wanted=sorted(set(n for p in shortlist for n in (p['same'],p['alternative'])))
    for name in wanted:
        sheet=Image.new('RGB',(1344,750),'white');d=ImageDraw.Draw(sheet)
        for i in range(9):
            x,y=(i%3)*448,(i//3)*250
            d.text((x+3,y+4),f"{case.name} {aliases[name]} control {4*i}: external | wrist",fill='black')
            sheet.paste(Image.open(bank/f'{name}_frames/{i:03}.png'),(x,y+25))
        sheet.save(case/f'{aliases[name]}_full.jpg',quality=97)
    for page in range((len(index)+7)//8):
        sheet=Image.new('RGB',(672,1160),'white');d=ImageDraw.Draw(sheet)
        for row,c in enumerate(index[page*8:(page+1)*8]):
            d.text((4,row*145+4),f"{case.name} {aliases[c['id']]}; controls 0,16,32",fill='black')
            for col,i in enumerate((0,4,8)):
                sheet.paste(Image.open(bank/f"{c['id']}_frames/{i:03}.png").resize((224,112)),(col*224,row*145+25))
        sheet.save(case/f'overview_{page}.jpg',quality=96)


def prepare(root,protocol):
    root.mkdir(parents=True,exist_ok=False)
    write(root/'protocol.json',protocol);snapshot_source(root)
    config=config_for(protocol);p=None
    try:
        for state in protocol['initial_states']:
            case=root/f'state_{state}';case.mkdir()
            write(root/'prepare_status.json',{'status':'running','state':state,'at':now()})
            p=new_pilot({**config,'initial_state_index':state,'action_seed':protocol['action_seeds'][0]},p)
            positions,_=physical_state(p.env,protocol['target_bodies'])
            direction=target_direction(*(positions[n] for n in protocol['target_bodies']))
            write(case/'target_geometry.json',{'positions':{k:v.tolist() for k,v in positions.items()},
                'world_xy_direction':direction.tolist(),'chosen_before_recipient_effects':True})
            bank=case/'bank';pilot_cli.prepare(p,bank)
            context=json.loads((bank/'context.json').read_text())
            index=json.loads((bank/'candidate_index.json').read_text())
            recipients={};norms={}
            for seed in protocol['action_seeds']:
                path=bank/f'baseline_seed_{seed}.npz'
                if seed==p.config['action_seed']:
                    shutil.copyfile(bank/'baseline.npz',path)
                else:
                    p.restore(p.saved_rng)
                    native=p.adapter.predict_action(replace(p.request,seed=seed))
                    observed,captured,latency=p.infer(seed=seed)
                    identity,_,_=p.infer(seed=seed,donor=captured['used'])
                    errors={'observer':float(np.max(np.abs(_array(native.actions)-_array(observed.actions)))),
                        'identity':float(np.max(np.abs(_array(identity.actions)-_array(observed.actions))))}
                    if any(not np.isfinite(v) or v>config['identity_atol'] for v in errors.values()):raise ValueError('Seed preflight failed')
                    np.savez_compressed(path,future=_array(captured['used']))
                    write(bank/f'preflight_seed_{seed}.json',{'errors':errors,'inference_with_capture_seconds':latency})
                c=deepcopy(context);c['action_seed']=seed;c['request_options']['seed']=seed
                recipients[str(seed)]={'context':c,'baseline_sha256':digest_file(path)}
                baseline=load_future(path)
                norms[str(seed)]={x['id']:latent_delta_norm(load_future(bank/f"{x['id']}.npz"),baseline) for x in index}
            targets={c['id']:protocol['intended_targets'][c['id'].rsplit('_seed_',1)[0]] for c in index}
            ranking=rank_pairs(targets,norms,protocol['matching'],protocol['maximum_review_pairs_per_state'],labels=protocol.get('contrast_labels',('soup','cream')))
            write(case/'recipient_contexts.json',recipients);write(case/'norms.json',norms);write(case/'ranking.json',ranking)
            wanted=sorted(set(n for pair in ranking['shortlist'] for n in (pair['same'],pair['alternative'])))
            write(case/'review.json',{'review_method':None,'human_review':'pending','frame_mapping_verified':False,
                'candidates':{name:{'all_nine_frames_reviewed':False,'label':None,'acceptable':False,'notes':'',
                    'landmarks_pixels':None,'landmarks_verified':False} for name in wanted},'pair_nuisance':{}})
            make_sheets(case,ranking['shortlist'],index)
            write(case/'acquisition.json',{'completed_at':now(),'candidates':len(index),'no_donor_actions_executed':True,
                'ranking_sha256':digest_file(case/'ranking.json'),'recipient_contexts_sha256':digest_file(case/'recipient_contexts.json'),
                'norms_sha256':digest_file(case/'norms.json'),'target_geometry_sha256':digest_file(case/'target_geometry.json')})
            print(json.dumps({'state':state,'candidates':len(index),'norm_eligible_pairs':ranking['norm_eligible_pairs'],
                'shortlisted_pairs':len(ranking['shortlist']),'candidates_to_review':len(wanted)}),flush=True)
        verify_source(root)
        write(root/'prepare_status.json',{'status':'complete','at':now(),'human_review':'pending'})
        export_review(root)
    except Exception as exc:
        write(root/'prepare_status.json',{'status':'failed','at':now(),'error':str(exc)});raise
    finally:
        if p is not None:p.close()


def export_review(root):
    with zipfile.ZipFile(root/'review_bundle.zip','w',zipfile.ZIP_DEFLATED) as z:
        for p in sorted(root.rglob('*')):
            if not p.is_file() or 'bank' in p.relative_to(root).parts:continue
            if p.suffix in ('.json','.jpg'):z.write(p,p.relative_to(root))


def reviewed_state(root,protocol,state):
    case=root/f'state_{state}';bank=case/'bank'
    acquisition=json.loads((case/'acquisition.json').read_text())
    for name in ('ranking','recipient_contexts','norms','target_geometry'):
        if digest_file(case/f'{name}.json')!=acquisition[f'{name}_sha256']:raise ValueError(f'Frozen {name} changed')
    index=json.loads((bank/'candidate_index.json').read_text())
    for c in index:
        if digest_file(bank/f"{c['id']}.npz")!=c['sha256']:raise ValueError('Native donor changed')
    contexts=json.loads((case/'recipient_contexts.json').read_text());norms={}
    for seed,c in contexts.items():
        path=bank/f'baseline_seed_{seed}.npz'
        if digest_file(path)!=c['baseline_sha256']:raise ValueError('Recipient baseline changed')
        baseline=load_future(path)
        norms[seed]={x['id']:latent_delta_norm(load_future(bank/f"{x['id']}.npz"),baseline) for x in index}
    targets={c['id']:protocol['intended_targets'][c['id'].rsplit('_seed_',1)[0]] for c in index}
    ranking=rank_pairs(targets,norms,protocol['matching'],protocol['maximum_review_pairs_per_state'],labels=protocol.get('contrast_labels',('soup','cream')))
    if ranking!=json.loads((case/'ranking.json').read_text()):raise ValueError('Ranking no longer agrees with raw latents')
    review=json.loads((case/'review.json').read_text())
    if ranking['shortlist']:
        if review.get('frame_mapping_verified') is not True or review.get('review_method') not in ('AI_assisted','human'):
            raise ValueError('Explicit review and frame-mapping verification required')
    selected=select_pair(ranking,review['candidates'],norms,protocol['matching'],review['pair_nuisance'],labels=protocol.get('contrast_labels',('soup','cream')))
    selected.update(review_sha256=digest_file(case/'review.json'),human_review=review.get('human_review','pending'))
    return selected,contexts,index,review


def select(root):
    verify_source(root)
    if (root/'selection_freeze.json').exists():raise FileExistsError('Selections already frozen')
    if json.loads((root/'prepare_status.json').read_text())['status']!='complete':raise ValueError('Acquisition incomplete')
    protocol=json.loads((root/'protocol.json').read_text());records={}
    for state in protocol['initial_states']:
        selected,contexts,index,review=reviewed_state(root,protocol,state)
        records[str(state)]=selected
    # Every state is validated before creating execution inputs or observing effects.
    write(root/'selection_freeze.json',{'at':now(),'states':records,'human_review':'pending','no_recipient_effects_observed':True})
    for state in protocol['initial_states']:
        pair=records[str(state)]['selected']
        if pair is None:continue
        case=root/f'state_{state}';bank=case/'bank'
        _,contexts,index,review=reviewed_state(root,protocol,state)
        selected_ids=[pair['same'],pair['alternative']]
        for seed in protocol['action_seeds']:
            out=case/f'seed_{seed}_prepared';out.mkdir()
            for name in selected_ids:shutil.copyfile(bank/f'{name}.npz',out/f'{name}.npz')
            shutil.copyfile(bank/f'baseline_seed_{seed}.npz',out/'baseline.npz')
            write(out/'context.json',contexts[str(seed)]['context'])
            write(out/'baseline_identity.json',{'sha256':digest_file(out/'baseline.npz'),'frames':9})
            chosen=[c for c in index if c['id'] in selected_ids]
            write(out/'candidate_index.json',chosen)
            write(out/'status.json',{'status':'awaiting_semantic_review','synthetic':False})
            direction=json.loads((case/'target_geometry.json').read_text())['world_xy_direction']
            write(out/'review.json',{'reviewed_before_execution':True,'frame_mapping_verified':True,
                'context_id':context_id(contexts[str(seed)]['context']),'property':'first_approach_target',
                'recipient_label':'soup','expected_sign':1 if direction[1]>=0 else -1,
                'same_property':pair['same'],'opposite_property':pair['alternative'],'frame_steps':list(range(0,33,4)),
                'candidates':[{**c,'label':review['candidates'][c['id']]['label'],'acceptable':True,
                    'nuisance_notes':review['candidates'][c['id']]['notes']} for c in chosen]})
    print(json.dumps({'selected_states':[s for s,r in records.items() if r['selected'] is not None],
        'no_match_states':[s for s,r in records.items() if r['selected'] is None]}),flush=True)


def run(root):
    verify_source(root)
    if (root/'run_status.json').exists():raise FileExistsError('No silent restart or overwrite')
    protocol=json.loads((root/'protocol.json').read_text())
    frozen=json.loads((root/'selection_freeze.json').read_text())
    for state in protocol['initial_states']:
        current,_,_,_=reviewed_state(root,protocol,state)
        if current!=frozen['states'][str(state)]:raise ValueError('Review changed after selection freeze')
    config=config_for(protocol);p=None;results=[]
    try:
        write(root/'run_status.json',{'status':'running','at':now()})
        for state in protocol['initial_states']:
            if frozen['states'][str(state)]['selected'] is None:continue
            case=root/f'state_{state}'
            direction=np.array(json.loads((case/'target_geometry.json').read_text())['world_xy_direction'])
            for seed in protocol['action_seeds']:
                p=new_pilot({**config,'initial_state_index':state,'action_seed':seed},p)
                out=case/f'seed_{seed}_run'
                pilot_cli.run(p,case/f'seed_{seed}_prepared',out)
                with np.load(out/'same_property_execution.npz') as a:same=a['eef_position'].copy()
                with np.load(out/'opposite_property_execution.npz') as a:alt=a['eef_position'].copy()
                if same.shape!=alt.shape:raise ValueError('Different termination lengths require a revised analysis')
                record={'state':state,'seed':seed,'target_direction':direction.tolist(),
                    'primary_target_projection_m':float(np.dot(alt[-1]-same[-1],direction)),
                    'world_y_contrast_m':float(alt[-1,1]-same[-1,1]),
                    'endpoint_separation_m':float(np.linalg.norm(alt[-1]-same[-1])),
                    'trajectory_separation_m':float(np.linalg.norm(alt-same,axis=1).mean())}
                write(out/'native_pair_comparison.json',record);results.append(record)
                write(root/'run_results.json',results);print(json.dumps(record),flush=True)
        verify_source(root)
        write(root/'run_status.json',{'status':'complete','at':now(),'cases':len(results),'branches':4*len(results),
            'human_review':'pending','no_unmatched_policy_comparisons':True})
    except Exception as exc:
        write(root/'run_status.json',{'status':'failed','error':str(exc),'at':now()});raise
    finally:
        if p is not None:p.close()


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('stage',choices=['prepare','select','run','export-review'])
    ap.add_argument('--protocol',type=Path,default=Path('configs/experiments/e1_native_search.json'))
    ap.add_argument('--output',type=Path,required=True)
    args=ap.parse_args()
    if args.stage=='prepare':prepare(args.output,json.loads(args.protocol.read_text()))
    elif args.stage=='select':select(args.output)
    elif args.stage=='export-review':export_review(args.output)
    else:run(args.output)
