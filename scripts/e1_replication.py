"""Prepare fixed exploratory replication banks, run reviewed arms, or record contact rollouts.

Stages are separate: preparation never executes donor actions. Reuse the loaded
model across states but recreate simulator/controllers for every paired branch.
"""
import argparse
from copy import deepcopy
from dataclasses import replace
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np

import e1_libero as pilot_cli
from wam_interp.experiments.e1 import context_id
from wam_interp.experiments.fastwam_session import _array
from wam_interp.experiments.futures import predict_with_future
from wam_interp.experiments.libero_pilot import LiberoPilot, load_pilot, provenance, save_frames, digest_file
from wam_interp.experiments.matching import latent_delta_norm, motion_features, choose_match


def write(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(data, indent=2, allow_nan=False), encoding='utf-8')
    tmp.replace(path)


def now():
    return datetime.now(timezone.utc).isoformat()


def new_pilot(config, loaded=None):
    if loaded is None:
        return load_pilot(config)
    from wam_interp.evaluations.LIBERO.runner import _get_suite
    from wam_interp.evaluations.LIBERO.environment import initial_states_for_task
    loaded.close()
    suite = _get_suite(config['suite'])
    p = LiberoPilot(config, loaded.adapter, task=suite.get_task(config['task_id']),
        initial_state=initial_states_for_task(suite, config['task_id'])[config['initial_state_index']],
        create_env=loaded.create_env, reset_env=loaded.reset_env, step_env=loaded.step_env)
    p.resolved_model_config = loaded.resolved_model_config
    return p


def load_future(path):
    with np.load(path, allow_pickle=False) as a:
        return a['future'].copy()


def preview_bank(bank):
    from PIL import Image, ImageDraw
    index = json.loads((bank/'candidate_index.json').read_text())
    for page in range((len(index)+7)//8):
        canvas=Image.new('RGB',(672,8*145),'white');d=ImageDraw.Draw(canvas)
        for row,c in enumerate(index[page*8:(page+1)*8]):
            d.text((4,row*145+3),c['id'],fill='black')
            for col,i in enumerate((0,4,8)):
                im=Image.open(bank/f'{c["id"]}_frames/{i:03d}.png').resize((224,112))
                canvas.paste(im,(col*224,row*145+25))
        canvas.save(bank/f'candidate_sheet_{page}.jpg',quality=94)


def prepare_banks(root, protocol, config):
    p=None
    try:
        for state in protocol['initial_states']:
            case=root/f'state_{state}';bank=case/'bank'
            cfg={**config,'initial_state_index':state,'action_seed':protocol['action_seeds'][0]}
            write(root/'prepare_status.json',{'status':'running','state':state,'updated_at':now()})
            p=new_pilot(cfg,p)
            if not bank.exists():
                pilot_cli.prepare(p,bank)
            elif json.loads((bank/'status.json').read_text())['status']!='awaiting_semantic_review':
                raise RuntimeError(f'Incomplete bank requires audit before resuming: {bank}')
            base_context=provenance(p)
            if json.loads((bank/'context.json').read_text())!=base_context:
                raise RuntimeError('Prepared context/source changed; do not silently reuse candidates')
            index=json.loads((bank/'candidate_index.json').read_text())
            contexts={}
            for seed in protocol['action_seeds']:
                destination=bank/f'baseline_seed_{seed}.npz'
                if not destination.exists():
                    if seed==cfg['action_seed']:
                        shutil.copyfile(bank/'baseline.npz',destination)
                        shutil.copytree(bank/'baseline_frames',bank/f'baseline_seed_{seed}_frames')
                    else:
                        p.restore(p.saved_rng)
                        native=p.adapter.predict_action(replace(p.request,seed=seed))
                        observed,captured,latency=p.infer(seed=seed)
                        identity,_,_=p.infer(seed=seed,donor=captured['used'])
                        errors={'observer':float(np.max(np.abs(_array(native.actions)-_array(observed.actions)))),
                                'identity':float(np.max(np.abs(_array(native.actions)-_array(identity.actions))))}
                        if any(not np.isfinite(v) or v>cfg['identity_atol'] for v in errors.values()):
                            raise RuntimeError(f'Seed {seed} preflight failed: {errors}')
                        np.savez_compressed(destination,future=_array(captured['used']))
                        save_frames(p.adapter.model,captured['used'],bank/f'baseline_seed_{seed}_frames')
                        write(bank/f'preflight_seed_{seed}.json',{'errors':errors,'latency_seconds':latency})
                context=deepcopy(base_context)
                context['action_seed']=seed;context['request_options']['seed']=seed
                baseline=load_future(destination)
                contexts[str(seed)]={'context':context,'baseline_sha256':digest_file(destination),
                    'candidate_latent_delta_norms':{c['id']:latent_delta_norm(load_future(bank/f'{c["id"]}.npz'),baseline) for c in index}}
            write(bank/'recipient_contexts.json',contexts)
            preview_bank(bank)
            review_path=case/'selection_review.json'
            if not review_path.exists():
                write(review_path,{'status':'awaiting_AI_assisted_review','human_review':'pending',
                    'reviewed_before_execution':False,'frame_mapping_verified':False,
                    'recipient_label':protocol['recipient_label'],'expected_sign':protocol['expected_sign'],
                    'primary_same':protocol['same_primary'],'primary_alternative':protocol['alternative_primary'],
                    'candidates':{c['id']:{'label':None,'acceptable':False,'nuisance_notes':'',
                        'review_complete':False,'landmarks_pixels':None,'landmarks_visually_verified':False} for c in index}})
            write(case/'acquisition_complete.json',{'completed_at':now(),'candidates':len(index),
                'source_sha256':base_context['source_sha256'],'no_donor_actions_executed':True})
            print(json.dumps({'state':state,'acquisition':'complete','candidates':len(index)}),flush=True)
        write(root/'prepare_status.json',{'status':'complete','states':protocol['initial_states'],'updated_at':now()})
    except Exception as exc:
        write(root/'prepare_status.json',{'status':'failed','error':str(exc),'updated_at':now()})
        raise
    finally:
        if p is not None:p.close()


def reviewed_selection(case, protocol, seed):
    """Freeze selection from bank imagery/norms, before any branch execution."""
    bank=case/'bank';review=json.loads((case/'selection_review.json').read_text())
    if review.get('reviewed_before_execution') is not True or review.get('frame_mapping_verified') is not True:
        raise ValueError('Explicit semantic/frame review required before execution')
    if review['primary_same']!=protocol['same_primary'] or review['primary_alternative']!=protocol['alternative_primary']:
        raise ValueError('Primary pair changed after protocol freeze')
    index={c['id']:c for c in json.loads((bank/'candidate_index.json').read_text())}
    contexts=json.loads((bank/'recipient_contexts.json').read_text());recipient=contexts[str(seed)]
    for name,c in index.items():
        if digest_file(bank/f'{name}.npz')!=c['sha256']:raise ValueError('Candidate hash changed')
    if digest_file(bank/f'baseline_seed_{seed}.npz')!=recipient['baseline_sha256']:
        raise ValueError('Recipient baseline changed')
    same,alt=protocol['same_primary'],protocol['alternative_primary']
    for name,label in [(same,protocol['recipient_label']),(alt,protocol['alternative_label'])]:
        c=review['candidates'][name]
        if not c.get('review_complete') or not c['acceptable'] or c['label']!=label or not c['nuisance_notes']:
            raise ValueError('Fixed primary donor lacks valid semantic review; record rejection')
    norms=recipient['candidate_latent_delta_norms'];rules=protocol['matching']
    # Recompute norms; stored descriptors cannot be edited to pass matching.
    baseline=load_future(bank/f'baseline_seed_{seed}.npz')
    norms={name:latent_delta_norm(load_future(bank/f'{name}.npz'),baseline) for name in index}
    candidates={};rejections={}
    for name,c in review['candidates'].items():
        if name not in index:raise ValueError('Unknown reviewed donor')
        if name in (same,alt):
            rejections[name]='primary donor, not an additional same-target control';continue
        if norms[alt]<=0:raise ValueError('Alternative donor has zero latent difference')
        ratio=abs(norms[name]-norms[alt])/norms[alt]
        if ratio>rules['latent_delta_relative_tolerance']:
            rejections[name]={'reason':'latent_norm_outside_tolerance','relative_difference':ratio};continue
        if c.get('review_complete') is not True or not c.get('nuisance_notes'):
            raise ValueError(f'Norm-eligible candidate {name} has not been semantically reviewed')
        if not c.get('acceptable') or c.get('label')!=protocol['recipient_label']:
            rejections[name]='reviewed, but not an accepted same-target donor';continue
        if c.get('landmarks_visually_verified') is not True or not c.get('nuisance_notes'):
            raise ValueError(f'Norm-eligible donor {name} still needs reviewed motion landmarks')
        candidates[name]={**motion_features(c['landmarks_pixels']),'latent_delta_norm':norms[name]}
    if candidates:
        c=review['candidates'][alt]
        if c.get('landmarks_visually_verified') is not True:raise ValueError('Alternative motion landmarks need review')
        alternative={**motion_features(c['landmarks_pixels']),'latent_delta_norm':norms[alt]}
        matching=choose_match(candidates,alternative,rules,exclude=(same,))
    else:
        matching={'selected':None,'outcome':'no_norm_eligible_reviewed_native_donor','comparisons':{}}
    matching.update(rejections=rejections,latent_delta_norms=norms,human_review='pending',
                    rules=rules,review_sha256=digest_file(case/'selection_review.json'))
    return review,recipient,matching


def prepare_recipient(case, protocol, seed):
    review,recipient,matching=reviewed_selection(case,protocol,seed)
    folder=case/f'seed_{seed}_prepared'
    if folder.exists():raise FileExistsError(folder)
    folder.mkdir()
    bank=case/'bank';index=json.loads((bank/'candidate_index.json').read_text())
    for c in index:shutil.copyfile(bank/f'{c["id"]}.npz',folder/f'{c["id"]}.npz')
    shutil.copyfile(bank/f'baseline_seed_{seed}.npz',folder/'baseline.npz')
    write(folder/'context.json',recipient['context'])
    write(folder/'baseline_identity.json',{'sha256':digest_file(folder/'baseline.npz'),'frames':9})
    write(folder/'candidate_index.json',index)
    write(folder/'status.json',{'status':'awaiting_semantic_review','synthetic':False})
    write(folder/'matching_report.json',matching)
    write(folder/'review.json',{'reviewed_before_execution':True,'frame_mapping_verified':True,
        'context_id':context_id(recipient['context']),'property':'first_approach_target',
        'recipient_label':protocol['recipient_label'],'expected_sign':protocol['expected_sign'],
        'same_property':protocol['same_primary'],'opposite_property':protocol['alternative_primary'],
        'matched_same_property':matching['selected'],'frame_steps':[0,4,8,12,16,20,24,28,32],
        'candidates':[{**c,**{k:review['candidates'][c['id']][k] for k in ('label','acceptable','nuisance_notes')}} for c in index]})
    return folder


def run_reviewed(root, protocol, config):
    p=None;outcomes=[]
    try:
        # Freeze ALL selections before seeing any replication effects.
        for state in protocol['initial_states']:
            case=root/f'state_{state}'
            review=json.loads((case/'selection_review.json').read_text())
            if review.get('status')=='rejected_primary_semantics':
                if not review.get('rejection_reason'):raise ValueError('Rejected state needs reason')
                outcomes.append({'state':state,'status':'rejected','reason':review['rejection_reason']});continue
            for seed in protocol['action_seeds']:
                folder=case/f'seed_{seed}_prepared'
                if not folder.exists():prepare_recipient(case,protocol,seed)
                elif (case/f'seed_{seed}_run').exists():
                    # Resume completed output only; interrupted branch needs audit.
                    status=json.loads((case/f'seed_{seed}_run/status.json').read_text())
                    if status['status']!='complete':raise RuntimeError('Partial execution requires audit')
                # Verify any existing frozen matching report still agrees with review.
                _,_,current=reviewed_selection(case,protocol,seed)
                if current!=json.loads((folder/'matching_report.json').read_text()):
                    raise ValueError('Matching review changed after selection freeze')
        selections={str(s):{str(seed):digest_file(root/f'state_{s}/seed_{seed}_prepared/review.json') for seed in protocol['action_seeds']}
                    for s in protocol['initial_states'] if (root/f'state_{s}/seed_0_prepared').exists()}
        freeze=root/'selection_freeze.json'
        if freeze.exists():
            if json.loads(freeze.read_text())['selections']!=selections:raise ValueError('Selections changed after effects')
        else:write(freeze,{'frozen_at':now(),'selections':selections,'human_review':'pending'})
        for state in protocol['initial_states']:
            case=root/f'state_{state}'
            for seed in protocol['action_seeds']:
                prepared=case/f'seed_{seed}_prepared';output=case/f'seed_{seed}_run'
                if not prepared.exists():continue
                if output.exists():
                    outcomes.append({'state':state,'seed':seed,'status':'complete','resumed':True});continue
                write(root/'run_status.json',{'status':'running','state':state,'seed':seed,'updated_at':now()})
                p=new_pilot({**config,'initial_state_index':state,'action_seed':seed},p)
                summary=pilot_cli.run(p,prepared,output)
                outcomes.append({'state':state,'seed':seed,'status':'complete','summary':summary})
                write(root/'run_outcomes.json',outcomes)
                print(json.dumps({'state':state,'seed':seed,'effect':summary['opposite_minus_same']}),flush=True)
        write(root/'run_status.json',{'status':'complete','outcomes':outcomes,'updated_at':now()})
    except Exception as exc:
        write(root/'run_status.json',{'status':'failed','error':str(exc),'updated_at':now()});raise
    finally:
        if p is not None:p.close()


def physical_state(env, object_names):
    sim=env.sim
    positions={name:np.array(sim.data.body_xpos[sim.model.body_name2id(name)]).copy() for name in object_names}
    contacts=[]
    for c in sim.data.contact[:sim.data.ncon]:
        names=[sim.model.geom_id2name(int(c.geom1)),sim.model.geom_id2name(int(c.geom2))]
        if any(n and ('finger' in n or 'gripper' in n) for n in names):
            contacts.append({'geoms':names,'distance':float(c.dist)})
    return positions,contacts


def contact_rollouts(root, protocol, config):
    from PIL import Image
    import torch
    from wam_interp.types import InferenceRequest,EnvironmentObservation
    p=None;follow=protocol['contact_followup']
    try:
        for state in follow['initial_states']:
            out=root/f'contact_state_{state}'
            if out.exists():raise FileExistsError('Do not overwrite or silently resume an interrupted contact episode')
            out.mkdir()
            p=new_pilot({**config,'initial_state_index':state,'action_seed':follow['action_seed']},p)
            p.verify_replay()
            object_names=[n for n in p.env.sim.model.body_names if n.endswith('_main') and not n.startswith('robot')]
            if not object_names:raise RuntimeError('No object bodies identified')
            write(out/'context.json',{**provenance(p),'object_body_names':object_names,'followup':follow})
            chunks=[];stop=False;global_step=0
            for chunk in range(follow['max_chunks']):
                folder=out/f'chunk_{chunk:02d}';folder.mkdir()
                request=replace(p.request,observation=p.adapter.prepare_observation(p.observation))
                torch.cuda.synchronize()
                from time import perf_counter
                start=perf_counter();result,captured=predict_with_future(p.adapter,request);torch.cuda.synchronize()
                inference_seconds=perf_counter()-start
                commands=_array(p.adapter.postprocess_actions(result.actions,environment='libero'))
                if commands.ndim==3:commands=commands[0]
                if commands.shape!=(32,7) or not np.isfinite(commands).all():raise ValueError('Invalid action chunk')
                np.savez_compressed(folder/'policy.npz',future=_array(captured['used']),actions=_array(result.actions),commands=commands)
                save_frames(p.adapter.model,captured['used'],folder/'imagined')
                observations=[p.observation];objects=[];contacts=[]
                positions,pairs=physical_state(p.env,object_names);objects.append(positions);contacts.append(pairs)
                executed=[];reason='chunk_complete'
                for command in commands[:follow['execute_steps']]:
                    step=p.step_env(p.env,command,instruction=p.task.language,metadata={})
                    executed.append(command.copy());p.observation=step.observation;observations.append(p.observation)
                    positions,pairs=physical_state(p.env,object_names);objects.append(positions);contacts.append(pairs)
                    if step.done or step.success:
                        reason='success' if step.success else 'environment_done';stop=True;break
                arrays={'eef_position':np.stack([o.state['eef_position'] for o in observations]),
                    'eef_quaternion':np.stack([o.state['eef_quaternion'] for o in observations]),
                    'gripper_position':np.stack([o.state['gripper_position'] for o in observations]),
                    'agentview':np.stack([o.cameras['agentview'] for o in observations]),
                    'wrist':np.stack([o.cameras['wrist'] for o in observations]),
                    'commands':np.array(executed),'control_steps':np.arange(len(observations))+global_step}
                arrays.update({f'object_{name}':np.stack([x[name] for x in objects]) for name in object_names})
                np.savez_compressed(folder/'execution.npz',**arrays)
                (folder/'executed').mkdir();pairs=[]
                for i,control in enumerate(config['frame_steps']):
                    if control>=len(observations):break
                    im=_array(p.adapter.prepare_observation(observations[control]).image)[0]
                    im=np.clip((im.transpose(1,2,0)+1)*127.5,0,255).astype(np.uint8)
                    Image.fromarray(im).save(folder/f'executed/{i:03d}.png')
                    pairs.append({'step':control,'imagined':f'imagined/{i:03d}.png','executed':f'executed/{i:03d}.png'})
                write(folder/'frame_pairs.json',{'state_id':f'libero_10/0/{state}/chunk_{chunk}',
                    'context_id':digest_file(out/'context.json'),'synthetic':False,'condition':'native_contact',
                    'coordinate_frame':'agentview_rotated_cropped','units':'normalized_image_xy','pairs':pairs})
                from wam_interp.experiments.annotation import make_annotation_page
                make_annotation_page(folder/'frame_pairs.json')
                motion={name:float(np.max(np.linalg.norm(arrays[f'object_{name}']-arrays[f'object_{name}'][0],axis=1))) for name in object_names}
                record={'chunk':chunk,'start_control':global_step,'executed_controls':len(executed),'termination':reason,
                    'object_motion_m':motion,'object_motion_above_threshold':[n for n,v in motion.items() if v>=follow['object_motion_threshold_m']],
                    'inference_seconds':inference_seconds,'contacts':contacts,
                    'note':'Object movement is not proof of a grasp; contact geom names are retained for inspection.'}
                write(folder/'summary.json',record);chunks.append(record);global_step+=len(executed)
                write(out/'status.json',{'status':'running','completed_chunks':len(chunks),'controls':global_step})
                print(json.dumps({'contact_state':state,'chunk':chunk,'moving_objects':record['object_motion_above_threshold']}),flush=True)
                if stop:break
            write(out/'status.json',{'status':'complete','termination':reason if stop else 'chunk_limit','chunks':chunks,'controls':global_step})
        write(root/'contact_status.json',{'status':'complete','states':follow['initial_states'],'updated_at':now()})
    except Exception as exc:
        write(root/'contact_status.json',{'status':'failed','error':str(exc),'updated_at':now()});raise
    finally:
        if p is not None:p.close()


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('stage',choices=['prepare','run','contact'])
    ap.add_argument('--protocol',type=Path,default=Path('configs/experiments/e1_replication.json'))
    ap.add_argument('--output',type=Path,required=True)
    args=ap.parse_args();protocol=json.loads(args.protocol.read_text())
    config=json.loads(Path(protocol['pilot_config']).read_text())
    config.update(task_id=protocol['task_id'],candidate_seeds=protocol['candidate_seeds'])
    root=args.output;root.mkdir(parents=True,exist_ok=True)
    locked=root/'protocol.json'
    if locked.exists():
        if json.loads(locked.read_text())!=protocol:raise ValueError('Protocol differs from frozen version')
    else:
        if args.stage!='prepare':raise ValueError('Preparation must freeze protocol first')
        write(locked,protocol);write(root/'protocol_frozen.json',{'utc':now(),'sha256':digest_file(locked),'human_review':'pending'})
    {'prepare':prepare_banks,'run':run_reviewed,'contact':contact_rollouts}[args.stage](root,protocol,config)


if __name__=='__main__':main()
