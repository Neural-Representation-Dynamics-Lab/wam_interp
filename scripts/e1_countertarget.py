"""Two alternative native targets under one unchanged recipient instruction."""
import argparse,json
from pathlib import Path
import numpy as np
from PIL import Image

import e1_native_search as search
from e1_replication import write,now,new_pilot,load_future,physical_state
from wam_interp.experiments.libero_pilot import provenance,save_frames
from wam_interp.experiments.fastwam_session import _array
from wam_interp.experiments.e1 import _validate_branch
from wam_interp.types import EnvironmentObservation


def select(root):
    search.verify_source(root)
    if (root/'selection_freeze.json').exists():raise FileExistsError('Selection already frozen')
    if json.loads((root/'prepare_status.json').read_text())['status']!='complete':raise ValueError('Acquisition incomplete')
    protocol=json.loads((root/'protocol.json').read_text())
    selected={str(s):search.reviewed_state(root,protocol,s)[0] for s in protocol['initial_states']}
    write(root/'selection_freeze.json',{'at':now(),'states':selected,'human_review':'pending','no_recipient_effects_observed':True})
    print(json.dumps({'matched_states':[s for s,v in selected.items() if v['selected'] is not None]}),flush=True)


def run(root):
    search.verify_source(root)
    if (root/'run_status.json').exists():raise FileExistsError('No silent restart')
    protocol=json.loads((root/'protocol.json').read_text())
    frozen=json.loads((root/'selection_freeze.json').read_text())
    for state in protocol['initial_states']:
        if search.reviewed_state(root,protocol,state)[0]!=frozen['states'][str(state)]:raise ValueError('Review changed')
    config=search.config_for(protocol);p=None;results=[]
    try:
        write(root/'run_status.json',{'status':'running','at':now()})
        for state in protocol['initial_states']:
            selected=frozen['states'][str(state)]['selected']
            if selected is None:continue
            case=root/f'state_{state}';bank=case/'bank'
            contexts=json.loads((case/'recipient_contexts.json').read_text())
            geometry=json.loads((case/'target_geometry.json').read_text())
            direction=np.array(geometry['world_xy_direction'])
            for seed in protocol['action_seeds']:
                p=new_pilot({**config,'initial_state_index':state,'action_seed':seed},p)
                p.verify_replay();current=provenance(p)
                if current!=contexts[str(seed)]['context']:raise ValueError('Recipient context changed')
                positions,_=physical_state(p.env,protocol['target_bodies'])
                for name,value in positions.items():
                    if not np.array_equal(value,geometry['positions'][name]):raise ValueError('Target positions changed')
                out=case/f'seed_{seed}_run';out.mkdir()
                write(out/'context.json',current)
                baseline_future=load_future(bank/f'baseline_seed_{seed}.npz')
                donors={'baseline':None,'identity':baseline_future,
                    'donor_a':load_future(bank/f"{selected['same']}.npz"),
                    'donor_b':load_future(bank/f"{selected['alternative']}.npz")}
                initial=None;baseline_actions=None;traces={}
                for name,donor in donors.items():
                    p.restore(p.saved_rng);branch=p.session.run(donor);_validate_branch(branch)
                    execution={k:v.copy() for k,v in p.last_rollout.items()}
                    if not np.array_equal(branch.future[:,:,:1],baseline_future[:,:,:1]):raise ValueError('Anchor changed')
                    if name=='baseline':
                        if not np.array_equal(branch.future,baseline_future):raise ValueError('Baseline future changed')
                        initial=execution;baseline_actions=branch.actions.copy()
                    if name=='identity':
                        if not np.array_equal(branch.actions,baseline_actions) or any(not np.array_equal(v,execution[k]) for k,v in initial.items()):
                            raise ValueError('Identity replay differs')
                    if donor is not None and not np.array_equal(branch.future[:,:,1:],donor[:,:,1:]):
                        raise ValueError('Native donor insertion differs')
                    if branch.actions.shape!=(32,7):raise ValueError('Unexpected action shape')
                    np.savez_compressed(out/f'{name}.npz',future=branch.future,actions=branch.actions,
                        commands=branch.commands,executed_commands=branch.executed_commands)
                    np.savez_compressed(out/f'{name}_execution.npz',**execution)
                    write(out/f'{name}.json',{'metadata':branch.metadata,'behavior_world_y_m':branch.behavior,
                        'donor_label':protocol['contrast_labels'][0] if name=='donor_a' else protocol['contrast_labels'][1] if name=='donor_b' else None,
                        'recipient_instruction':p.task.language})
                    save_frames(p.adapter.model,branch.future,out/f'{name}_imagined')
                    actual=out/f'{name}_executed';actual.mkdir();pairs=[]
                    for i,step in enumerate(config['frame_steps']):
                        if step>=len(execution['control_steps']):break
                        obs=EnvironmentObservation(cameras={k:execution[k][step] for k in ('agentview','wrist')},
                            state=p.reference.state,instruction=p.task.language)
                        im=_array(p.adapter.prepare_observation(obs).image)[0]
                        Image.fromarray(np.clip((im.transpose(1,2,0)+1)*127.5,0,255).astype(np.uint8)).save(actual/f'{i:03}.png')
                        pairs.append({'step':step,'imagined':f'{name}_imagined/{i:03}.png','executed':f'{name}_executed/{i:03}.png'})
                    write(out/f'{name}_frame_pairs.json',{'pairs':pairs,'coordinate_frame':'agentview_native_224_pixels'})
                    traces[name]=execution['eef_position']
                    print(json.dumps({'state':state,'seed':seed,'arm':name,'complete':True}),flush=True)
                a,b=traces['donor_a'],traces['donor_b']
                if a.shape!=b.shape:raise ValueError('Unequal rollout lengths')
                r={'state':state,'seed':seed,'target_direction':direction.tolist(),
                    'primary_target_projection_m':float(np.dot(b[-1]-a[-1],direction)),
                    'world_y_contrast_m':float(b[-1,1]-a[-1,1]),
                    'endpoint_separation_m':float(np.linalg.norm(b[-1]-a[-1])),
                    'trajectory_separation_m':float(np.linalg.norm(b-a,axis=1).mean())}
                write(out/'comparison.json',r);results.append(r)
                write(out/'status.json',{'status':'complete','branches':4,'identity_exact':True})
                write(root/'run_results.json',results)
        search.verify_source(root)
        write(root/'run_status.json',{'status':'complete','cases':len(results),'branches':4*len(results),'at':now(),
            'human_review':'pending','no_unmatched_comparisons':True})
    except Exception as exc:
        write(root/'run_status.json',{'status':'failed','error':str(exc),'at':now()});raise
    finally:
        if p is not None:p.close()


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('stage',choices=['prepare','select','run'])
    ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--protocol',type=Path,default=Path('configs/experiments/e1_countertarget.json'))
    args=ap.parse_args()
    if args.stage=='prepare':search.prepare(args.output,json.loads(args.protocol.read_text()))
    elif args.stage=='select':select(args.output)
    else:run(args.output)
