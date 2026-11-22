"""Validate native target reassignment before a held-out fixed-motion study."""
import argparse
from functools import partial
from pathlib import Path
import json
import numpy as np
from e1_material_cues import (prepare as prepare_partition, controls as run_controls,
    donors as run_donors, request, run_branch, load_refs, status, require)
from e1_replication import write, now
from e1_simulator_pairs import run_reference, sheet
from e1_visual_replication import verify, verify_manifest, attach_physics, model_identity
from wam_interp.experiments.libero_pilot import LiberoPilot, load_pilot, digest_file, digest_array
from wam_interp.experiments.fastwam_session import _array
from wam_interp.experiments.simulator_identity import ObjectAssetCapture, reset_with_layout
from wam_interp.experiments.simulator_video import valid_approach
from wam_interp.experiments.object_identity import identity_reversal

read = lambda path: json.loads(Path(path).read_text())


def make_pilot(protocol, state, loaded=None):
    from wam_interp.evaluations.LIBERO.runner import _get_suite
    from wam_interp.evaluations.LIBERO.environment import (initial_states_for_task,
        create_libero_environment, step_libero)
    c = {**read(protocol['pilot_config']), 'suite': protocol['suite'], 'task_id': protocol['task_id'],
         'initial_state_index': state, 'action_seed': 0, 'object_offsets_xy_m': protocol['layouts'][str(state)]}
    c.pop('donor_prompts', None)
    if c['warmup_steps'] <= 0: raise ValueError('Layout edits require an observed warm-up')
    if loaded is None: loaded = load_pilot(c)
    loaded.close(); suite = _get_suite(c['suite'])
    obj = LiberoPilot(c, loaded.adapter, task=suite.get_task(c['task_id']),
        initial_state=initial_states_for_task(suite, c['task_id'])[state],
        create_env=create_libero_environment,
        reset_env=partial(reset_with_layout, offsets=protocol['layouts'][str(state)]), step_env=step_libero)
    obj.resolved_model_config = loaded.resolved_model_config
    obj.verify_replay(); attach_physics(obj)
    return obj


def prepare(root, p):
    if root.exists(): raise FileExistsError('Use a fresh experiment directory')
    states = p['development_states'] + p['heldout_states']
    if len(set(states)) != len(states) or set(p['layouts']) != {str(s) for s in states}:
        raise ValueError('Disjoint declared layouts required')
    root.mkdir(parents=True); write(root/'protocol.json', p)
    for partition in ('development', 'heldout'):
        selected = p[partition+'_states']
        prepare_partition(root/partition, {**p, 'states': selected, 'minimum_states': len(selected)})
    write(root/'freeze.json', {'at': now(), 'protocol_sha256': digest_file(root/'protocol.json'),
        'partition_freeze_hashes': {s: digest_file(root/s/'freeze.json') for s in ('development', 'heldout')}})
    status(root, 'prepared')


def verify_all(root):
    f = read(root/'freeze.json')
    if digest_file(root/'protocol.json') != f['protocol_sha256']: raise ValueError('Protocol drift')
    for partition, value in f['partition_freeze_hashes'].items():
        if digest_file(root/partition/'freeze.json') != value: raise ValueError('Partition freeze drift')
        verify(root/partition)


def references(root):
    require(root, 'prepared'); verify_all(root); status(root, 'references_running')
    all_positions = []; all_valid = True
    for partition in ('development', 'heldout'):
        folder = root/partition; p = read(folder/'protocol.json'); rows = []
        base = read(p['reference_config']); base.pop('previous_results', None)
        for state in p['states']:
            c = {**base, 'suite': p['suite'], 'task_id': p['task_id'], 'targets': p['targets'],
                'appearance': {'factors': {'swapped': [1,1,1], 'material_only': [1,1,1]}},
                'save_variant_depths': True, 'object_offsets_xy_m': p['layouts'][str(state)],
                'controller': p.get('reference_controller',base['controller'])}
            case = folder/f'state_{state}'; dest = case/'references/a'; dest.mkdir(parents=True)
            write(case/'reference_config.json', c)
            try:
                factory = lambda env: ObjectAssetCapture(env, list(c['targets'].values()))
                reset = partial(reset_with_layout, offsets=c['object_offsets_xy_m'])
                a, m = run_reference(c, state, 'a', capture_factory=factory, reset_fn=reset)
                repeated, _ = run_reference(c, state, 'a', commands=a['commands'], capture=False, capture_factory=factory, reset_fn=reset)
                if any(not np.array_equal(a[k], v) for k, v in repeated.items()): raise ValueError('Fresh unedited replay differs')
                reasons = []
                if not valid_approach(m['approach'], c['approach_limits']): reasons.append('approach_geometry_or_contacts')
                for item in m['appearance_validation']:
                    if item['changed_outside_boundary_support']: reasons.append('RGB_outside_union')
                    if item['variant'] == 'swapped':
                        if not all(item[k] for k in ('outside_support_depth_exact', 'robot_interior_rgb_exact', 'physics_camera_and_geometry_restored_exact')):
                            reasons.append('asset_geometry_or_robot_gate')
                    elif not item['depth_exact']: reasons.append('material_depth_changed')
                for cam in ('agentview', 'wrist'):
                    if sum(v['changed_pixels'] >= p['minimum_changed_pixels'] for v in m['appearance_validation'] if v['camera']==cam and v['variant']=='swapped') < p['minimum_visible_frames_per_camera']:
                        reasons.append('insufficient_visibility')
                np.savez_compressed(dest/'reference.npz', **a)
                for v in ('original','swapped','material_only'): sheet(a['clip_'+v], dest/(v+'_sheet.jpg'))
                m.update(accepted=not reasons, rejection_reasons=sorted(set(reasons)), reference_sha256=digest_file(dest/'reference.npz'),
                    fresh_unedited_replay_exact=True, identity_scope='Complete native visual assets translated between XY slots; collision geometry and physical identities unchanged.')
                all_positions.append({'partition':partition, 'state':state, 'names':m['object_names'], 'positions':a['object_position'][0].tolist()})
            except Exception as exc:
                m = {'accepted':False, 'rejection_reasons':['reference_error'], 'error':repr(exc)}
            write(dest/'metadata.json', m); rows.append({'state':state, 'accepted':m['accepted'], 'reasons':m['rejection_reasons']})
            print(json.dumps(rows[-1]), flush=True)
        write(folder/'reference_selection.json', rows)
        write(folder/'reference_manifest.json', {f.relative_to(folder).as_posix():digest_file(f) for f in folder.glob('state_*/**/*') if f.is_file()})
        valid = all(x['accepted'] for x in rows); all_valid &= valid
        status(folder, 'references_ready' if valid else 'stopped_reference_gate')
    for i, one in enumerate(all_positions):
        for two in all_positions[:i]:
            if np.max(np.linalg.norm(np.array(one['positions'])-two['positions'], axis=1)) < .01:
                all_valid = False
    write(root/'layout_positions.json', all_positions); verify_all(root)
    write(root/'source_review.json', {'method':None, 'before_policy_outcomes':False, 'human_review':'pending', 'states':{str(s['state']):None for s in all_positions}})
    status(root, 'references_ready' if all_valid else 'stopped_reference_gate')


def evaluate_identity_gate(folder, minimum_passing_states):
    p = read(folder/'protocol.json'); rows = []; runtime = []
    for state in p['states']:
        base = folder/f'state_{state}'; meta = read(base/'references/a/metadata.json')
        with np.load(base/'references/a/reference.npz', allow_pickle=False) as z:
            positions = z['object_position'][0]
        axis = positions[meta['other_index']] - positions[meta['target_index']]; axis[2] = 0
        for seed in p['action_seeds']:
            endpoints = {}; safe = True
            for current in ('original', 'swapped'):
                for language in p['instructions']:
                    case = base/f'seed_{seed}'/current/language/'baseline'; record = read(case/'record.json')
                    if digest_file(case/'execution.npz') != record['execution_sha256']: raise ValueError('Execution hash mismatch')
                    if record['recipient_instruction'] != p['instructions'][language]: raise ValueError('Instruction mismatch')
                    with np.load(case/'execution.npz', allow_pickle=False) as z:
                        endpoints[language,current] = z['eef_position'][-1].copy()
                        movement = float(np.linalg.norm(z['object_position']-z['object_position'][:1],axis=-1).max())
                    contacts = len(record['metadata']['rollout']['contact_events'])
                    safe &= contacts==0 and movement<=.001
                    runtime.append({'state':state,'seed':seed,'current':current,'language':language,'contact_substeps':contacts,'max_object_translation_m':movement})
            item = identity_reversal(endpoints, axis, **p['identity_limits'])
            item.update(state=state, seed=seed, contact_and_object_gate=safe)
            item['pass'] &= safe; rows.append(item)
    passed = [s for s in p['states'] if all(x['pass'] for x in rows if x['state']==s)]
    result = {'at':now(),'rows':rows,'runtime':runtime,'passing_states':passed,'all_states_retained':p['states'],
              'minimum_passing_states':minimum_passing_states,'pass':len(passed)>=minimum_passing_states,
              'scope':'Operational target reassignment under an image edit; not proof of a distinct internal identity representation.'}
    write(folder/'native_identity_gate.json', result); return result


def development(root):
    require(root,'references_ready'); verify_all(root)
    review = read(root/'source_review.json'); p = read(root/'protocol.json')
    if review['method'] not in ('AI_assisted','human') or not review['before_policy_outcomes']:
        raise ValueError('Actual pre-outcome source review required')
    if set(review['states']) != set(p['layouts']) or any(not x or not x.get('accepted') or not x.get('notes') for x in review['states'].values()):
        raise ValueError('Review every source arrangement before policy outcomes')
    write(root/'validation_freeze.json', {'at':now(),'source_review_sha256':digest_file(root/'source_review.json')})
    folder=root/'development'; p=read(folder/'protocol.json'); obj=None; branches=0
    status(root,'development_running')
    try:
        for state in p['states']:
            verify_manifest(folder,folder/'reference_manifest.json'); obj=make_pilot(p,state,obj); native=obj.request; refs=load_refs(folder,state)
            if not (folder/'model_identity.json').exists(): write(folder/'model_identity.json',model_identity(obj))
            for current in ('original','swapped','material_only'):
                for language,prompt in p['instructions'].items():
                    for seed in p['action_seeds']:
                        request(obj,native,refs,current,seed,prompt)
                        case=folder/f'state_{state}/seed_{seed}/{current}/{language}'
                        write(case/'context.json',{'state':state,'seed':seed,'current':current,'language':language,'prompt':prompt,
                            'image_sha256':digest_array(obj.request.observation.image),'proprio_sha256':digest_array(obj.request.observation.proprio),
                            'state_sha256':digest_array(obj.initial_state),'layout_offsets_xy_m':p['layouts'][str(state)]})
                        result,capture,_=obj.infer(); baseline=_array(capture['used'])
                        b,e=run_branch(obj,case,'baseline',None,baseline); branches+=1
                        if not np.array_equal(b.future,baseline) or not np.array_equal(b.actions,_array(result.actions)): raise ValueError('Native repeat differs')
                        if current=='original':run_branch(obj,case,'identity',baseline,baseline,e,b.actions);branches+=1
                        write(root/'progress.json',{'stage':'development','branches':branches,'state':state,'current':current,'language':language,'seed':seed})
        gate=evaluate_identity_gate(folder,len(p['states'])); verify_all(root)
        status(folder,'complete',branches=branches)
        status(root,'development_validated' if gate['pass'] else 'stopped_identity_gate',branches=branches,passing_states=gate['passing_states'])
    finally:
        if obj is not None:obj.close()


def controls(root):
    require(root,'development_validated');verify_all(root);status(root,'heldout_controls_running')
    run_controls(root/'heldout',pilot_factory=make_pilot)
    if read(root/'heldout/status.json')['status']!='controls_ready':status(root,'stopped_baseline_gate');return
    gate=evaluate_identity_gate(root/'heldout',read(root/'protocol.json')['minimum_heldout_identity_states'])
    status(root,'awaiting_codec_review' if gate['pass'] else 'stopped_heldout_identity_gate',passing_states=gate['passing_states'])


def donors(root):
    require(root,'awaiting_codec_review');verify_all(root)
    if digest_file(root/'source_review.json')!=read(root/'validation_freeze.json')['source_review_sha256']:raise ValueError('Source review drift')
    run_donors(root/'heldout',pilot_factory=make_pilot)
    child=read(root/'heldout/status.json');status(root,child['status'],heldout_branches=child.get('branches'),development_branches=48)


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('stage',choices=['prepare','references','development','controls','donors'])
    ap.add_argument('--output',type=Path,required=True);ap.add_argument('--protocol',type=Path,default=Path('configs/experiments/e1_object_identity.json'));args=ap.parse_args()
    try:
        if args.stage=='prepare':prepare(args.output,read(args.protocol))
        else:globals()[args.stage](args.output)
    except BaseException as exc:
        if args.stage!='prepare':write(args.output/'failure.json',{'at':now(),'stage':args.stage,'error':repr(exc)});status(args.output,'failed',stage=args.stage,error=repr(exc))
        raise
