"""Freeze and execute a bounded equal-norm diagnostic on previously recorded states.

Not a matched native-donor experiment. Never edits the source experiment.
"""
import argparse
from pathlib import Path
import json
import shutil
import zipfile

import numpy as np

from e1_replication import write, now, new_pilot, load_future
from wam_interp.experiments.equal_norm import equal_norm_plan, realized_geometry
from wam_interp.experiments.e1 import _validate_branch
from wam_interp.experiments.fastwam_session import _array
from wam_interp.experiments.libero_pilot import digest_file, provenance, save_frames


def freeze(protocol, root):
    if root.exists():
        raise FileExistsError('Use a fresh output; interrupted runs require an explicit audit')
    source = Path(protocol['source_results'])
    # Validate all input paths before creating any output.
    paths = {}
    for state in protocol['initial_states']:
        bank = source / f'state_{state}' / 'bank'
        for seed in protocol['action_seeds']:
            key = f'state_{state}/seed_{seed}'
            run = source / f'state_{state}' / f'seed_{seed}_run'
            prepared = source / f'state_{state}' / f'seed_{seed}_prepared'
            paths[key] = {
                'baseline.npz': bank / f'baseline_seed_{seed}.npz',
                'same.npz': bank / f"{protocol['same_donor']}.npz",
                'alternative.npz': bank / f"{protocol['alternative_donor']}.npz",
                'previous_context.json': prepared / 'context.json',
                'previous_baseline.npz': run / 'baseline.npz',
                'previous_execution.npz': run / 'baseline_execution.npz',
                'donor_review.json': source / f'state_{state}' / 'selection_review.json',
            }
            for p in paths[key].values():
                if not p.is_file(): raise FileNotFoundError(p)
    root.mkdir(parents=True)
    write(root / 'protocol.json', protocol)
    manifest = {'frozen_at': now(), 'inputs': {}, 'protocol_sha256': digest_file(root/'protocol.json')}
    for key, files in paths.items():
        folder = root / key / 'inputs'; folder.mkdir(parents=True)
        for name, path in files.items():
            shutil.copyfile(path, folder/name)
            manifest['inputs'][f'{key}/inputs/{name}'] = {'source': str(path), 'sha256': digest_file(folder/name)}
        arms, meta = equal_norm_plan(*(load_future(folder/f'{name}.npz') for name in ('baseline', 'same', 'alternative')),
            fractions=protocol['radius_fractions'], random_seeds=protocol['random_direction_seeds'])
        np.savez_compressed(folder/'planned_interventions.npz', **arms)
        write(folder/'plan.json', meta)
        for name in ('planned_interventions.npz', 'plan.json'):
            manifest['inputs'][f'{key}/inputs/{name}'] = {'source': 'fixed algorithm, no recipient effects', 'sha256': digest_file(folder/name)}
    # Freeze current source, including dirty files, without assets/environments.
    with zipfile.ZipFile(root/'source.zip', 'w', zipfile.ZIP_DEFLATED) as archive:
        for folder in ('src', 'configs', 'scripts'):
            for p in sorted(Path(folder).rglob('*')):
                if p.is_file() and p.suffix in ('.py', '.json', '.yaml', '.sh'):
                    archive.write(p, p.as_posix())
        for name in ('__init__.py', 'pyproject.toml'):
            archive.write(name, name)
    manifest['source_archive_sha256'] = digest_file(root/'source.zip')
    write(root/'freeze.json', manifest)
    write(root/'status.json', {'status': 'prepared', 'human_review': 'pending'})
    print(json.dumps({'prepared_cases': len(paths), 'root': str(root)}), flush=True)


def verify_frozen(root):
    frozen = json.loads((root/'freeze.json').read_text())
    if digest_file(root/'protocol.json') != frozen['protocol_sha256']:
        raise ValueError('Protocol changed after freeze')
    for name, record in frozen['inputs'].items():
        if digest_file(root/name) != record['sha256']:
            raise ValueError(f'Frozen input changed: {name}')
    if digest_file(root/'source.zip') != frozen['source_archive_sha256']:
        raise ValueError('Frozen source archive changed')
    with zipfile.ZipFile(root/'source.zip') as archive:
        for name in archive.namelist():
            if Path(name).read_bytes() != archive.read(name):
                raise ValueError(f'Source changed after freeze: {name}')


def check_context(previous, current):
    # New diagnostic source is explicitly versioned. All actual inference inputs
    # and assets must match; source identity is not silently equated.
    mismatches = [key for key in previous if key != 'source_sha256' and previous[key] != current.get(key)]
    if mismatches:
        raise ValueError(f'Historical recipient context changed: {mismatches}')


def run(root):
    import torch
    from PIL import Image
    from wam_interp.types import EnvironmentObservation
    verify_frozen(root)
    if json.loads((root/'status.json').read_text())['status'] != 'prepared':
        raise ValueError('Run is not freshly prepared; no silent restart')
    protocol = json.loads((root/'protocol.json').read_text())
    config = json.loads(Path(protocol['pilot_config']).read_text())
    p = None; records = []
    try:
        write(root/'status.json', {'status': 'running', 'started_at': now()})
        for state in protocol['initial_states']:
            for seed in protocol['action_seeds']:
                case = root/f'state_{state}'/f'seed_{seed}'; inputs = case/'inputs'
                previous = json.loads((inputs/'previous_context.json').read_text())
                p = new_pilot({**config, 'initial_state_index': state, 'action_seed': seed,
                    'candidate_seeds': previous['protocol']['donor_generation']['seeds']}, p)
                p.verify_replay()
                current = provenance(p)
                check_context(previous, current)
                write(case/'context.json', current)
                baseline = load_future(inputs/'baseline.npz')
                plan = json.loads((inputs/'plan.json').read_text())
                with np.load(inputs/'planned_interventions.npz', allow_pickle=False) as a:
                    arms = {k: a[k].copy() for k in a.files}
                donors = {'baseline': None, 'identity': baseline,
                          'native_same': load_future(inputs/'same.npz'),
                          'native_alternative': load_future(inputs/'alternative.npz'), **arms}
                baseline_arrays = None; baseline_actions = None
                branches = {}
                for name, donor in donors.items():
                    folder = case/name; folder.mkdir()
                    p.restore(p.saved_rng)
                    branch = p.session.run(donor)
                    _validate_branch(branch)
                    if branch.actions.shape != (32, 7): raise ValueError('Unexpected action shape')
                    execution = {k: v.copy() for k, v in p.last_rollout.items()}
                    if not np.array_equal(branch.future[:, :, :1], baseline[:, :, :1]):
                        raise ValueError('Observed anchor changed')
                    if name == 'baseline':
                        if not np.array_equal(branch.future, baseline): raise ValueError('Historical baseline future differs')
                        with np.load(inputs/'previous_baseline.npz', allow_pickle=False) as a:
                            if not np.array_equal(a['actions'], branch.actions): raise ValueError('Historical baseline actions differ')
                        with np.load(inputs/'previous_execution.npz', allow_pickle=False) as a:
                            if set(a.files) != set(execution) or any(not np.array_equal(a[k], execution[k]) for k in a.files):
                                raise ValueError('Historical baseline simulator trace differs')
                        baseline_arrays, baseline_actions = execution, branch.actions.copy()
                    elif name == 'identity':
                        if not np.array_equal(branch.actions, baseline_actions) or any(
                            not np.array_equal(v, execution[k]) for k, v in baseline_arrays.items()
                        ): raise ValueError('Identity control differs')
                    geometry = None
                    if donor is not None:
                        expected = torch.as_tensor(donor).to(torch.bfloat16).float().numpy()
                        if not np.array_equal(expected[:, :, 1:], branch.future[:, :, 1:]):
                            raise ValueError('Inserted future differs from planned bf16 values')
                    if name in arms:
                        geometry = realized_geometry(baseline, donor, branch.future)
                        if geometry['relative_norm_error'] > protocol['maximum_realized_norm_relative_error'] or geometry['direction_cosine'] < protocol['minimum_realized_direction_cosine']:
                            raise ValueError(f'Quantized geometry failed: {name}: {geometry}')
                    np.savez_compressed(folder/'policy.npz', future=branch.future, actions=branch.actions,
                        commands=branch.commands, executed_commands=branch.executed_commands)
                    np.savez_compressed(folder/'execution.npz', **execution)
                    count = save_frames(p.adapter.model, branch.future, folder/'imagined')
                    if count != 9: raise ValueError('Unexpected decoded frame count')
                    (folder/'executed').mkdir(); pairs = []
                    for i, step in enumerate(config['frame_steps']):
                        if step >= len(execution['control_steps']): break
                        obs = EnvironmentObservation(cameras={k: execution[k][step] for k in ('agentview', 'wrist')},
                            state=p.reference.state, instruction=p.task.language)
                        image = _array(p.adapter.prepare_observation(obs).image)[0]
                        pixels = np.clip((image.transpose(1, 2, 0)+1)*127.5, 0, 255).astype(np.uint8)
                        Image.fromarray(pixels).save(folder/f'executed/{i:03}.png')
                        pairs.append({'step': step, 'imagined': f'imagined/{i:03}.png', 'executed': f'executed/{i:03}.png'})
                    record = {'state': state, 'action_seed': seed, 'arm': name, 'behavior': branch.behavior,
                        'geometry': geometry, 'metadata': branch.metadata, 'frame_pairs': pairs,
                        'audit': {'finite': True, 'anchor_exact': True, 'commands_exact_prefix': True,
                                  'future_insertion_exact': True}}
                    write(folder/'record.json', record); records.append(record)
                    branches[name] = execution['eef_position']
                    print(json.dumps({'state': state, 'seed': seed, 'arm': name, 'status': 'complete'}), flush=True)
                comparisons = []
                for index, fraction in enumerate(protocol['radius_fractions']):
                    same = branches[f'radius_{index}_same']; alt = branches[f'radius_{index}_alternative']
                    if same.shape != alt.shape: raise ValueError('Unequal trajectory lengths require separate analysis')
                    comparisons.append({'fraction': fraction,
                        'alternative_minus_same_world_y_m': float((alt[-1, 1]-alt[0, 1])-(same[-1, 1]-same[0, 1])),
                        'endpoint_separation_m': float(np.linalg.norm(alt[-1]-same[-1])),
                        'mean_trajectory_separation_m': float(np.linalg.norm(alt-same, axis=1).mean())})
                write(case/'comparisons.json', comparisons)
                write(case/'status.json', {'status': 'complete', 'branches': len(donors),
                    'historical_baseline_exact': True, 'identity_actions_and_all_simulator_arrays_exact': True})
                write(root/'records.json', records)
        verify_frozen(root)
        write(root/'status.json', {'status': 'complete', 'completed_at': now(), 'branches': len(records),
                                  'human_review': 'pending', 'interpretation': protocol['interpretation']})
    except Exception as exc:
        write(root/'status.json', {'status': 'failed', 'error': str(exc), 'at': now()})
        raise
    finally:
        if p is not None: p.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['prepare', 'run'])
    parser.add_argument('--protocol', type=Path, default=Path('configs/experiments/e1_equal_norm.json'))
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.stage == 'prepare': freeze(json.loads(args.protocol.read_text()), args.output)
    else: run(args.output)


if __name__ == '__main__': main()
