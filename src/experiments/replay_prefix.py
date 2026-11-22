"""Replay recorded native commands to a later, exactly checked observation."""
from pathlib import Path
import hashlib
import numpy as np


OBSERVATION_FIELDS = {
    'agentview': ('cameras', 'agentview'), 'wrist': ('cameras', 'wrist'),
    'eef_position': ('state', 'eef_position'),
    'eef_quaternion': ('state', 'eef_quaternion'),
    'gripper_position': ('state', 'gripper_position'),
}


def load_replay_prefix(config):
    spec = config.get('replay_prefix')
    if spec is None:
        return None, None
    state = str(config['initial_state_index'])
    path = Path(spec['directory']) / f'state_{state}.npz'
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != spec['sha256'][state]:
        raise ValueError('Recorded replay prefix hash differs')
    steps = spec['steps']
    if type(steps) is not int or steps < 1:
        raise ValueError('Positive integer prefix length required')
    with np.load(path, allow_pickle=False) as archive:
        data = {k: archive[k].copy() for k in (*OBSERVATION_FIELDS, 'commands', 'control_steps')}
    if data['commands'].shape != (steps, 7):
        raise ValueError('Replay commands must have shape (prefix_steps, 7)')
    if not np.array_equal(data['control_steps'], np.arange(steps + 1)):
        raise ValueError('Replay control steps differ')
    for name, value in data.items():
        if not np.isfinite(value).all():
            raise ValueError(f'Nonfinite replay data: {name}')
        if name in OBSERVATION_FIELDS and len(value) != steps + 1:
            raise ValueError(f'Incomplete replay observations: {name}')
        value.flags.writeable = False
    return data, {'path': str(path), 'sha256': digest, 'steps': steps,
                  'verification': 'all prefix observations, both cameras and proprioception, exact'}


def verify_prefix_observation(observation, data, index):
    for key, (group, field) in OBSERVATION_FIELDS.items():
        if not np.array_equal(getattr(observation, group)[field], data[key][index]):
            raise RuntimeError(f'Recorded prefix replay differs at control {index}: {key}')


def replay_prefix(pilot):
    data = pilot.replay_data
    if data is None:
        return
    verify_prefix_observation(pilot.observation, data, 0)
    for index, command in enumerate(data['commands'], 1):
        step = pilot.step_env(pilot.env, command.copy(), instruction=pilot.task.language,
                              metadata={'replayed_prefix_step': index})
        if step.done or step.success:
            raise RuntimeError('Episode terminated during recorded prefix')
        pilot.observation = step.observation
        verify_prefix_observation(pilot.observation, data, index)
