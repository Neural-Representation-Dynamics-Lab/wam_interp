import hashlib
from types import SimpleNamespace
import numpy as np
import pytest

from wam_interp.experiments.replay_prefix import load_replay_prefix, replay_prefix
from wam_interp.experiments.libero_pilot import LiberoPilot
from wam_interp.types import EnvironmentObservation, PreparedObservation


def obs(x):
    return EnvironmentObservation(
        {'agentview': np.full((2,2,3), x), 'wrist': np.full((2,2,3), x)},
        {'eef_position': np.array([x,0.,0.]), 'eef_quaternion': np.array([0,0,0,1]),
         'gripper_position': np.zeros(2)}, 'original')


def fixture(tmp_path):
    observations=[obs(i) for i in range(3)]
    data={'commands':np.array([[1,0,0,0,0,0,-1]]*2,dtype=float),'control_steps':np.arange(3)}
    for group in ('cameras','state'):
        for key in getattr(observations[0],group):
            data[key]=np.stack([getattr(o,group)[key] for o in observations])
    path=tmp_path/'state_3.npz';np.savez_compressed(path,**data)
    config={'initial_state_index':3,'replay_prefix':{'directory':str(tmp_path),'steps':2,
        'sha256':{'3':hashlib.sha256(path.read_bytes()).hexdigest()}},
        'action_seed':0,'inference_steps':10,'execute_steps':32,'task_id':0,
        'camera_resolution':2,'environment_seed':0,'warmup_steps':0,'behavior_axis':1}
    return config,data


def test_native_prefix_restores_controller_and_observation(tmp_path):
    config,data=fixture(tmp_path);envs=[]
    def create(*args):
        e=SimpleNamespace(x=0.,steps=0,closed=False)
        e.close=lambda:setattr(e,'closed',True);envs.append(e);return e
    def reset(e,*args,**kw):return obs(0.)
    def step(e,command,**kw):
        e.x+=command[0];e.steps+=1
        return SimpleNamespace(observation=obs(e.x),success=False,done=False)
    adapter=SimpleNamespace(inference_defaults=lambda:{'num_video_frames':9},
        prepare_observation=lambda o:PreparedObservation(None,None,o.instruction))
    p=LiberoPilot(config,adapter,task=SimpleNamespace(language='original'),initial_state=np.zeros(2),
        create_env=create,reset_env=reset,step_env=step)
    try:
        assert p.env.steps==2 and p.reference.state['eef_position'][0]==2
        assert p.request.observation.prompt=='original'
        p.verify_replay();p.restore(p.saved_rng)
        assert p.env.steps==2 and p.env.x==2
        assert all(e.closed for e in envs[:-1])
        assert p.replay_identity['steps']==2
    finally:p.close()
    assert all(e.closed for e in envs)


@pytest.mark.parametrize('failure',['hash','length','nan'])
def test_reject_changed_or_invalid_prefix(tmp_path,failure):
    config,data=fixture(tmp_path);path=tmp_path/'state_3.npz'
    if failure=='hash':config['replay_prefix']['sha256']['3']='wrong'
    else:
        if failure=='length':data['commands']=data['commands'][:1]
        else:data['commands'][0,0]=np.nan
        np.savez_compressed(path,**data)
        config['replay_prefix']['sha256']['3']=hashlib.sha256(path.read_bytes()).hexdigest()
    with pytest.raises(ValueError):load_replay_prefix(config)


@pytest.mark.parametrize('failure',['camera','termination'])
def test_reject_prefix_drift_and_early_termination(tmp_path,failure):
    config,_=fixture(tmp_path);data,_=load_replay_prefix(config)
    def step(*args,**kw):
        o=obs(1.)
        if failure=='camera':o.cameras['wrist'][0,0,0]=99
        return SimpleNamespace(observation=o,done=failure=='termination',success=False)
    p=SimpleNamespace(replay_data=data,observation=obs(0.),env=None,
        task=SimpleNamespace(language='original'),step_env=step)
    with pytest.raises(RuntimeError):replay_prefix(p)


def test_prefix_absent_preserves_existing_behavior():
    assert load_replay_prefix({})==(None,None)
