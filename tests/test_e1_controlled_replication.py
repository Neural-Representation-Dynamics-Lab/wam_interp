import copy
import numpy as np
import pytest
from wam_interp.experiments.controlled_replication import validate_protocol,choose_appearance,state_bootstrap


def test_partition_guard_rejects_leakage_and_previous_exposure():
    p={'calibration_states':[16,17],'test_states':[20,21],'tasks':[{'task_id':0,'targets':{'a':'a','b':'b'}}], 'previously_exposed':{'0':[3,4,5]},'human_review':'pending'}
    validate_protocol(p)
    for key,value in [('test_states',[16,20]),('test_states',[3,20]),('calibration_states',[16,16])]:
        q=copy.deepcopy(p);q[key]=value
        with pytest.raises(ValueError):validate_protocol(q)


def calibration_rows():
    return [{'task':t,'state':s,'appearance':a,'motion_rms':2.,'appearance_rms':v,'valid':True}
            for t in (0,7) for s in (16,17) for a,v in [('weak',.2),('strong',1.8)]]


def test_calibration_uses_distances_and_rejects_policy_outcome_fields():
    rows=calibration_rows();assert choose_appearance(rows,['weak','strong'])['selected']=='strong'
    rows[0]['endpoint_shift']=0.
    with pytest.raises(ValueError,match='only representation'):choose_appearance(rows,['weak','strong'])


def test_invalid_candidate_or_missing_calibration_state_cannot_win():
    rows=calibration_rows();next(r for r in rows if r['appearance']=='strong')['valid']=False
    assert choose_appearance(rows,['weak','strong'])['selected']=='weak'
    with pytest.raises(ValueError,match='same calibration states'):choose_appearance(rows[:-1],['weak','strong'])


def test_bootstrap_clusters_seeds_and_weights_tasks_equally():
    rows=[{'task':0,'state':20,'seed':0,'effect':0.}, {'task':0,'state':20,'seed':1,'effect':2.},
          {'task':0,'state':21,'seed':0,'effect':3.}, {'task':0,'state':21,'seed':1,'effect':5.},
          {'task':7,'state':20,'seed':0,'effect':10.}, {'task':7,'state':20,'seed':1,'effect':10.}]
    result=state_bootstrap(rows,'effect',repeats=1000)
    assert result['tasks']['0']['states']=={'20':1.,'21':4.}
    assert result['equal_task_mean']==6.25
    assert result==state_bootstrap(rows,'effect',repeats=1000)
    with pytest.raises(ValueError,match='Duplicate'):state_bootstrap(rows+[rows[0]],'effect')


def test_live_physics_tracking_restores_step_function_on_failure(monkeypatch):
    import sys
    from pathlib import Path
    from types import SimpleNamespace
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
    import e1_visual_replication as runner
    events=[]
    class Capture:
        def __init__(self,env):pass
        def __enter__(self):events.append('enter');return self
        def __exit__(self,*exc):events.append('exit')
    old=lambda *a,**k:None
    p=SimpleNamespace(observation=None,env=None,step_env=old,session=SimpleNamespace())
    def failure(commands):
        assert p.step_env is not old
        raise RuntimeError('step failure')
    p.rollout=failure
    monkeypatch.setattr(runner,'RobotContactCapture',Capture)
    monkeypatch.setattr(runner,'observation_arrays',lambda *a:{'qpos':np.zeros(2)})
    runner.attach_physics(p)
    with pytest.raises(RuntimeError,match='step failure'):p.session.rollout_fn(np.zeros((1,7)))
    assert p.step_env is old and events==['enter','exit']
