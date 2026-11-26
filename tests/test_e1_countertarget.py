import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
from PIL import Image
from wam_interp.experiments.e1 import Branch


def setup(tmp_path,monkeypatch,selected):
    root=Path(__file__).parents[1];monkeypatch.syspath_prepend(str(root/'scripts'))
    spec=importlib.util.spec_from_file_location('countertarget_cli',root/'scripts/e1_countertarget.py')
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    protocol={'initial_states':[13],'action_seeds':[0],'contrast_labels':['cream','butter'],'target_bodies':['a','b']}
    frozen={'selected':selected}
    (tmp_path/'protocol.json').write_text(json.dumps(protocol))
    (tmp_path/'selection_freeze.json').write_text(json.dumps({'states':{'13':frozen}}))
    monkeypatch.setattr(m.search,'verify_source',lambda r:None)
    monkeypatch.setattr(m.search,'reviewed_state',lambda *args:(frozen,{},[],{}))
    monkeypatch.setattr(m.search,'config_for',lambda p:{'frame_steps':list(range(0,33,4))})
    return m


def test_no_match_never_loads_model(tmp_path,monkeypatch):
    m=setup(tmp_path,monkeypatch,None)
    def forbidden(*args):raise AssertionError('Model must not load')
    monkeypatch.setattr(m,'new_pilot',forbidden)
    m.run(tmp_path)
    assert json.loads((tmp_path/'run_status.json').read_text())['branches']==0


@pytest.mark.parametrize('bad_identity',[False,True])
def test_two_alternative_branches_preserve_instruction_and_cleanup(tmp_path,monkeypatch,bad_identity):
    m=setup(tmp_path,monkeypatch,{'same':'a','alternative':'b'})
    case=tmp_path/'state_13';bank=case/'bank';bank.mkdir(parents=True)
    (case/'recipient_contexts.json').write_text(json.dumps({'0':{'context':{}}}))
    (case/'target_geometry.json').write_text(json.dumps({'world_xy_direction':[1,0,0],'positions':{'a':[0,0,0],'b':[1,0,0]}}))
    base=np.zeros((1,1,3,1,1),np.float32)
    np.savez_compressed(bank/'baseline_seed_0.npz',future=base)
    for name,value in [('a',1),('b',2)]:
        f=base.copy();f[:,:,1:]=value;np.savez_compressed(bank/f'{name}.npz',future=f)
    p=SimpleNamespace(task=SimpleNamespace(language='original soup and tomato'),reference=SimpleNamespace(state={}),
        verify_replay=lambda:None,restore=lambda state:None,saved_rng={},closed=False,calls=0)
    p.close=lambda:setattr(p,'closed',True)
    p.adapter=SimpleNamespace(model=None,prepare_observation=lambda obs:SimpleNamespace(image=np.zeros((1,3,224,448))))
    def branch(donor):
        p.calls+=1;future=base.copy() if donor is None else donor.copy()
        amount=float(future[0,0,1,0,0]);actions=np.full((32,7),amount)
        if bad_identity and p.calls==2:actions[0,0]=99
        positions=np.zeros((33,3));positions[:,0]=np.linspace(0,.01*amount,33)
        p.last_rollout={'eef_position':positions,'eef_quaternion':np.zeros((33,4)),
            'agentview':np.zeros((33,4,4,3),np.uint8),'wrist':np.zeros((33,4,4,3),np.uint8),'control_steps':np.arange(33)}
        return Branch(future,actions,actions.copy(),actions.copy(),0.,{})
    p.session=SimpleNamespace(run=branch)
    monkeypatch.setattr(m,'new_pilot',lambda *args:p)
    monkeypatch.setattr(m,'provenance',lambda p:{})
    monkeypatch.setattr(m,'physical_state',lambda *args:({'a':np.zeros(3),'b':np.array([1.,0,0])},[]))
    p.env=None
    def frames(model,future,directory):
        directory.mkdir()
        for i in range(9):Image.new('RGB',(448,224)).save(directory/f'{i:03}.png')
    monkeypatch.setattr(m,'save_frames',frames)
    if bad_identity:
        with pytest.raises(ValueError,match='Identity'):m.run(tmp_path)
        assert json.loads((tmp_path/'run_status.json').read_text())['status']=='failed'
    else:
        m.run(tmp_path)
        result=json.loads((case/'seed_0_run/comparison.json').read_text())
        assert result['primary_target_projection_m']==pytest.approx(.01)
        for name in ['baseline','identity','donor_a','donor_b']:
            assert json.loads((case/f'seed_0_run/{name}.json').read_text())['recipient_instruction']=='original soup and tomato'
        assert p.calls==4
    assert p.closed
