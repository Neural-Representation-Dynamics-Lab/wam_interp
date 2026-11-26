from copy import deepcopy
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from wam_interp.experiments.matching import latent_delta_norm, motion_features, choose_match
from wam_interp.experiments.e1 import run_e1
from test_e1 import inputs, Session


ROOT=Path(__file__).parents[1]
RULES=json.loads((ROOT/'configs/experiments/e1_replication.json').read_text())['matching']


def test_matching_rejects_large_change_despite_same_path():
    a={**motion_features([[i*10,20] for i in range(9)]),'latent_delta_norm':100.}
    b={**a,'latent_delta_norm':40.}
    assert choose_match({'same':b},a,RULES)['selected'] is None
    b['latent_delta_norm']=100.
    b['normalized_progress']=[0,0,0,0,0,0,.1,.2,1]
    assert choose_match({'same':b},a,RULES)['selected'] is None


def test_matching_selects_deterministically_and_excludes_primary():
    a={**motion_features([[i*10,20] for i in range(9)]),'latent_delta_norm':100.}
    result=choose_match({'z':a,'b':a,'primary':a},a,RULES,exclude=('primary',))
    assert result['selected']=='b'


def test_missing_landmark_is_not_interpolated():
    with pytest.raises(ValueError):motion_features([[0,0]]*8+[None])
    with pytest.raises(ValueError):motion_features([[0,0]]*9)


def test_latent_metric_ignores_observed_anchor():
    a=np.zeros((1,2,3,2,2));b=a.copy();b[:,:,0]=100
    assert latent_delta_norm(a,b)==0
    b[:,:,1:]=1
    assert latent_delta_norm(a,b)==4


def test_matched_arm_has_same_semantics_and_independent_restoration(tmp_path):
    manifest,donors=inputs()
    manifest['donors']['matched_same_property']=deepcopy(manifest['donors']['same_property'])
    donors['matched_same_property']=np.ones((2,1))*2
    session=Session()
    result=run_e1(session,donors=donors,manifest=manifest,output=tmp_path/'run')
    assert session.starts==[0.]*5 and session.state==0
    assert result['opposite_minus_matched_same']==-6
    manifest['donors']['matched_same_property']['label']='left'
    with pytest.raises(ValueError):run_e1(Session(),donors=donors,manifest=manifest,output=tmp_path/'bad')


def script(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT/'scripts'))
    spec=importlib.util.spec_from_file_location('e1_replication',ROOT/'scripts/e1_replication.py')
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m


def bank_fixture(tmp_path,monkeypatch):
    m=script(monkeypatch);bank=tmp_path/'bank';bank.mkdir()
    protocol=json.loads((ROOT/'configs/experiments/e1_replication.json').read_text())
    names=[protocol['same_primary'],protocol['alternative_primary'],'extra']
    baseline=np.zeros((1,1,3,1,1));np.savez_compressed(bank/'baseline_seed_0.npz',future=baseline)
    index=[];reviews={}
    for name,value in zip(names,[.4,1,1]):
        np.savez_compressed(bank/f'{name}.npz',future=np.ones_like(baseline)*value)
        index.append({'id':name,'sha256':m.digest_file(bank/f'{name}.npz')})
        reviews[name]={'label':protocol['alternative_label'] if name==protocol['alternative_primary'] else protocol['recipient_label'],
            'acceptable':True,'review_complete':True,'nuisance_notes':'test fixture',
            'landmarks_pixels':[[50+i*5,40+i] for i in range(9)],'landmarks_visually_verified':True}
    review={'reviewed_before_execution':True,'frame_mapping_verified':True,
        'primary_same':names[0],'primary_alternative':names[1],'candidates':reviews}
    m.write(bank/'candidate_index.json',index)
    m.write(bank/'recipient_contexts.json',{'0':{'baseline_sha256':m.digest_file(bank/'baseline_seed_0.npz'),
        'candidate_latent_delta_norms':{},'context':{}}})
    m.write(tmp_path/'selection_review.json',review)
    return m,protocol,review


def test_norm_eligible_unreviewed_candidate_cannot_be_silently_rejected(tmp_path,monkeypatch):
    m,p,r=bank_fixture(tmp_path,monkeypatch)
    assert m.reviewed_selection(tmp_path,p,0)[2]['selected']=='extra'
    r['candidates']['extra']['review_complete']=False
    m.write(tmp_path/'selection_review.json',r)
    with pytest.raises(ValueError,match='not been semantically reviewed'):m.reviewed_selection(tmp_path,p,0)


def test_modified_donor_rejected_before_matching(tmp_path,monkeypatch):
    m,p,r=bank_fixture(tmp_path,monkeypatch)
    np.savez_compressed(tmp_path/'bank/extra.npz',future=np.ones((1,1,3,1,1))*30)
    with pytest.raises(ValueError,match='hash changed'):m.reviewed_selection(tmp_path,p,0)


def test_primary_pair_cannot_be_changed_after_freeze(tmp_path,monkeypatch):
    m,p,r=bank_fixture(tmp_path,monkeypatch)
    r['primary_same']='extra';m.write(tmp_path/'selection_review.json',r)
    with pytest.raises(ValueError,match='Primary pair changed'):m.reviewed_selection(tmp_path,p,0)


def test_contact_rollout_replans_from_current_observation_and_records_motion(tmp_path,monkeypatch):
    from types import SimpleNamespace
    import torch
    from PIL import Image
    from wam_interp.types import EnvironmentObservation, PreparedObservation, InferenceRequest, InferenceResult
    m=script(monkeypatch)
    def obs(x):
        return EnvironmentObservation({'agentview':np.zeros((8,8,3),np.uint8),'wrist':np.zeros((8,8,3),np.uint8)},
            {'eef_position':np.array([x,0,0]),'eef_quaternion':np.array([0,0,0,1]),'gripper_position':np.zeros(2)},'pick')
    adapter=SimpleNamespace(model=object(),
        prepare_observation=lambda o:PreparedObservation(torch.zeros(1,3,224,448),o.state['eef_position'],'pick'),
        postprocess_actions=lambda a,environment:a)
    env=SimpleNamespace(value=0.,sim=SimpleNamespace(model=SimpleNamespace(body_names=['alphabet_soup_1_main'])))
    p=SimpleNamespace(adapter=adapter,observation=obs(0.),task=SimpleNamespace(language='pick'),env=env,
        verify_replay=lambda:None,close=lambda:None)
    p.request=InferenceRequest(adapter.prepare_observation(p.observation),condition='idm',seed=0)
    def step(env,command,**kw):
        env.value+=.001
        return SimpleNamespace(observation=obs(env.value),done=False,success=False)
    p.step_env=step
    seen=[]
    def predict(adapter,request):
        seen.append(float(request.observation.proprio[0]))
        return InferenceResult(np.ones((32,7))),{'used':np.zeros((1,1,3,1,1))}
    def frames(model,latents,folder):
        folder.mkdir()
        for i in range(9):Image.new('RGB',(448,224)).save(folder/f'{i:03}.png')
    monkeypatch.setattr(m,'new_pilot',lambda config,loaded:p)
    monkeypatch.setattr(m,'provenance',lambda pilot:{'test_fixture':True})
    monkeypatch.setattr(m,'physical_state',lambda env,names:({'alphabet_soup_1_main':np.array([env.value,0,0])},[]))
    monkeypatch.setattr(m,'predict_with_future',predict)
    monkeypatch.setattr(m,'save_frames',frames)
    monkeypatch.setattr(torch.cuda,'synchronize',lambda:None)
    protocol={'contact_followup':{'initial_states':[3],'action_seed':0,'max_chunks':2,'execute_steps':32,'object_motion_threshold_m':.01}}
    m.contact_rollouts(tmp_path,protocol,{'frame_steps':list(range(0,33,4))})
    assert seen==pytest.approx([0.,.032])
    with np.load(tmp_path/'contact_state_3/chunk_01/execution.npz') as a:
        assert a['control_steps'].tolist()==list(range(32,65))
        assert a['object_alphabet_soup_1_main'][-1,0]==pytest.approx(.064)
    summary=json.loads((tmp_path/'contact_state_3/chunk_01/summary.json').read_text())
    assert summary['object_motion_above_threshold']==['alphabet_soup_1_main']
