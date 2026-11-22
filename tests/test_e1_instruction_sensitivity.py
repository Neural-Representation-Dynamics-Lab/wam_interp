import numpy as np
import pytest
from wam_interp.experiments.instruction_sensitivity import balanced_contrasts, future_distance, semantic_pair


def bank():
    return [dict(state=s,target=t,wording=w,seed=n,id=f'{s}_{t}_{w}_{n}')
            for s in (3,4,5) for t in ('cream','butter') for w in ('pick','put') for n in (18,19,20,21)]


def test_balanced_pairs_hold_other_factors_fixed():
    records=bank(); lookup={r['id']:r for r in records}; pairs=balanced_contrasts(records)
    assert {k:sum(p['kind']==k for p in pairs) for k in ('target_word','action_wording','seed')} == {
        'target_word':24,'action_wording':24,'seed':72}
    for p in pairs:
        a,b=lookup[p['a']],lookup[p['b']]
        changed=[k for k in ('state','target','wording','seed') if a[k]!=b[k]]
        assert changed == [{'target_word':'target','action_wording':'wording','seed':'seed'}[p['kind']]]


@pytest.mark.parametrize('mode',['missing','duplicate'])
def test_unbalanced_bank_fails(mode):
    r=bank()
    with pytest.raises(ValueError):balanced_contrasts(r[:-1] if mode=='missing' else r+[r[0]])


def test_distance_excludes_anchor_and_rejects_changed_observation():
    a=np.zeros((1,2,3,2,2)); b=a.copy(); b[:,:,1:]=2
    assert future_distance(a,b)['latent_rmse']==2
    b[:,:,0]=1
    with pytest.raises(ValueError,match='anchor'):future_distance(a,b)


def test_nonfinite_and_wrong_shape_rejected():
    a=np.zeros((1,2,3,2,2)); b=a.copy(); b[:,:,1]=np.nan
    with pytest.raises(ValueError,match='Nonfinite'):future_distance(a,b)
    with pytest.raises(ValueError,match='matching'):future_distance(a,a[0])


def test_unclear_is_not_a_semantic_switch():
    a=dict(target='cream',label='cream',all_nine_frames_reviewed=True,notes='fixture')
    b=dict(target='butter',label='unclear',all_nine_frames_reviewed=True,notes='fixture')
    assert not semantic_pair(a,b)['different_clear_targets']
    b['label']='butter'
    assert semantic_pair(a,b)['both_follow_intended_target']
    b['all_nine_frames_reviewed']=False
    with pytest.raises(ValueError):semantic_pair(a,b)
