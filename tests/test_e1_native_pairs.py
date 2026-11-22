import numpy as np
import pytest
from wam_interp.experiments.native_pairs import rank_pairs,select_pair,target_direction

RULES={'latent_delta_relative_tolerance':.2,'path_length_relative_tolerance':.15,
       'endpoint_displacement_relative_tolerance':.15,'normalized_progress_max_absolute_difference':.15}


def fixture():
    norms={'0':{'s':10.,'a':10.},'1':{'s':11.,'a':11.}}
    rank=rank_pairs({'s':'soup','a':'cream'},norms,RULES,8)
    reviews={name:{'all_nine_frames_reviewed':True,'notes':'visual test fixture','label':label,
        'acceptable':True,'landmarks_verified':True,'landmarks_pixels':[[20+5*i,30] for i in range(9)]}
        for name,label in [('s','soup'),('a','cream')]}
    nuisance={'s__a':{'review_complete':True,'acceptable':True,'notes':'fixture orientation comparable'}}
    return norms,rank,reviews,nuisance


def test_both_recipient_seeds_must_pass_norm_gate():
    norms,_,_,_=fixture();norms['1']['s']=2
    assert not rank_pairs({'s':'soup','a':'cream'},norms,RULES,8)['shortlist']


def test_ranking_ties_and_unreviewed_denominator():
    rank=rank_pairs({'s2':'soup','s1':'soup','a':'cream'},{'0':{'s1':10,'s2':10,'a':10}},RULES,1)
    assert rank['shortlist'][0]['same']=='s1'
    assert rank['unreviewed_norm_eligible_pairs']==1


def test_prompt_target_does_not_override_visual_rejection():
    norms,rank,reviews,nuisance=fixture();reviews['a']['label']='soup'
    assert select_pair(rank,reviews,norms,RULES,nuisance)['selected'] is None


def test_pair_requires_nuisance_review():
    norms,rank,reviews,nuisance=fixture()
    with pytest.raises(ValueError,match='orientation'):select_pair(rank,reviews,norms,RULES,{})
    assert select_pair(rank,reviews,norms,RULES,nuisance)['selected']['same']=='s'


def test_unreviewed_candidate_is_not_silently_excluded():
    norms,rank,reviews,nuisance=fixture();reviews['a']['all_nine_frames_reviewed']=False
    with pytest.raises(ValueError,match='Incomplete'):select_pair(rank,reviews,norms,RULES,nuisance)


def test_missing_landmarks_cannot_pass():
    norms,rank,reviews,nuisance=fixture();reviews['a']['landmarks_pixels'][3]=None
    with pytest.raises(ValueError):select_pair(rank,reviews,norms,RULES,nuisance)


def test_target_projection_uses_initial_world_xy_and_sign():
    direction=target_direction([1,2,8],[4,6,-10])
    assert direction==pytest.approx([.6,.8,0])
    assert np.dot([.03,.04,99],direction)==pytest.approx(.05)
    with pytest.raises(ValueError):target_direction([0,0,1],[0,0,2])


def test_two_alternative_target_labels_are_explicit():
    norms,_,reviews,nuisance=fixture()
    reviews['s']['label']='cream';reviews['a']['label']='butter'
    rank=rank_pairs({'s':'cream','a':'butter'},norms,RULES,8,labels=('cream','butter'))
    assert select_pair(rank,reviews,norms,RULES,nuisance,labels=('cream','butter'))['selected']
    with pytest.raises(ValueError):rank_pairs({'s':'cream','a':'butter'},norms,RULES,8,labels=('cream','cream'))
