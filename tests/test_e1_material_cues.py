import numpy as np
import pytest
from types import SimpleNamespace
from wam_interp.experiments.material_cues import swapped_materials,masked_registration,cue_interaction,robot_pixel_audit

def test_mixed_boundary_is_reported_but_real_robot_pixel_changes_are_detected():
    a=np.zeros((20,20,3),dtype=np.uint8);b=a.copy();robot=np.ones((20,20),bool);obj=np.zeros((20,20),bool);obj[10,10]=True
    b[10,11]=10;_,m=robot_pixel_audit(a,b,robot,obj)
    assert m['robot_object_boundary_changed']==1 and m['robot_interior_changed']==0
    b[2,2]=1;_,m=robot_pixel_audit(a,b,robot,obj)
    assert m['robot_interior_changed']==1

def test_material_transaction_restores_after_renderer_failure():
    a=np.array([10,20,30]);original=a.copy()
    with pytest.raises(RuntimeError):
        with swapped_materials(a,[0,2]):
            assert a.tolist()==[30,20,10]
            raise RuntimeError('render failed')
    np.testing.assert_array_equal(a,original)
    with pytest.raises(ValueError):
        with swapped_materials(a,[1,1]):pass

def test_sensor_wrapper_returns_native_image_and_restores_render_function():
    from wam_interp.experiments.simulator_materials import ObjectMaterialCapture
    materials=np.array([10,20,30]);model=SimpleNamespace(geom_matid=materials)
    def render(*args,**kwargs):
        h,w=kwargs['height'],kwargs['width'];seg=np.zeros((h,w,2),int);seg[...,0]=5;seg[...,1]=-1
        seg[8:12,8:12,1]=0;seg[18:22,18:22,1]=2;seg[3:6,24:28,1]=1
        if kwargs.get('segmentation'):return seg
        rgb=np.zeros((h,w,3),np.uint8);rgb[seg[...,1]==0]=materials[0];rgb[seg[...,1]==2]=materials[2];rgb[seg[...,1]==1]=100
        return (rgb,np.ones((h,w))) if kwargs.get('depth') else rgb
    sim=SimpleNamespace(model=model,data=SimpleNamespace(time=0.),render=render,
        _render_context_offscreen=SimpleNamespace(vopt=SimpleNamespace(sitegroup=np.zeros(6))))
    obj=ObjectMaterialCapture.__new__(ObjectMaterialCapture);obj.sim=sim;obj.enabled=True;obj.convention=1;obj.coverage_scale=1;obj.geom_type=5
    obj.geom_ids=np.array([0,2]);obj.material_ids=materials[obj.geom_ids].copy();obj.robot_geoms=np.array([1]);obj.gripper_geoms=np.array([1]);obj.latest={};obj.calls=0;obj.snapshot=lambda:{}
    expected=render(width=30,height=30,camera_name='agentview')
    with obj:
        actual=sim.render(width=30,height=30,camera_name='agentview')
        np.testing.assert_array_equal(actual,expected)
        assert not np.array_equal(obj.latest['agentview']['variants']['swapped'],expected)
    assert sim.render is render
    np.testing.assert_array_equal(materials,[10,20,30])

def test_gripper_registration_detects_motion_and_reports_missing_pixels():
    rng=np.random.default_rng(3);a=rng.uniform(0,255,(40,40,3));mask=np.zeros((40,40),bool);mask[10:30,10:30]=True
    assert masked_registration(a,a,mask)['shift_pixels']==0
    b=np.roll(a,2,axis=1);r=masked_registration(a,b,mask)
    assert r['best_shift_xy']==[2,0] and r['shift_pixels']==2
    assert not masked_registration(a,b,np.zeros_like(mask))['measurable']

def test_instruction_independent_sensitivity_does_not_create_semantic_interaction():
    e={(q,v):np.array([.1 if v=='swapped' else 0,0,0]) for q in ('soup','tomato','cream') for v in ('original','swapped')}
    r=cue_interaction(e,[1,0,0]);assert r['instruction_interaction_m']==0 and not r['directional_pair']

def test_opposite_noun_responses_and_missing_cells():
    e={(q,v):np.zeros(3) for q in ('soup','tomato','cream') for v in ('original','swapped')}
    e['soup','swapped']=np.array([.1,0,0]);e['tomato','swapped']=np.array([-.2,0,0])
    r=cue_interaction(e,[1,0,0]);assert r['instruction_interaction_m']==pytest.approx(.15) and r['directional_pair']
    del e['cream','swapped']
    with pytest.raises(ValueError):cue_interaction(e,[1,0,0])
