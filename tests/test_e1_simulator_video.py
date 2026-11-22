from types import SimpleNamespace
import numpy as np
import pytest
from wam_interp.experiments.simulator_video import approach_command,appearance_metrics,approach_metrics,valid_approach,paired_camera_clip,encode_reference_video
from wam_interp.experiments.simulator_libero import AppearanceCapture


def test_approach_is_bounded_keeps_gripper_open_and_rejects_bad_geometry():
    command=approach_command(np.zeros(3),np.zeros(3),np.ones(3),32)
    assert np.array_equal(command[:3],[.4]*3)
    assert np.array_equal(command[3:],[0,0,0,-1])
    with pytest.raises(ValueError):approach_command([np.nan]*3,[0]*3,[1]*3,1)


def test_appearance_rejects_pixels_outside_mask_and_detects_depth_changes():
    a=np.zeros((10,10,3),dtype=np.uint8);b=a.copy();mask=np.zeros((10,10),bool);mask[4:6,4:6]=True;depth=np.ones((10,10))
    b[4,4]=10;m=appearance_metrics(a,b,mask,depth,depth)
    assert m['depth_exact'] and m['changed_outside_boundary_support']==0
    b[0,0]=1;other=depth.copy();other[5,5]=.5
    m=appearance_metrics(a,b,mask,depth,other)
    assert m['changed_outside_boundary_support']==1 and not m['depth_exact']


def test_geometric_gate_requires_target_specific_approach_and_no_contacts():
    eef=np.zeros((33,3));eef[-1]=[0,0,.14]
    objects=np.zeros((33,2,3));objects[:,1,1]=.2
    limits={'endpoint_goal_error_m':.02,'maximum_object_translation_m':.001,'target_margin_m':.1}
    m=approach_metrics(eef,objects,0,1,[0,0,.14],0)
    assert valid_approach(m,limits)
    assert not valid_approach({**m,'robot_scene_contact_substeps':1},limits)
    assert not valid_approach({**m,'endpoint_other_xy_distance_m':.01},limits)


def test_camera_rotation_and_order_preserve_native_convention():
    a=np.zeros((9,224,224,3),np.uint8);b=np.zeros_like(a)
    a[:,:112,:112,0]=255;b[...,2]=200
    clip=paired_camera_clip(a,b)
    assert clip.shape==(9,224,448,3)
    assert clip[0,200,200,0]==255 and clip[0,20,20,0]==0
    assert clip[0,20,300,2]==200


def fake_capture(fail=False):
    sim=SimpleNamespace(model=SimpleNamespace(mat_rgba=np.ones((1,4))),
        data=SimpleNamespace(time=1.,**{k:np.zeros((1,3)) for k in ('qpos','qvel','ctrl','body_xpos','body_xquat','cam_xpos','cam_xmat')}),
        _render_context_offscreen=SimpleNamespace(vopt=SimpleNamespace(sitegroup=np.array([1,1,1,0,0,0]))))
    def render(*args,**kw):
        if fail and sim.model.mat_rgba[0,0]<1:raise RuntimeError('renderer failure')
        if kw.get('segmentation'):return np.tile([5,2],(4,4,1))
        rgb=np.full((4,4,3),int(100*sim.model.mat_rgba[0,0]),np.uint8)
        return (rgb,np.ones((4,4))) if kw.get('depth') else rgb
    sim.render=render
    c=object.__new__(AppearanceCapture);c.sim=sim;c.enabled=True;c.latest={};c.calls=0;c.coverage_scale=1
    c.material_id=0;c.factors={'edited':np.array([.9,1,.9])};c.geom_ids=np.array([2]);c.geom_type=5;c.convention=-1;c.original_rgba=np.ones(4)
    return c,sim,render


def test_capture_returns_original_and_restores_materials_and_render_method():
    c,sim,render=fake_capture()
    with c:
        output=sim.render(4,4,camera_name='agentview')
        assert np.all(output==100) and np.all(c.latest['agentview']['variants']['edited']==90)
        assert c.latest['agentview']['sham_exact']
    assert sim.render is render and np.all(sim.model.mat_rgba==1)
    assert np.array_equal(sim._render_context_offscreen.vopt.sitegroup,[1,1,1,0,0,0])


def test_render_failure_restores_material_and_wrapper():
    c,sim,render=fake_capture(True)
    with pytest.raises(RuntimeError,match='renderer failure'):
        with c:sim.render(4,4,camera_name='agentview')
    assert sim.render is render and np.all(sim.model.mat_rgba==1)


def test_reference_encoding_sets_current_image_without_mutating_future_input():
    import torch
    class Codec:
        calls=0
        def encode(self,rgb,scale):
            assert rgb.shape==(1,3,9,224,448)
            assert torch.allclose(rgb[:,:,0],torch.full_like(rgb[:,:,0],20*2/255-1))
            self.calls+=1
            return rgb[:,:,::4].clone()
        def decode(self,z,scale):return z.repeat_interleave(4,dim=2)[:,:,:9]
    codec=Codec();model=SimpleNamespace(vae=SimpleNamespace(model=codec,scale=[0,1]),device='cpu',torch_dtype=torch.float32)
    clip=np.full((9,224,448,3),100,np.uint8);observed=np.full((224,448,3),20,np.uint8);anchor=np.zeros((1,3,3,224,448),np.float32)
    arrays,meta=encode_reference_video(model,clip,observed,anchor)
    assert codec.calls==2 and meta['repeated_encoding_exact'] and meta['input_text_context'] is None
    assert np.all(clip==100) and np.all(arrays['source_rgb'][0]==20)
    assert np.array_equal(arrays['future'][:,:,:1],anchor[:,:,:1])
    assert np.allclose(arrays['future'][:,:,1:],100*2/255-1)
