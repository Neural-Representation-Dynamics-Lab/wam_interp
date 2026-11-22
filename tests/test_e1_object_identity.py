import numpy as np
import pytest
from wam_interp.experiments.object_identity import exchanged_visual_positions, identity_reversal


def test_asset_exchange_preserves_height_and_restores_after_failure():
    positions = np.array([[0., 1., .2], [9., 9., 9.], [2., 3., .3]])
    old = positions.copy()
    with pytest.raises(RuntimeError):
        with exchanged_visual_positions(positions, [0, 2], [[0, 1, .1], [2, 3, .9]]):
            np.testing.assert_array_equal(positions[[0, 2]], [[2, 3, .2], [0, 1, .3]])
            np.testing.assert_array_equal(positions[1], old[1])
            raise RuntimeError('renderer failed')
    np.testing.assert_array_equal(positions, old)
    with pytest.raises(ValueError):
        with exchanged_visual_positions(positions, [0, 0], [[0, 1, 0], [2, 3, 0]]): pass


def test_identity_criterion_rejects_generic_motion_and_wrong_noun():
    e = {(q, v): np.array([0., 0., 0.]) for q in ('soup', 'tomato', 'cream') for v in ('original', 'swapped')}
    e['tomato', 'original'][0] = .2
    e['tomato', 'swapped'][0] = .2
    assert not identity_reversal(e, [1, 0, 0])['pass']
    e['soup', 'swapped'][0] = .2
    e['tomato', 'swapped'][0] = 0
    assert identity_reversal(e, [1, 0, 0])['pass']
    e['cream', 'swapped'][0] = .03
    assert not identity_reversal(e, [1, 0, 0])['criteria']['untouched_target_stable']
    del e['cream', 'original']
    with pytest.raises(ValueError): identity_reversal(e, [1, 0, 0])


def test_layout_reset_edits_only_declared_free_joint_xy(monkeypatch):
    from types import SimpleNamespace as NS
    from wam_interp.evaluations.LIBERO import environment
    from wam_interp.experiments.simulator_identity import reset_with_layout
    observation=object();monkeypatch.setattr(environment,'reset_to_initial_state',lambda *a,**k:observation)
    qpos=np.arange(14,dtype=float);old=qpos.copy();calls=[]
    model=NS(jnt_bodyid=np.array([10,20]),jnt_type=np.zeros(2,int),jnt_qposadr=np.array([0,7]))
    env=NS(sim=NS(model=model,data=NS(qpos=qpos),forward=lambda:calls.append(True)),env=NS(obj_body_id={'soup':20}))
    assert reset_with_layout(env,None,instruction='q',metadata={},offsets={'soup':[.02,-.03]}) is observation
    expected=old.copy();expected[7:9]+=[.02,-.03]
    np.testing.assert_array_equal(qpos,expected);assert calls==[True]


def test_asset_sensor_capture_returns_native_and_restores_all_visual_state():
    from types import SimpleNamespace as NS
    from wam_interp.experiments.simulator_identity import ObjectAssetCapture
    positions=np.array([[8.,8.,.1],[30.,5.,.4],[22.,22.,.2]])
    materials=np.array([10,100,30]);initial=positions.copy()
    data=NS(time=0.,geom_xpos=positions,geom_xmat=np.tile(np.eye(3).ravel(),(3,1)),body_xpos=positions.copy())
    model=NS(geom_matid=materials,geom_bodyid=np.arange(3))
    def render(*args,**kw):
        h,w=kw['height'],kw['width'];scale=h//40;seg=np.zeros((h,w,2),int);seg[...,0]=5;seg[...,1]=-1
        for i,(x,y,z) in enumerate(positions):
            x,y=int(x)*scale,int(y)*scale;seg[y:y+3*scale,x:x+3*scale,1]=i
        if kw.get('segmentation'):return seg
        rgb=np.zeros((h,w,3),np.uint8);depth=np.ones((h,w))
        for i in range(3):rgb[seg[...,1]==i]=materials[i];depth[seg[...,1]==i]=positions[i,2]
        return (rgb,depth) if kw.get('depth') else rgb
    sim=NS(model=model,data=data,render=render,_render_context_offscreen=NS(vopt=NS(sitegroup=np.zeros(6))))
    obj=ObjectAssetCapture.__new__(ObjectAssetCapture)
    obj.sim=sim;obj.enabled=True;obj.convention=1;obj.coverage_scale=1;obj.geom_type=5;obj.geom_ids=np.array([0,2])
    obj.material_ids=materials[[0,2]].copy();obj.robot_geoms=np.array([1]);obj.gripper_geoms=np.array([1]);obj.latest={};obj.calls=0;obj.snapshot=lambda:{}
    native=render(width=40,height=40,camera_name='agentview')
    with obj:
        np.testing.assert_array_equal(sim.render(width=40,height=40,camera_name='agentview'),native)
        captured=obj.latest['agentview'];assert not np.array_equal(captured['variants']['swapped'],native)
        assert captured['asset_diagnostics']['outside_support_depth_exact']
        assert captured['asset_diagnostics']['robot_interior_rgb_exact']
    np.testing.assert_array_equal(positions,initial);np.testing.assert_array_equal(materials,[10,100,30]);assert sim.render is render
