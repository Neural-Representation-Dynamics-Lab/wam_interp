from types import SimpleNamespace as NS
import numpy as np
import pytest
from wam_interp.experiments.physical_relocation import relocate_free_objects, NativeObjectCapture


def fake_scene():
    m=NS(nbody=4,body_parentid=np.array([0,0,0,1]),jnt_bodyid=np.array([1,2]),jnt_type=np.zeros(2,int),jnt_qposadr=np.array([0,7]),
        geom_bodyid=np.array([1,3,2,2]),geom_contype=np.array([1,0,1,0]),geom_conaffinity=np.array([1,0,1,0]),geom_group=np.array([0,1,0,1]),
        geom_pos=np.zeros((4,3)),geom_quat=np.tile([1.,0,0,0],(4,1)),geom_matid=np.arange(4),geom_size=np.ones((4,3)),body_mass=np.ones(4),body_inertia=np.ones((4,3)))
    q=np.array([-.2,-.1,.04,1,0,0,0,.2,.1,.07,1,0,0,0.])
    d=NS(qpos=q,qvel=np.arange(12,dtype=float),geom_xpos=np.zeros((4,3)),geom_xmat=np.tile(np.eye(3).ravel(),(4,1)),
         body_xpos=np.zeros((4,3)),cam_xpos=np.ones((2,3)),cam_xmat=np.ones((2,9)))
    def forward():
        d.body_xpos[1]=q[:3];d.body_xpos[2]=q[7:10];d.body_xpos[3]=q[:3]+[.01,0,0]
        d.geom_xpos[:]=d.body_xpos[m.geom_bodyid]
    sim=NS(model=m,data=d,forward=forward);forward();return sim


def test_physical_swap_moves_collision_and_nested_visual_geom_together():
    sim=fake_scene();before=sim.data.qpos.copy();vel=sim.data.qvel.copy()
    audit=relocate_free_objects(sim,{'soup':1,'tomato':2},{'soup':[.01,0],'tomato':[0,-.02]},('soup','tomato'))
    np.testing.assert_allclose(sim.data.qpos[:2],[.2,.08])
    np.testing.assert_allclose(sim.data.qpos[7:9],[-.19,-.1])
    np.testing.assert_array_equal(sim.data.qpos[2:7],before[2:7]);np.testing.assert_array_equal(sim.data.qpos[9:],before[9:])
    np.testing.assert_array_equal(sim.data.qvel,vel)
    np.testing.assert_allclose(sim.data.geom_xpos[1]-sim.data.geom_xpos[0],[.01,0,0])
    assert audit['max_geometry_translation_error_m']<1e-12
    assert all(x['collision_geometry_ids'] and x['visual_geometry_ids'] for x in audit['objects'])


def test_invalid_second_object_does_not_partially_mutate_scene():
    sim=fake_scene();old=sim.data.qpos.copy()
    with pytest.raises(ValueError):relocate_free_objects(sim,{'a':1,'b':2},{'a':[.01,0],'b':[np.nan,0]},('a','b'))
    np.testing.assert_array_equal(sim.data.qpos,old)
    with pytest.raises(ValueError):relocate_free_objects(sim,{'a':1},{'a':[0,0]},('a','a'))


def test_audit_failure_restores_physics_not_only_visuals():
    sim=fake_scene();old=sim.data.qpos.copy();native=sim.forward;calls=[]
    def broken_once():
        native();calls.append(1)
        if len(calls)==1:sim.data.geom_xpos[1,0]+=.1
    sim.forward=broken_once
    with pytest.raises(ValueError,match='coherent translation'):
        relocate_free_objects(sim,{'a':1,'b':2},{'a':[0,0]},('a','b'))
    np.testing.assert_array_equal(sim.data.qpos,old)
    np.testing.assert_allclose(sim.data.geom_xpos[1],old[:3]+[.01,0,0])


def test_native_capture_restores_render_function_on_exception():
    obj=NativeObjectCapture.__new__(NativeObjectCapture)
    render=lambda **kw:None;obj.sim=NS(render=render)
    with pytest.raises(RuntimeError):
        with obj:raise RuntimeError('sensor failure')
    assert obj.sim.render is render


def test_paired_start_accepts_roundoff_but_rejects_physical_drift():
    from wam_interp.experiments.physical_relocation import paired_robot_start
    a={k:np.zeros(3) for k in ('eef_position','eef_quaternion','gripper_position','ctrl')}
    b={k:v.copy() for k,v in a.items()};b['ctrl'][0]=1e-15
    assert paired_robot_start(a,b)['pass'] and not paired_robot_start(a,b)['exact']
    b['eef_position'][0]=1e-7
    assert not paired_robot_start(a,b)['pass']
