"""Coherent free-joint relocations and native sensor evidence, without a WAM."""
from contextlib import AbstractContextManager
import numpy as np


def object_geometry_ids(model, body):
    descendants = []
    for child in range(model.nbody):
        node = child
        while node > 0:
            if node == body:
                descendants.append(child)
                break
            node = int(model.body_parentid[node])
    return np.flatnonzero(np.isin(model.geom_bodyid, descendants))


def relocate_free_objects(sim, body_ids, offsets, swap_pair=()):
    """Translate actual free joints, then recompute visual AND collision geometry.

    XY only: each object's own Z, quaternion, velocity and inertial parameters
    remain unchanged. Validate every edit before mutation; restore on failure.
    Offsets apply first, then the two resulting XY slots may be exchanged.
    """
    m, d = sim.model, sim.data
    if swap_pair and (len(swap_pair) != 2 or len(set(swap_pair)) != 2):
        raise ValueError('Swap requires two distinct objects')
    names = sorted(set(offsets) | set(swap_pair))
    if not names: raise ValueError('At least one declared object required')
    addresses, geometries = {}, {}
    for name in names:
        xy = np.asarray(offsets.get(name, [0, 0]), dtype=float)
        if xy.shape != (2,) or not np.isfinite(xy).all() or np.abs(xy).max() > .05:
            raise ValueError('Offsets must be finite XY values within 5 cm')
        body = body_ids[name]
        joints = np.flatnonzero(m.jnt_bodyid == body)
        if len(joints) != 1 or int(m.jnt_type[joints[0]]) != 0:
            raise ValueError('Object must have exactly one free joint')
        addresses[name] = int(m.jnt_qposadr[joints[0]])
        ids = object_geometry_ids(m, body)
        if not len(ids) or not np.any((m.geom_contype[ids] != 0) | (m.geom_conaffinity[ids] != 0)):
            raise ValueError('Object must include collision geometry')
        if not np.any(m.geom_group[ids] == 1): raise ValueError('Missing native visual geometry')
        geometries[name] = ids
    if len(set(addresses.values())) != len(names): raise ValueError('Aliased free joints')
    before = {k: np.array(getattr(d, k), copy=True) for k in
              ('qpos', 'qvel', 'geom_xpos', 'geom_xmat', 'body_xpos', 'cam_xpos', 'cam_xmat')}
    expected = before['qpos'].copy()
    for name, a in addresses.items(): expected[a:a+2] += offsets.get(name, [0, 0])
    if swap_pair:
        a, b = [addresses[n] for n in swap_pair]
        expected[a:a+2], expected[b:b+2] = expected[b:b+2].copy(), expected[a:a+2].copy()
    static = {k: np.array(getattr(m, k), copy=True) for k in
              ('geom_pos', 'geom_quat', 'geom_matid', 'geom_size', 'body_mass', 'body_inertia')}
    try:
        d.qpos[:] = expected
        sim.forward()
        if not np.allclose(d.qpos, expected, atol=1e-12, rtol=0): raise ValueError('Unexpected qpos changes')
        if not np.array_equal(d.qvel, before['qvel']): raise ValueError('Velocity changed')
        if any(not np.array_equal(getattr(m,k),v) for k,v in static.items()): raise ValueError('Static model changed')
        if any(not np.allclose(getattr(d,k),before[k],atol=1e-12,rtol=0) for k in ('cam_xpos','cam_xmat')):
            raise ValueError('Camera changed during relocation')
        expected_geom = before['geom_xpos'].copy()
        rows = []
        for name, a in addresses.items():
            delta = expected[a:a+3] - before['qpos'][a:a+3]
            ids = geometries[name]; expected_geom[ids] += delta
            if not np.allclose(d.body_xpos[body_ids[name]],before['body_xpos'][body_ids[name]]+delta,atol=1e-10,rtol=0):
                raise ValueError('Body did not follow free-joint translation')
            rows.append({'object':name,'qpos_address':a,'translation_m':delta.tolist(),
                         'geometry_ids':ids.tolist(),'collision_geometry_ids':ids[(m.geom_contype[ids]!=0)|(m.geom_conaffinity[ids]!=0)].tolist(),
                         'visual_geometry_ids':ids[m.geom_group[ids]==1].tolist()})
        error = float(np.max(np.abs(d.geom_xpos-expected_geom)))
        if error > 1e-10 or not np.allclose(d.geom_xmat,before['geom_xmat'],atol=1e-10,rtol=0):
            raise ValueError('Visual/collision geometry failed coherent translation')
        return {'objects':rows,'max_geometry_translation_error_m':error,'static_model_unchanged':True,
                'only_declared_xy_changed':True,'velocities_unchanged':True,'camera_unchanged':True}
    except BaseException:
        d.qpos[:] = before['qpos']; d.qvel[:] = before['qvel']; sim.forward()
        raise


def reset_relocated(env, initial_state, *, instruction, metadata, offsets, swap_pair=()):
    from wam_interp.evaluations.LIBERO.environment import reset_to_initial_state
    obs = reset_to_initial_state(env, initial_state, instruction=instruction, metadata=metadata)
    env._relocation_audit = relocate_free_objects(env.sim, env.env.obj_body_id, offsets, swap_pair)
    # The caller MUST take its declared warm-up steps before using observations.
    return obs


def paired_robot_start(a, b, tolerance=1e-12):
    """Distinguish floating-point solver roundoff from changed paired starts."""
    keys=('eef_position','eef_quaternion','gripper_position','ctrl')
    if not np.isfinite(tolerance) or tolerance < 0:raise ValueError('Invalid tolerance')
    differences={}
    for key in keys:
        x,y=np.asarray(a[key]),np.asarray(b[key])
        if x.shape!=y.shape or not np.isfinite(x).all() or not np.isfinite(y).all():raise ValueError('Invalid paired robot state')
        differences[key]=float(np.max(np.abs(x-y)))
    return {'exact':all(v==0 for v in differences.values()),'max_absolute_differences':differences,
            'absolute_tolerance':tolerance,'pass':all(v<=tolerance for v in differences.values())}


class NativeObjectCapture(AbstractContextManager):
    """Read segmentation at the native sensor timestamp; never alter scene assets."""
    def __init__(self, env, objects):
        import mujoco
        from robosuite import macros
        from robosuite.utils.mjcf_utils import IMAGE_CONVENTION_MAPPING
        self.sim = env.sim; self.enabled = False; self.latest = {}
        self.convention = IMAGE_CONVENTION_MAPPING[macros.IMAGE_CONVENTION]
        self.geom_type = int(mujoco.mjtObj.mjOBJ_GEOM)
        self.ids = {n:object_geometry_ids(self.sim.model,env.env.obj_body_id[n]) for n in objects}

    def __enter__(self):
        self.render = self.sim.render; self.sim.render = self._render; return self

    def __exit__(self,*exc): self.sim.render = self.render

    def _render(self,*args,**kwargs):
        native = self.render(*args,**kwargs); camera = kwargs.get('camera_name')
        if not self.enabled or camera not in ('agentview','robot0_eye_in_hand') or kwargs.get('depth') or kwargs.get('segmentation'):
            return native
        d = self.sim.data; before = {k:np.array(getattr(d,k),copy=True) for k in ('qpos','qvel','geom_xpos','cam_xpos','cam_xmat')}
        time = d.time; opts = {**kwargs,'depth':True,'segmentation':False}
        rgb,depth = self.render(*args,**opts)
        groups=self.sim._render_context_offscreen.vopt.sitegroup; saved=groups.copy()
        try:
            groups[:]=0; seg=self.render(*args,**{**kwargs,'depth':False,'segmentation':True})
        finally: groups[:]=saved
        if not np.array_equal(rgb,native) or not np.array_equal(native,self.render(*args,**kwargs)):
            raise ValueError('Native camera changed during audit')
        if d.time!=time or any(not np.array_equal(getattr(d,k),v) for k,v in before.items()):
            raise ValueError('Sensor audit mutated simulation')
        c=self.convention
        self.latest[camera]={'rgb':rgb[::c].copy(),'depth':depth[::c].copy(),'segmentation':seg[::c].copy(),
            'visible_pixels':{n:int(((seg[...,0]==self.geom_type)&np.isin(seg[...,1],ids)).sum()) for n,ids in self.ids.items()},
            'native_sensor_exact':True}
        return native
