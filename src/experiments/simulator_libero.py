"""Capture appearance variants at the exact native camera-sensor invocation.

Rendering after env.step can differ from the cached observation's timestamp.
This wrapper renders all variants without stepping physics, returns the original
image to the simulator, and retains segmentation/depth and an exact sham render.
"""
from contextlib import AbstractContextManager
import numpy as np


class AppearanceCapture(AbstractContextManager):
    def __init__(self,env,*,material_name,factors,coverage_scale=4):
        import mujoco
        from robosuite import macros
        from robosuite.utils.mjcf_utils import IMAGE_CONVENTION_MAPPING
        self.env=env;self.sim=env.sim;self.material_name=material_name
        if type(coverage_scale) is not int or coverage_scale not in (1,2,4):raise ValueError('Coverage scale must be 1, 2 or 4')
        self.coverage_scale=coverage_scale
        self.factors={k:np.asarray(v,dtype=float) for k,v in factors.items()}
        if any(v.shape!=(3,) or not np.isfinite(v).all() or (v<=0).any() or (v>1).any() for v in self.factors.values()):
            raise ValueError('Appearance factors must be finite RGB multipliers in (0,1]')
        self.material_id=mujoco.mj_name2id(self.sim.model._model,mujoco.mjtObj.mjOBJ_MATERIAL,material_name)
        if self.material_id<0:raise ValueError('Missing appearance material')
        self.geom_ids=np.flatnonzero(self.sim.model.geom_matid==self.material_id)
        if len(self.geom_ids)!=1:raise ValueError('Initial experiment requires one isolated visual geometry')
        self.geom_type=int(mujoco.mjtObj.mjOBJ_GEOM)
        self.convention=IMAGE_CONVENTION_MAPPING[macros.IMAGE_CONVENTION]
        self.original_rgba=self.sim.model.mat_rgba[self.material_id].copy()
        groups=self.sim._render_context_offscreen.vopt.sitegroup
        if np.any((self.sim.model.site_rgba[:,3]>0)&(groups[self.sim.model.site_group]>0)):
            raise ValueError('Visible diagnostic sites require an explicit segmentation policy')
        self.enabled=False;self.latest={};self.calls=0

    def __enter__(self):
        self.render=self.sim.render;self.sim.render=self._render
        return self

    def __exit__(self,*exc):
        self.sim.model.mat_rgba[self.material_id]=self.original_rgba
        self.sim.render=self.render

    def snapshot(self):
        d=self.sim.data
        return {k:np.array(getattr(d,k),copy=True) for k in ('qpos','qvel','ctrl','body_xpos','body_xquat','cam_xpos','cam_xmat')}

    def _render(self,*args,**kwargs):
        image=self.render(*args,**kwargs)
        cam=kwargs.get('camera_name')
        if not self.enabled or cam not in ('agentview','robot0_eye_in_hand') or kwargs.get('depth') or kwargs.get('segmentation'):
            return image
        before=self.snapshot();time=float(self.sim.data.time)
        opts=dict(kwargs);opts['depth']=True;opts['segmentation']=False
        rgb,depth=self.render(*args,**opts)
        if not np.array_equal(rgb,image):raise RuntimeError('Repeated original render differs')
        segopts={**opts,'depth':False,'segmentation':True}
        # MuJoCo ID rendering includes zero/negative-alpha diagnostic sites that
        # are invisible in RGB. Hide those sites for segmentation only, otherwise
        # they falsely punch holes in the tabletop mask. Native RGB is unchanged.
        groups=self.sim._render_context_offscreen.vopt.sitegroup
        saved_groups=groups.copy()
        try:
            groups[:]=0
            seg=self.render(*args,**segopts)
            if self.coverage_scale>1:
                h,w=image.shape[:2];scale=self.coverage_scale
                high=self.render(**{**segopts,'width':w*scale,'height':h*scale})
                high_mask=(high[...,0]==self.geom_type)&np.isin(high[...,1],self.geom_ids)
                coverage=high_mask.reshape(h,scale,w,scale).any(axis=(1,3))
            else:
                coverage=(seg[...,0]==self.geom_type)&np.isin(seg[...,1],self.geom_ids)
        finally:
            groups[:]=saved_groups
        mask=coverage
        variants={};depths={}
        try:
            for name,factor in self.factors.items():
                self.sim.model.mat_rgba[self.material_id,:3]=self.original_rgba[:3]*factor
                variants[name],depths[name]=self.render(*args,**opts)
        finally:
            self.sim.model.mat_rgba[self.material_id]=self.original_rgba
        sham=self.render(*args,**kwargs)
        if not np.array_equal(image,sham):raise RuntimeError('Appearance restoration changed render')
        after=self.snapshot()
        if any(not np.array_equal(before[k],after[k]) for k in before) or self.sim.data.time!=time:
            raise RuntimeError('Rendering changed physics or camera geometry')
        conv=self.convention
        self.latest[cam]={'rgb':rgb[::conv].copy(),'depth':depth[::conv].copy(),
            'mask':mask[::conv].copy(),'segmentation':seg[::conv].copy(),
            'variants':{k:v[::conv].copy() for k,v in variants.items()},
            'variant_depths':{k:v[::conv].copy() for k,v in depths.items()},
            'sensor_time':time,'sensor_state':before,'sham_exact':True}
        self.calls+=1
        return image


class RobotContactCapture(AbstractContextManager):
    """Inspect every integration step, not only contacts at control boundaries."""
    def __init__(self,env):
        import mujoco
        self.sim=env.sim;self.events=[];self.enabled=False;self.substeps=0
        native=self.sim.model._model
        roots=[mujoco.mj_name2id(native,mujoco.mjtObj.mjOBJ_BODY,n) for n in ('robot0_base','gripper0_right_gripper')]
        robot=set()
        for body in range(self.sim.model.nbody):
            node=body
            while node>0:
                if node in roots:robot.add(body);break
                node=int(self.sim.model.body_parentid[node])
        self.robot_geoms=set(np.flatnonzero(np.isin(self.sim.model.geom_bodyid,list(robot))).tolist())
        scene_bodies=set(env.env.obj_body_id.values())|{mujoco.mj_name2id(native,mujoco.mjtObj.mjOBJ_BODY,'living_room_table_col')}
        self.scene_geoms=set(np.flatnonzero(np.isin(self.sim.model.geom_bodyid,list(scene_bodies))).tolist())

    def __enter__(self):
        self.step=self.sim.step;self.sim.step=self._step;return self

    def __exit__(self,*exc):self.sim.step=self.step

    def _step(self,*args,**kwargs):
        result=self.step(*args,**kwargs)
        if self.enabled:
            self.substeps+=1
            pairs=[]
            for c in self.sim.data.contact[:self.sim.data.ncon]:
                a,b=int(c.geom1),int(c.geom2)
                if (a in self.robot_geoms and b in self.scene_geoms) or (b in self.robot_geoms and a in self.scene_geoms):pairs.append([a,b,float(c.dist)])
            if pairs:self.events.append({'time':float(self.sim.data.time),'pairs':pairs})
        return result
