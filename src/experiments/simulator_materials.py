"""Swap two native object materials inside sensor renders, never physics steps."""
from contextlib import AbstractContextManager
import numpy as np
from .material_cues import swapped_materials,robot_pixel_audit
from .simulator_video import dilate
from .simulator_libero import AppearanceCapture


class ObjectMaterialCapture(AbstractContextManager):
    def __init__(self,env,objects,*,coverage_scale=4):
        import mujoco
        from robosuite import macros
        from robosuite.utils.mjcf_utils import IMAGE_CONVENTION_MAPPING
        if len(objects)!=2 or len(set(objects))!=2:raise ValueError('Two distinct objects required')
        self.env=env;self.sim=env.sim;self.coverage_scale=coverage_scale
        self.convention=IMAGE_CONVENTION_MAPPING[macros.IMAGE_CONVENTION]
        self.geom_type=int(mujoco.mjtObj.mjOBJ_GEOM);self.geom_ids=[]
        model=self.sim.model
        for name in objects:
            body=env.env.obj_body_id[name]
            ids=np.flatnonzero((model.geom_bodyid==body)&(model.geom_group==1))
            if len(ids)!=1:raise ValueError('Each object needs exactly one isolated visual geometry')
            i=int(ids[0])
            if model.geom_contype[i] or model.geom_conaffinity[i]:raise ValueError('Visual geometry participates in collisions')
            self.geom_ids.append(i)
        self.geom_ids=np.array(self.geom_ids);self.material_ids=model.geom_matid[self.geom_ids].copy()
        if len(set(self.material_ids.tolist()))!=2 or (self.material_ids<0).any():raise ValueError('Distinct native materials required')
        self.material_id=-1  # Multiple materials; explicit IDs are recorded by the caller.
        native=model._model
        def descendants(root_names):
            roots={mujoco.mj_name2id(native,mujoco.mjtObj.mjOBJ_BODY,n) for n in root_names};roots.discard(-1)
            if not roots:raise ValueError('Missing robot/gripper body')
            bodies=[]
            for b in range(model.nbody):
                node=b
                while node>0:
                    if node in roots:bodies.append(b);break
                    node=int(model.body_parentid[node])
            return np.flatnonzero(np.isin(model.geom_bodyid,bodies))
        self.robot_geoms=descendants(['robot0_base','gripper0_right_gripper'])
        self.gripper_geoms=descendants(['gripper0_right_gripper'])
        self.enabled=False;self.latest={};self.calls=0
        groups=self.sim._render_context_offscreen.vopt.sitegroup
        if np.any((model.site_rgba[:,3]>0)&(groups[model.site_group]>0)):raise ValueError('Visible diagnostic sites unsupported')

    snapshot=AppearanceCapture.snapshot

    def __enter__(self):
        self.render=self.sim.render;self.sim.render=self._render;return self

    def __exit__(self,*exc):
        self.sim.model.geom_matid[self.geom_ids]=self.material_ids;self.sim.render=self.render

    def _render(self,*args,**kwargs):
        image=self.render(*args,**kwargs);cam=kwargs.get('camera_name')
        if not self.enabled or cam not in ('agentview','robot0_eye_in_hand') or kwargs.get('depth') or kwargs.get('segmentation'):return image
        before=self.snapshot();time=float(self.sim.data.time)
        opts={**kwargs,'depth':True,'segmentation':False};rgb,depth=self.render(*args,**opts)
        if not np.array_equal(image,rgb):raise RuntimeError('Native render drift')
        groups=self.sim._render_context_offscreen.vopt.sitegroup;saved=groups.copy()
        try:
            groups[:]=0;segopts={**kwargs,'depth':False,'segmentation':True}
            seg=self.render(*args,**segopts);h,w=image.shape[:2];s=self.coverage_scale
            high=self.render(**{**segopts,'height':h*s,'width':w*s})
            def mask(ids):return ((high[...,0]==self.geom_type)&np.isin(high[...,1],ids)).reshape(h,s,w,s).any(axis=(1,3))
            edit_mask=mask(self.geom_ids);robot_mask=mask(self.robot_geoms);gripper_mask=mask(self.gripper_geoms)
        finally:groups[:]=saved
        with swapped_materials(self.sim.model.geom_matid,self.geom_ids):edited,edited_depth=self.render(*args,**opts)
        if not np.array_equal(image,self.render(*args,**kwargs)):raise RuntimeError('Material restoration drift')
        after=self.snapshot()
        if self.sim.data.time!=time or any(not np.array_equal(v,after[k]) for k,v in before.items()):raise RuntimeError('Rendering changed physics or camera')
        robot_support=robot_mask.copy();gripper_support=gripper_mask.copy()
        robot_mask,audit=robot_pixel_audit(rgb,edited,robot_support,edit_mask)
        gripper_mask &= ~dilate(edit_mask,2)
        if audit['robot_interior_changed']:raise RuntimeError('Material swap changed unambiguous robot-interior pixels')
        c=self.convention
        self.latest[cam]={'rgb':rgb[::c].copy(),'depth':depth[::c].copy(),'mask':edit_mask[::c].copy(),
            'robot_mask':robot_mask[::c].copy(),'gripper_mask':gripper_mask[::c].copy(),
            'robot_support_mask':robot_support[::c].copy(),'gripper_support_mask':gripper_support[::c].copy(),
            'robot_pixel_audit':audit,'segmentation':seg[::c].copy(),
            'variants':{'swapped':edited[::c].copy()},'variant_depths':{'swapped':edited_depth[::c].copy()},
            'sensor_time':time,'sensor_state':before,'sham_exact':True}
        self.calls+=1
        return image
