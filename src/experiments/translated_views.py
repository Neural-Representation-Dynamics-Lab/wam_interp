"""Render a prescribed, smooth object translation on a common simulator trajectory."""
import numpy as np

from .physical_relocation import object_geometry_ids


class TranslatedViews:
    def __init__(self,env,objects):
        import mujoco
        from robosuite import macros
        from robosuite.utils.mjcf_utils import IMAGE_CONVENTION_MAPPING
        self.sim=env.sim
        self.api=mujoco
        self.order=IMAGE_CONVENTION_MAPPING[macros.IMAGE_CONVENTION]
        self.ids={name:object_geometry_ids(self.sim.model,env.env.obj_body_id[name]) for name in objects}
        model=self.sim.model
        self.solids=set(np.flatnonzero((model.geom_contype!=0)|(model.geom_conaffinity!=0)).tolist())
        self.obstacles={}
        for name,ids in self.ids.items():
            # Table support is allowed. Compare with every other movable object, fixture and robot.
            others=set()
            for idx in self.solids.difference(ids):
                b=model.body_id2name(int(model.geom_bodyid[idx])) or ""
                if b in ("world","floor") or "table" in b:
                    continue
                others.add(idx)
            self.obstacles[name]=others
        self.step=0
        self.capture=False
        self.views={}
        self.min_clearance={}
        self.visibility={}

    def __enter__(self):
        self.native=self.sim.render
        self.sim.render=self.render
        return self

    def __exit__(self,*args):
        self.sim.render=self.native

    def render(self,*args,**kwargs):
        original=self.native(*args,**kwargs)
        cam=kwargs.get("camera_name")
        if not self.capture or cam not in ("agentview","robot0_eye_in_hand") or kwargs.get("segmentation") or kwargs.get("depth"):
            return original
        self.views[("sham",cam)]=original[::self.order].copy()
        frac=min(self.step/24,1.)
        advance=.03*(10*frac**3-15*frac**4+6*frac**5)
        d=self.sim.data
        m=self.sim.model
        for obj,ids in self.ids.items():
            start=d.geom_xpos[ids].copy()
            for sign in (-1,1):
                key=f"{obj}_{sign}"
                try:
                    d.geom_xpos[ids,0]=start[:,0]+sign*advance
                    edited=self.native(*args,**kwargs)
                    self.views[key,cam]=edited[::self.order].copy()
                    if cam=="agentview":
                        groups=self.sim._render_context_offscreen.vopt.sitegroup
                        keep=groups.copy()
                        try:
                            groups[:]=0
                            seg=self.native(*args,**{**kwargs,"segmentation":True})
                        finally:
                            groups[:]=keep
                        mask=(seg[...,0]==int(self.api.mjtObj.mjOBJ_GEOM))&np.isin(seg[...,1],ids)
                        self.visibility[key]=min(self.visibility.get(key,mask.size),int(mask.sum()))
                        dist=.02
                        for a in self.solids.intersection(ids):
                            for b in self.obstacles[obj]:
                                gap=self.api.mj_geomDistance(m._model,d._data,int(a),int(b),.02,None)
                                dist=min(dist,float(gap))
                        self.min_clearance[key]=min(dist,self.min_clearance.get(key,.02))
                finally:
                    d.geom_xpos[ids]=start
        if not np.array_equal(original,self.native(*args,**kwargs)):
            raise RuntimeError("Restoring rendered geometry did not reproduce the original image")
        return original
