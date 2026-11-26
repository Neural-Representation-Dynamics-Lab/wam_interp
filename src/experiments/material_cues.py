"""Fixed-motion material-cue diagnostics; no simulator or policy imports."""
from contextlib import contextmanager
import numpy as np
from .simulator_video import dilate


def robot_pixel_audit(original,edited,robot_mask,object_mask,*,boundary_pixels=2):
    """Separate mixed object-boundary pixels from unambiguous robot pixels."""
    a,b=np.asarray(original),np.asarray(edited);robot=np.asarray(robot_mask,bool);objects=np.asarray(object_mask,bool)
    if a.shape!=b.shape or a.shape[:2]!=robot.shape or robot.shape!=objects.shape:raise ValueError('Matching RGB and masks required')
    boundary=robot&dilate(objects,boundary_pixels);interior=robot&~boundary
    changed=np.any(a!=b,axis=-1)
    return interior,{'robot_interior_pixels':int(interior.sum()),'robot_interior_changed':int((changed&interior).sum()),
                     'robot_object_boundary_pixels':int(boundary.sum()),'robot_object_boundary_changed':int((changed&boundary).sum())}


@contextmanager
def swapped_materials(array, indices):
    ids=np.asarray(indices,dtype=int)
    if ids.shape!=(2,) or len(set(ids.tolist()))!=2 or (ids<0).any() or (ids>=len(array)).any():
        raise ValueError('Two distinct valid geometry indices required')
    original=array[ids].copy()
    try:
        array[ids]=original[::-1]
        yield
    finally:array[ids]=original


def masked_registration(a,b,mask,*,radius=3):
    """Small translation diagnostic on a fixed visible gripper mask, in RGB units.

    This is an image-space feasibility check, not proof of equal internal robot
    representations. Missing/low-texture masks are explicitly unmeasurable.
    """
    a,b=np.asarray(a,float),np.asarray(b,float);mask=np.asarray(mask,bool)
    if a.shape!=b.shape or a.ndim!=3 or a.shape[-1]!=3 or mask.shape!=a.shape[:2]:
        raise ValueError('Matching finite RGB images and mask required')
    if not np.isfinite(a).all() or not np.isfinite(b).all():raise ValueError('Nonfinite image')
    mask=mask.copy();mask[:radius]=False;mask[-radius:]=False;mask[:,:radius]=False;mask[:,-radius:]=False
    y,x=np.where(mask)
    if len(y)<25:return {'measurable':False,'reason':'fewer_than_25_interior_pixels','pixels':len(y)}
    ref=a[y,x];mae=float(np.abs(ref-b[y,x]).mean())
    if ref.std()<3:return {'measurable':False,'reason':'low_texture','pixels':len(y),'mae_255':mae}
    costs=[(float(np.mean((ref-b[y+dy,x+dx])**2)),dx*dx+dy*dy,dy,dx)
           for dy in range(-radius,radius+1) for dx in range(-radius,radius+1)]
    mse,_,dy,dx=min(costs)
    return {'measurable':True,'pixels':len(y),'mae_255':mae,'best_shift_xy':[dx,dy],
            'shift_pixels':float(np.hypot(dx,dy)),'best_mse':mse}


def cue_interaction(endpoints,axis):
    """Opposite noun-conditioned responses to the identical visual cue swap.

    Keys are (instruction, future appearance). Positive interaction means soup
    moves toward original tomato position while tomato moves the opposite way.
    """
    axis=np.asarray(axis,float)
    if axis.shape!=(3,) or not np.isclose(np.linalg.norm(axis),1):raise ValueError('Unit 3D axis required')
    expected={(q,v) for q in ('soup','tomato','cream') for v in ('original','swapped')}
    if set(endpoints)!=expected:raise ValueError('Complete balanced instruction by cue factorial required')
    out={}
    for q in ('soup','tomato','cream'):
        x,y=[np.asarray(endpoints[q,v],float) for v in ('original','swapped')]
        if x.shape!=(3,) or y.shape!=(3,) or not np.isfinite([x,y]).all():raise ValueError('Finite endpoints required')
        d=y-x;out[q]={'axis_delta_m':float(d@axis),'endpoint_norm_m':float(np.linalg.norm(d)),'delta_xyz_m':d.tolist()}
    out['instruction_interaction_m']=.5*(out['soup']['axis_delta_m']-out['tomato']['axis_delta_m'])
    out['directional_pair']=bool(out['soup']['axis_delta_m']>0 and out['tomato']['axis_delta_m']<0)
    return out
