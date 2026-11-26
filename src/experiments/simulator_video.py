"""Validation and assembly for simulator-derived visual interventions.

No policy, torch, or simulator is imported here. Geometry labels describe the
rendered simulator reference, never inferred coordinates of a generated video.
"""
import numpy as np


def approach_command(start, position, target, step, *, ramp_steps=24, max_command=.4):
    """Smooth task-space approach with fixed orientation and open gripper."""
    values=[np.asarray(v,dtype=float) for v in (start,position,target)]
    if any(v.shape!=(3,) or not np.isfinite(v).all() for v in values):
        raise ValueError('Finite 3D points required')
    if not 1<=step<=32 or ramp_steps<1 or not 0<max_command<=1:
        raise ValueError('Invalid controller schedule')
    start,position,target=values
    t=min(step/ramp_steps,1.0);weight=t*t*(3-2*t)
    waypoint=start+weight*(target-start)
    return np.r_[np.clip((waypoint-position)/.05,-max_command,max_command),0.,0.,0.,-1.]


def dilate(mask,radius=2):
    mask=np.asarray(mask,dtype=bool)
    if mask.ndim!=2 or type(radius) is not int or radius<0:
        raise ValueError('2D mask and nonnegative integer radius required')
    padded=np.pad(mask,radius);out=np.zeros_like(mask)
    for y in range(2*radius+1):
        for x in range(2*radius+1):out|=padded[y:y+mask.shape[0],x:x+mask.shape[1]]
    return out


def appearance_metrics(original,edited,mask,depth,edited_depth,*,boundary_pixels=2):
    a,b=np.asarray(original),np.asarray(edited);mask=np.asarray(mask,dtype=bool)
    if a.shape!=b.shape or a.ndim!=3 or a.shape[2]!=3 or mask.shape!=a.shape[:2]:
        raise ValueError('Matching RGB images and support mask required')
    if a.dtype!=np.uint8 or b.dtype!=np.uint8:raise ValueError('Native uint8 renders required')
    if np.shape(depth)!=mask.shape or np.shape(edited_depth)!=mask.shape:
        raise ValueError('Depth must match image support')
    if not np.isfinite(depth).all() or not np.isfinite(edited_depth).all():raise ValueError('Nonfinite depth')
    change=np.abs(a.astype(float)-b.astype(float));changed=np.any(change>0,axis=-1)
    support=dilate(mask,boundary_pixels)
    return {'depth_exact':bool(np.array_equal(depth,edited_depth)),
        'changed_pixels':int(changed.sum()),'mask_pixels':int(mask.sum()),
        'changed_outside_mask':int((changed&~mask).sum()),
        'changed_outside_boundary_support':int((changed&~support).sum()),
        'rgb_mae_255':float(change.mean()),
        'support_rgb_mae_255':float(change[mask].mean()) if mask.any() else 0.0,
        'boundary_pixels':boundary_pixels}


def paired_camera_clip(agentview,wrist):
    """Use the existing FastWAM LIBERO rotate/crop/resize before concatenation."""
    from wam_interp.models.fastwam.libero_adapter import _rotate_180,_center_crop_resize
    a,b=np.asarray(agentview),np.asarray(wrist)
    if a.shape!=b.shape or a.ndim!=4 or a.shape[0]!=9 or a.shape[-1]!=3:
        raise ValueError('Nine matching paired-camera frames required')
    if a.dtype!=np.uint8 or b.dtype!=np.uint8:raise ValueError('uint8 camera frames required')
    return np.stack([np.concatenate([_center_crop_resize(_rotate_180(v[i]),width=224,height=224)
        for v in (a,b)],axis=1) for i in range(9)])


def approach_metrics(eef,objects,target_index,other_index,goal,robot_contacts):
    eef,objects=np.asarray(eef),np.asarray(objects)
    if eef.shape!=(33,3) or objects.ndim!=3 or objects.shape[0]!=33 or objects.shape[2]!=3:
        raise ValueError('Complete 32-control trajectories required')
    if not np.isfinite(eef).all() or not np.isfinite(objects).all():raise ValueError('Nonfinite trajectory')
    target=objects[0,target_index];other=objects[0,other_index]
    return {'endpoint_goal_error_m':float(np.linalg.norm(eef[-1]-goal)),
        'endpoint_target_xy_distance_m':float(np.linalg.norm(eef[-1,:2]-target[:2])),
        'endpoint_other_xy_distance_m':float(np.linalg.norm(eef[-1,:2]-other[:2])),
        'maximum_object_translation_m':float(np.linalg.norm(objects-objects[:1],axis=-1).max()),
        'robot_scene_contact_substeps':int(robot_contacts),
        'target_xy_separation_m':float(np.linalg.norm(target[:2]-other[:2]))}


def valid_approach(metrics,limits):
    if not all(np.isfinite(v) for v in metrics.values()):return False
    return (metrics['endpoint_goal_error_m']<=limits['endpoint_goal_error_m']
        and metrics['maximum_object_translation_m']<=limits['maximum_object_translation_m']
        and metrics['robot_scene_contact_substeps']==0
        and metrics['endpoint_other_xy_distance_m']-metrics['endpoint_target_xy_distance_m']>=limits['target_margin_m'])


def encode_reference_video(model,clip,observed_rgb,anchor):
    """Encode actual simulator RGB, replacing frame zero for appearance factorials.

    RGB originates as native uint8 simulator output; subsequent model arithmetic
    matches FastWAM's bf16 observation transform. No donor text is supplied.
    """
    import torch
    from .video_bottleneck import anchor_future
    clip=np.asarray(clip);observed_rgb=np.asarray(observed_rgb)
    if clip.shape!=(9,224,448,3) or observed_rgb.shape!=clip.shape[1:] or clip.dtype!=np.uint8 or observed_rgb.dtype!=np.uint8:
        raise ValueError('Native nine-frame uint8 montage and current image required')
    rgb=clip.copy();rgb[0]=observed_rgb
    tensor=torch.as_tensor(rgb.copy(),device=model.device).permute(3,0,1,2).unsqueeze(0).to(model.torch_dtype)
    tensor=tensor*(2.0/255.0)-1.0
    with torch.inference_mode():
        encoded=model.vae.model.encode(tensor,model.vae.scale)
        repeated=model.vae.model.encode(tensor,model.vae.scale)
        if not torch.equal(encoded,repeated):raise ValueError('Repeated native encoding differs')
        before=encoded.float().cpu().numpy().copy()
        future=anchor_future(before,anchor)
        decoded=model.vae.model.decode(torch.as_tensor(future,device=model.device,dtype=model.torch_dtype),model.vae.scale).clamp(-1,1)
        if decoded.shape!=tensor.shape or not torch.isfinite(decoded).all():raise ValueError('Invalid reference reconstruction')
        reconstructed=decoded.float().cpu().numpy().copy();normalized=tensor.float().cpu().numpy().copy()
    return {'future':future,'source_rgb':rgb,'decoded_rgb':reconstructed}, {
        'input_kind':'simulator_rgb','input_text_context':None,'compute_dtype':str(model.torch_dtype),
        'whole_clip_temporal_encode':True,'repeated_encoding_exact':True,
        'anchor_error_before_restore':float(np.max(np.abs(before[:,:,:1]-np.asarray(anchor)[:,:,:1]))),
        'anchor_exact_after_restore':True,'rgb_mae_255':float(np.abs(reconstructed-normalized).mean()*127.5),
        'first_frame_rgb_mae_255':float(np.abs(reconstructed[:,:,:1]-normalized[:,:,:1]).mean()*127.5),
        'future_rgb_mae_255':float(np.abs(reconstructed[:,:,1:]-normalized[:,:,1:]).mean()*127.5)}
