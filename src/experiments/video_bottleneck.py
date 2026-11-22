"""Text-free native VAE round trips, with explicit codec and anchor provenance.

This is a lossy visual bottleneck, not a language-information eraser.
Importing this module does not load torch, model weights or a simulator.
"""
import numpy as np


def anchor_future(reconstructed, reference):
    a,b=np.asarray(reconstructed),np.asarray(reference)
    if a.ndim!=5 or a.shape!=b.shape or a.shape[2]<2:
        raise ValueError('Matching B,C,T,H,W latents with future slots required')
    if not np.isfinite(a).all() or not np.isfinite(b).all():raise ValueError('Nonfinite latent')
    out=a.copy();out[:,:,:1]=b[:,:,:1]
    return out


def trajectory_difference(a,b):
    a,b=np.asarray(a,dtype=float),np.asarray(b,dtype=float)
    if a.ndim!=2 or a.shape!=b.shape or a.shape[1]!=3 or len(a)<2:
        raise ValueError('Equal nonempty Nx3 trajectories required')
    if not np.isfinite(a).all() or not np.isfinite(b).all():raise ValueError('Nonfinite trajectory')
    if not np.array_equal(a[0],b[0]):raise ValueError('Different initial EEF positions')
    d=b-a
    return {'endpoint_separation_m':float(np.linalg.norm(d[-1])),
            'mean_trajectory_separation_m':float(np.linalg.norm(d,axis=1).mean()),
            'signed_world_y_m':float(d[-1,1])}


def baseline_gate(comparison,limits):
    for key in ('endpoint_separation_m','mean_trajectory_separation_m'):
        if not np.isfinite(comparison[key]) or not np.isfinite(limits[key]) or limits[key]<=0:
            raise ValueError('Finite positive gate limits and finite measurements required')
    return all(comparison[k]<=limits[k] for k in ('endpoint_separation_m','mean_trajectory_separation_m'))


def roundtrip_future(model,latent):
    """Decode and jointly encode all frames using the native eager temporal VAE.

    Compute stays at the model dtype; RGB remains a floating tensor. The decoder's
    usual [-1,1] clamp is retained and quantified. No uint8/JPEG, per-frame encode,
    text context, compiled encoder, interpolation or rescaling is introduced.
    """
    import torch
    a=np.asarray(latent)
    if a.ndim!=5 or a.shape[0]!=1 or a.shape[2]<2 or not np.isfinite(a).all():
        raise ValueError('Finite batch-one video latent required')
    z=torch.as_tensor(a,device=model.device,dtype=model.torch_dtype)
    vae=model.vae
    def cpu(t):return t.detach().float().cpu().numpy().copy()
    with torch.inference_mode():
        decoded_raw=vae.model.decode(z,vae.scale)
        if not torch.isfinite(decoded_raw).all():raise ValueError('Nonfinite decoded RGB')
        expected_t=1+(z.shape[2]-1)*vae.temporal_downsample_factor
        if decoded_raw.ndim!=5 or tuple(decoded_raw.shape[:3])!=(1,3,expected_t):
            raise ValueError('Incorrect temporal decoding layout')
        clipped=float(((decoded_raw<-1)|(decoded_raw>1)).float().mean().item())
        rgb=decoded_raw.clamp(-1,1)
        reconstructed=vae.model.encode(rgb,vae.scale)
        before=cpu(reconstructed);original=cpu(z)
        after=anchor_future(before,original)
        used=torch.as_tensor(after,device=model.device,dtype=model.torch_dtype)
        decoded_after=vae.model.decode(used,vae.scale).clamp(-1,1)
        if decoded_after.shape!=rgb.shape or not torch.isfinite(decoded_after).all():
            raise ValueError('Invalid round-trip reconstruction')
        raw_rgb,out_rgb=cpu(rgb),cpu(decoded_after)
    diff=np.abs(raw_rgb-out_rgb)*127.5
    mid=diff.shape[-1]//2
    if diff.shape[-1]%2:raise ValueError('Two equal horizontal camera views required')
    metrics={'compute_dtype':str(model.torch_dtype),'storage_dtype':'float32',
        'rgb_quantized_to_uint8':False,'rgb_clip_fraction':clipped,
        'latent_shape':list(original.shape),'rgb_shape':list(raw_rgb.shape),
        'anchor_max_abs_error_before_restore':float(np.max(np.abs(before[:,:,:1]-original[:,:,:1]))),
        'anchor_exact_after_restore':bool(np.array_equal(after[:,:,:1],original[:,:,:1])),
        'future_latent_rmse':float(np.sqrt(np.mean((before[:,:,1:].astype(float)-original[:,:,1:])**2))),
        'future_external_rgb_mae_255':float(diff[:,:,1:,:,:mid].mean()),
        'future_wrist_rgb_mae_255':float(diff[:,:,1:,:,mid:].mean()),
        'first_frame_rgb_mae_255':float(diff[:,:,:1].mean()),
        'input_text_context':None,'temporal_encoding':'whole_clip_eager_native_VAE'}
    return {'future':after,'before_anchor_restore':before,'decoded_rgb':raw_rgb,'roundtrip_rgb':out_rgb},metrics
