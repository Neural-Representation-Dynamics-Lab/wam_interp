import numpy as np
import pytest
from wam_interp.experiments.video_bottleneck import anchor_future,trajectory_difference,baseline_gate,roundtrip_future


def test_anchor_restoration_preserves_all_future_slots_and_inputs():
    a=np.ones((1,2,3,2,2));b=np.zeros_like(a);out=anchor_future(a,b)
    assert np.array_equal(out[:,:,:1],b[:,:,:1])
    assert np.array_equal(out[:,:,1:],a[:,:,1:]) and np.all(a==1)


def test_invalid_latents_fail():
    a=np.ones((1,2,3,2,2));b=a.copy();b[0,0,1,0,0]=np.nan
    with pytest.raises(ValueError,match='Nonfinite'):anchor_future(a,b)
    with pytest.raises(ValueError,match='Matching'):anchor_future(a,a[0])


def test_gate_is_behavioral_and_rejects_invalid_limits():
    a=np.zeros((33,3));b=a.copy();b[1:,1]=.002
    c=trajectory_difference(a,b);limits={'endpoint_separation_m':.01,'mean_trajectory_separation_m':.005}
    assert baseline_gate(c,limits)
    b[-1,1]=.02;assert not baseline_gate(trajectory_difference(a,b),limits)
    b[0,1]=.01
    with pytest.raises(ValueError,match='initial'):trajectory_difference(a,b)
    with pytest.raises(ValueError):baseline_gate(c,{**limits,'endpoint_separation_m':np.nan})


def test_native_codec_uses_whole_clip_without_language_or_uint8():
    import torch
    class Codec:
        def decode(self,z,scale):return z.repeat_interleave(4,dim=2)[:,:,:9]
        def encode(self,rgb,scale):
            assert rgb.dtype==torch.float32 and rgb.shape==(1,3,9,2,4)
            return rgb[:,:,::4]+.01
    class VAE: model=Codec();scale=[0,1];temporal_downsample_factor=4
    class Model:vae=VAE();device='cpu';torch_dtype=torch.float32
    z=np.linspace(-.8,.8,72,dtype=np.float32).reshape(1,3,3,2,4)
    arrays,meta=roundtrip_future(Model(),z)
    assert np.array_equal(arrays['future'][:,:,:1],z[:,:,:1])
    assert np.allclose(arrays['future'][:,:,1:],z[:,:,1:]+.01)
    assert meta['input_text_context'] is None and not meta['rgb_quantized_to_uint8']
    assert meta['anchor_max_abs_error_before_restore']>0
