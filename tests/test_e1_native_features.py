import pytest
import torch
from types import SimpleNamespace

from wam_interp.experiments.native_features import native_features, edit_future_kv, lingbot_predicted_kv


def test_selected_stage_preserves_observed_tokens_and_cfg_rows():
    owner = SimpleNamespace()
    keys = torch.arange(24, dtype=torch.float32).reshape(2, 4, 3)
    vals = -keys
    replacement = {1: (torch.ones_like(keys[:,2:]), torch.zeros_like(vals[:,2:]))}
    with native_features(owner, stage=1, donors=replacement) as cap:
        assert edit_future_kv(cap, keys, vals, 2) is None
        k, v = edit_future_kv(cap, keys, vals, 2)
        assert torch.equal(k[:,:2], keys[:,:2]) and torch.equal(v[:,:2], vals[:,:2])
        assert torch.equal(k[:,2:], replacement[1][0]) and torch.equal(v[:,2:], replacement[1][1])
    assert not hasattr(owner, "_e1_features")


def test_failure_cleans_up_and_cannot_patch_incompatible_donor():
    owner = SimpleNamespace()
    with pytest.raises(ValueError,match="Invalid donor"):
        with native_features(owner, donors={0:(torch.zeros(2),)}) as cap:
            cap.visit((torch.zeros(3),))
    assert not hasattr(owner, "_e1_features")
    with pytest.raises(RuntimeError,match="never reached"):
        with native_features(owner, stage=1) as cap:
            cap.visit((torch.zeros(3),))
    assert not hasattr(owner, "_e1_features")


def test_lingbot_protects_observed_history_and_first_chunk_anchor():
    keys=torch.arange(16).reshape(2,8,1).float()
    cache={"k":keys.clone(),"v":-keys.clone(),"mask":torch.tensor([1,1,1,1,1,1,0,0],dtype=torch.bool),
           "is_pred":torch.tensor([0,0,1,1,1,1,1,0],dtype=torch.bool),"id":torch.tensor([0,0,1,1,1,1,0,-1])}
    layer=SimpleNamespace(attn_caches={"pos":cache})
    runtime=SimpleNamespace(config=SimpleNamespace(patch_size=(1,1,1),frame_chunk_size=2),
              latent_height=1,latent_width=2,frame_st_id=0,cache_name="pos",_cache_attentions=lambda:[layer])
    zeros=torch.zeros_like(keys[:,4:6])
    with native_features(runtime,layer=0,donors={0:(zeros,zeros)}) as cap:
        lingbot_predicted_kv(runtime,cap)
    assert torch.equal(cache["k"][:,:4],keys[:,:4])
    assert torch.equal(cache["k"][:,4:6],zeros)
    assert torch.equal(cache["k"][:,6:],keys[:,6:])
    assert cache["mask"].tolist()==[1,1,1,1,1,1,0,0]

    assert cap.frame_start == 0
    with pytest.raises(ValueError, match="different video frames"):
        with native_features(runtime, layer=0, expected_frame_start=4) as wrong:
            lingbot_predicted_kv(runtime, wrong)
    assert not hasattr(runtime, "_e1_features")
