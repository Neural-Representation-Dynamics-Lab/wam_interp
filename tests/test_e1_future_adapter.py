from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from wam_interp.experiments.future_adapter import FutureAdapter
from wam_interp.experiments.futures import BOUNDARY, apply_future_hook
from wam_interp.types import InferenceRequest, InferenceResult, PreparedObservation


class Stub:
    def __init__(self, backend):
        self.name = backend
        self.native_condition = "idm"
        self.model = SimpleNamespace(
            research_future_boundary=BOUNDARY,
            mot=SimpleNamespace(condition_layer_set={0, 4, 8}),
            transformer=SimpleNamespace(blocks=[1, 2, 3]),
            device="cpu",
            torch_dtype=torch.float32,
        )
        self.architecture = SimpleNamespace(mot_driver=SimpleNamespace(num_layers=3))
        self.x = torch.arange(12, dtype=torch.float32).reshape(1, 2, 3, 1, 2)

    def predict_action(self, request):
        if self.name == "fastwam":
            used = apply_future_hook(self.model, self.x.clone())
            assert torch.equal(used[:, :, :1], self.x[:, :, :1])
            value = used[:, :, 1:]
        else:
            owner = self.architecture.mot_driver if self.name == "openwam" else self.model
            value = self.x[:, :, 1:].clone()
            hook = getattr(owner, "_e1_features", None)
            if hook:
                replacement = hook.visit((value,))
                if replacement is not None:
                    (value,) = replacement
        return InferenceResult(value.sum()[None])


def request(mode, image=None):
    obs = PreparedObservation(torch.zeros(1, 3, 2, 2) if image is None else image, None, "move")
    return InferenceRequest(obs, condition=mode, num_video_frames=3)


@pytest.mark.parametrize(
    "backend,condition",
    [
        ("fastwam", "idm"),
        ("fasterwam", "one_pass_future_cache"),
        ("lingbot_va", "video"),
        ("longwam", "idm"),
        ("openwam", "joint"),
    ],
)
def test_common_donor_api(backend, condition):
    adapter = Stub(backend)
    api = FutureAdapter(adapter)
    req = request(condition)
    baseline = adapter.predict_action(req).actions
    captured, donor = api.capture(req)
    assert torch.equal(captured.actions, baseline)
    assert torch.equal(api.predict_with_future(req, donor).actions, baseline)
    zeros = replace(donor, parts=tuple(torch.zeros_like(v) for v in donor.parts))
    assert api.predict_with_future(req, zeros).actions.item() == 0
    assert torch.equal(adapter.predict_action(req).actions, baseline)
    with pytest.raises(ValueError, match="loaded adapter"):
        FutureAdapter(adapter).predict_with_future(req, donor)
    with pytest.raises(ValueError, match="schedule"):
        api.predict_with_future(replace(req, num_inference_steps=5), donor)
    with pytest.raises(ValueError):
        api.predict_with_future(req, replace(donor, parts=(donor.parts[0].double(),)))
    if not api.supports_clip:
        with pytest.raises(NotImplementedError, match="clip"):
            api.from_clip(req, np.zeros((3, 2, 2, 3), np.uint8))


def test_clean_video_uses_only_future_frames_and_checks_anchor():
    adapter = Stub("fastwam")
    adapter.model.vae = SimpleNamespace(model=SimpleNamespace(encode=lambda video, scale: video), scale=1)
    api = FutureAdapter(adapter)
    clip = np.zeros((3, 2, 2, 3), np.uint8)
    clip[-1] = 255
    req = request("idm", -torch.ones(1, 3, 2, 2))
    donor = api.from_clip(req, clip)
    assert donor.parts[0].shape == (1, 3, 2, 2, 2)
    assert torch.all(donor.parts[0][:, :, -1] == 1)
    with pytest.raises(ValueError, match="frame zero"):
        api.from_clip(request("idm"), clip)


def test_longwam_encodes_recipient_history_and_uses_paired_noise():
    adapter = Stub("longwam")
    adapter.model.num_imagine_frames = 2
    adapter.model.num_clean_frames = 2
    adapter.model.imagine_timestep = 90
    adapter.model.vae = SimpleNamespace(temporal_downsample_factor=1)
    adapter.model.train_video_scheduler = SimpleNamespace(
        add_noise=lambda clean, noise, t: 0.1 * clean + 0.9 * noise
    )
    seen = []

    def encode(video, tiled):
        seen.append(video.clone())
        return video

    adapter.model._encode_video_latents = encode
    adapter._in_episode = True
    adapter.past_obs_size = 1
    adapter.sample_stride = 1
    adapter._frames = [torch.zeros(3, 2, 2), -torch.ones(3, 2, 2)]
    req = request("idm", adapter._frames[-1][None])
    api = FutureAdapter(adapter)
    clip = np.zeros((3, 2, 2, 3), np.uint8)
    a = api.from_clip(req, clip, noise_seed=0)
    b = api.from_clip(req, clip, noise_seed=0)
    assert torch.equal(a.parts[0], b.parts[0])
    clip[1:] = 255
    c = api.from_clip(req, clip, noise_seed=0)
    torch.testing.assert_close(c.parts[0] - b.parts[0], torch.full_like(c.parts[0], 0.2))
    assert torch.equal(seen[0][0, :, 0], adapter._frames[0])
    assert torch.equal(seen[0][0, :, 1], adapter._frames[1])
    assert a.parts[0].shape[2] == 2


def test_held_out_direction_gate_uses_only_development_templates():
    from wam_interp.experiments.clip_study import score_directions

    def case(split, index, sign):
        direction = np.array([sign, 0.0])
        return {"split": split, "state": index, "directions": {"arm": direction, "book": direction}}

    records = [case("development", 0, 1.0)]
    records += [case("held_out", i, sign) for i, sign in enumerate((1.0, -1.0, 1.0, 1.0), start=24)]
    scores = score_directions(records)
    assert scores["book"]["correct"] == 3 and scores["book"]["passed"]
    assert not scores["arm"]["passed"]
    assert scores["mug"]["correct"] == 0 and not scores["mug"]["passed"]
    assert not score_directions(records[:2])["book"]["passed"]


def test_prepared_frames_match_rgb_donor_and_reject_one_pass():
    model = Stub("fastwam")
    model.model.vae = SimpleNamespace(model=SimpleNamespace(encode=lambda video, scale: video), scale=1)
    api = FutureAdapter(model)
    rgb = np.zeros((3,2,2,3), dtype=np.uint8)
    rgb[1:] = 255
    frames = [torch.from_numpy(x).permute(2,0,1)[None].float().mul(2/255).sub(1) for x in rgb]
    req = request("idm", frames[0])
    assert torch.equal(api.from_clip(req,rgb).parts[0], api.from_prepared_frames(req,frames).parts[0])
    with pytest.raises(ValueError,match="recipient observation"):
        api.from_prepared_frames(req,frames[::-1])
    with pytest.raises(NotImplementedError,match="sigma=1"):
        FutureAdapter(Stub("fasterwam")).from_prepared_frames(request("one_pass_future_cache"),frames)
