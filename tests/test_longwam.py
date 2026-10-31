"""CPU checks for the Long-WAM backend with tiny random-weight models built from the vendored classes."""

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
import math
import subprocess
import sys

import numpy as np
import pytest
import torch

from wam_interp.models.longwam.adapter import LongWAMAdapter
from wam_interp.models.longwam.libero_adapter import LongWAMLiberoAdapter
from wam_interp.models.longwam.libero_plus_adapter import LongWAMLiberoPlusAdapter
from wam_interp.models.longwam.robotwin_adapter import LongWAMRobotWinAdapter
from wam_interp.models.longwam.transforms import build_observation_window
from wam_interp.types import EnvironmentObservation, InferenceRequest, PreparedObservation

Z = 4  # tiny VAE latent channels
TEXT_DIM = 16


def _tiny_vae():
    from wam_interp.models.longwam.model_code.models.wan22.wan_video_vae import (
        VideoVAE38_, WanVideoVAE, WanVideoVAE38)

    class TinyVAE38(WanVideoVAE38):
        """WanVideoVAE38.__init__ with a narrow encoder/decoder; same encode/temporal causality."""

        def __init__(self):
            super(WanVideoVAE, self).__init__()
            self.mean, self.std = torch.zeros(Z), torch.ones(Z)
            self.scale = [self.mean, 1.0 / self.std]
            self.model = VideoVAE38_(z_dim=Z, dim=4, dec_dim=4).eval().requires_grad_(False)
            self.upsampling_factor, self.temporal_downsample_factor, self.z_dim = 16, 4, Z

    return TinyVAE38()


class StubTokenizer:
    """Same call contract as HuggingfaceTokenizer(seq_len=8)."""

    def __call__(self, prompt, return_mask=False, add_special_tokens=True):
        ids = torch.zeros(1, 8, dtype=torch.long)
        mask = torch.zeros(1, 8, dtype=torch.long)
        for index, word in enumerate(prompt.split()[-8:]):
            ids[0, index] = sum(map(ord, word)) % 63 + 1
            mask[0, index] = 1
        return (ids, mask) if return_mask else ids


def build_tiny_longwam(*, joint=False, action_dim=7, proprio_dim=8, past=48, k=2, seed=0):
    from wam_interp.models.longwam.model_code.models.wan22.action_dit import ActionDiT
    from wam_interp.models.longwam.model_code.models.wan22.long_wam import LongWAM
    from wam_interp.models.longwam.model_code.models.wan22.mot import MoT
    from wam_interp.models.longwam.model_code.models.wan22.wan_video_dit import WanVideoDiT
    from wam_interp.models.longwam.model_code.models.wan22.wan_video_text_encoder import WanTextEncoder

    torch.manual_seed(seed)
    common = dict(text_dim=TEXT_DIM, freq_dim=32, eps=1e-6, num_heads=2, attn_head_dim=12, num_layers=2)
    video = WanVideoDiT(hidden_dim=32, in_dim=Z, ffn_dim=64, out_dim=Z, patch_size=(1, 2, 2), has_image_input=False,
                        seperated_timestep=True, video_attention_mask_mode="per_frame_causal", **common)
    action = ActionDiT(hidden_dim=16, action_dim=action_dim, ffn_dim=32, **common)
    mot = MoT({"video": video, "action": action}, mot_checkpoint_mixed_attn=False)
    text = WanTextEncoder(vocab=64, dim=TEXT_DIM, dim_attn=TEXT_DIM, dim_ffn=32, num_heads=2, num_layers=1, num_buckets=8)
    return LongWAM(video_expert=video, action_expert=action, mot=mot, vae=_tiny_vae(), text_encoder=text,
                   tokenizer=StubTokenizer(), text_dim=TEXT_DIM, proprio_dim=proprio_dim, device="cpu",
                   torch_dtype=torch.float32, longwam_past_obs_size=past, longwam_num_imagine_frames=k,
                   longwam_imagine_sigma=0.0 if joint else 0.9, longwam_imagine_infer_steps=4,
                   longwam_joint_denoise=joint).eval()


def libero_data_config(camera_hw=(224, 224)):
    h, w = camera_hw
    images = [{"key": key, "raw_shape": [3, 512, 512], "shape": [3, h, w]} for key in ("image", "wrist_image")]
    shape_meta = {"images": images, "action": [{"key": "default", "raw_shape": 7, "shape": 7}],
                  "state": [{"key": "default", "raw_shape": 8, "shape": 8}]}
    return {"shape_meta": shape_meta, "num_frames": 81, "past_obs_size": 48, "action_chunk": 32,
            "action_video_freq_ratio": 4, "video_size": [h, 2 * w], "concat_multi_camera": "horizontal",
            "processor": {"shape_meta": shape_meta, "num_output_cameras": 2, "action_output_dim": 7,
                          "proprio_output_dim": 8, "action_state_transforms": None,
                          "use_stepwise_action_norm": False, "norm_default_mode": "min/max",
                          "norm_exception_mode": None}}


def libero_stats():
    def field(n):
        low, high = [-2.0] * n, [4.0] * n
        low[0] = high[0] = 0.5  # a constant dimension exercises the range_tol branch
        return {"global_min": low, "global_max": high, "stepwise_min": [[9.0] * n], "stepwise_max": [[9.0] * n]}
    return {"state": {"default": field(8)}, "action": {"default": field(7)}, "num_episodes": 1}


def robotwin_data_config():
    images = [{"key": key, "raw_shape": [3, 480, 640], "shape": [3, 240, 320]}
              for key in ("cam_high", "cam_left_wrist", "cam_right_wrist")]
    shape_meta = {"images": images, "action": [{"key": "default", "raw_shape": 14, "shape": 14}],
                  "state": [{"key": "default", "raw_shape": 14, "shape": 14}]}
    return {"shape_meta": shape_meta, "num_frames": 81, "past_obs_size": 48, "action_chunk": 32,
            "action_video_freq_ratio": 4, "video_size": [384, 320], "concat_multi_camera": "robotwin",
            "processor": {"shape_meta": shape_meta, "num_output_cameras": 3, "action_output_dim": 14,
                          "proprio_output_dim": 14, "action_state_transforms": None,
                          "use_stepwise_action_norm": False, "norm_default_mode": "z-score",
                          "norm_exception_mode": None}}


def robotwin_stats():
    mean = [0.1 * i for i in range(14)]
    std = [0.5 + 0.05 * i for i in range(14)]
    return {kind: {"default": {"global_mean": mean, "global_std": std}} for kind in ("state", "action")}


def libero_observation(step, size=256):
    rng = np.random.default_rng(step)
    return EnvironmentObservation(
        cameras={"agentview": rng.integers(0, 256, (size, size, 3), dtype=np.uint8),
                 "wrist": rng.integers(0, 256, (size, size, 3), dtype=np.uint8)},
        state={"eef_position": np.array([0.1, -0.2, 0.3]) + 0.01 * step,
               "eef_quaternion": np.array([0.0, 0.0, math.sin(0.25), math.cos(0.25)]),
               "gripper_position": np.array([0.02, -0.02])},
        instruction="put the bowl on the plate", metadata={"step": step})


def robotwin_observation(step):
    rng = np.random.default_rng(1000 + step)
    cameras = {name: rng.integers(0, 256, (240, 320, 3), dtype=np.uint8) for name in ("head", "left_wrist", "right_wrist")}
    return EnvironmentObservation(cameras=cameras, state={"joint_position": np.linspace(-1, 1, 14) + 0.01 * step},
                                  instruction="lift the pot", metadata={"step": step})


def window_request(frames=13, hw=(32, 64), **changes):
    image = torch.rand(1, 3, frames, *hw, generator=torch.Generator().manual_seed(3)) * 2 - 1
    request = InferenceRequest(PreparedObservation(image, torch.zeros(1, 8), "pick cup"), condition="idm",
                               action_horizon=32, num_inference_steps=3, seed=0)
    return replace(request, **changes)


def test_package_import_is_light_and_registers():
    code = ("import sys; import wam_interp.models.longwam; from wam_interp.models import registry; "
            "assert 'longwam' in registry.registered_model_adapters() and 'longwam' in registry.registered_model_loaders(); "
            "assert not {'torch', 'hydra', 'omegaconf'} & set(sys.modules)")
    subprocess.run([sys.executable, "-c", code], check=True)


def test_vendored_code_is_self_contained():
    from wam_interp.models.longwam.model_code.models.wan22.long_wam import LongWAM
    root = Path(__file__).resolve().parents[1] / "src" / "models" / "longwam" / "model_code"
    offending = [str(path) for path in root.rglob("*.py")
                 if any(line.lstrip().startswith(("from longwam", "import longwam")) for line in path.read_text().splitlines())]
    assert not offending
    assert "longwam" not in sys.modules
    assert LongWAM.LongWAM_RAW_FRAMES_PER_CLEAN_LATENT == 16
    for name in ("LICENSE", "NOTICE", "THIRD_PARTY_NOTICES.md", "licenses/FastWAM-MIT.txt", "licenses/Apache-2.0.txt"):
        assert (root / name).is_file()


@pytest.mark.parametrize("joint", [False, True])
def test_tiny_model_has_release_history_and_imagination_contract(joint):
    model = build_tiny_longwam(joint=joint)
    assert model.num_clean_frames == 4  # P48 -> 48 // 16 + 1 clean latents
    assert model.num_imagine_frames == 2 and model.joint_denoise is joint
    assert model.imagine_timestep == (0.0 if joint else 900.0) and model.imagine_infer_steps == 4
    assert LongWAMAdapter(model).native_condition == ("codenoise" if joint else "idm")
    assert LongWAMAdapter(model).inference_defaults() == {"condition": "codenoise" if joint else "idm"}


@pytest.mark.parametrize("joint", [False, True])
def test_generic_inference_returns_finite_chunk_and_metadata(joint):
    model = build_tiny_longwam(joint=joint)
    adapter = LongWAMAdapter(model)
    request = window_request(condition="codenoise" if joint else "idm")
    result = adapter.predict_action(request)
    assert result.actions.shape == (32, 7) and result.actions.dtype == torch.float32
    assert torch.isfinite(result.actions).all()
    meta = result.metadata
    assert meta["inference_method"] == "infer_joint_ar" and meta["num_clean_frames"] == 4
    assert meta["observation_window_shape"] == [1, 3, 13, 32, 64] and meta["input_image_shape"] == [1, 3, 32, 64]
    assert meta["video_inference_steps"] == (3 if joint else 4)
    assert meta["imagine_sigma"] == pytest.approx(0.0 if joint else 0.9)
    assert meta["prompt"].endswith("instruction: pick cup")
    torch.testing.assert_close(adapter.predict_action(request).actions, result.actions, rtol=0, atol=0)
    assert not torch.equal(adapter.predict_action(replace(request, seed=1)).actions, result.actions)


@pytest.mark.parametrize("changes,error", [
    ({"condition": "codenoise"}, "trained for 'idm'"), ({"condition": "first_frame"}, "conditions are"),
    ({"compile_action_infer": True}, "compiled"), ({"text_cfg_scale": 2.0}, "text guidance"),
    ({"action_horizon": 0}, "positive"),
])
def test_generic_adapter_rejects_unsupported_requests(changes, error):
    with pytest.raises(ValueError, match=error):
        LongWAMAdapter(build_tiny_longwam()).predict_action(window_request(**changes))


def test_generic_adapter_requires_history_window():
    adapter = LongWAMAdapter(build_tiny_longwam())
    single = window_request()
    single = replace(single, observation=replace(single.observation, image=single.observation.image[:, :, -1]))
    with pytest.raises(ValueError, match="clean history latents"):
        adapter.predict_action(single)
    with pytest.raises(ValueError, match=r"\[1,3,13,H,W\]"):
        adapter.predict_action(window_request(frames=9))


def test_idm_prefills_video_kv_once_and_reuses_it_for_every_action_step(monkeypatch):
    model = build_tiny_longwam()
    calls = {"prefill": [], "action_cache": [], "video_t": []}
    prefill, forward_action = model.mot.prefill_video_cache, model.mot.forward_action_with_video_cache
    pre_dit = model.video_expert.pre_dit

    def record_prefill(**kwargs):
        cache = prefill(**kwargs)
        calls["prefill"].append(cache)
        return cache

    def record_action(**kwargs):
        calls["action_cache"].append(kwargs["video_kv_cache"])
        return forward_action(**kwargs)

    def record_pre_dit(**kwargs):
        calls["video_t"].append(float(kwargs["timestep"].reshape(-1)[0]))
        return pre_dit(**kwargs)

    monkeypatch.setattr(model.mot, "prefill_video_cache", record_prefill)
    monkeypatch.setattr(model.mot, "forward_action_with_video_cache", record_action)
    monkeypatch.setattr(model.video_expert, "pre_dit", record_pre_dit)
    LongWAMAdapter(model).predict_action(window_request(num_inference_steps=10))

    assert len(calls["prefill"]) == 1
    cache = calls["prefill"][0]
    tokens = (4 + 2) * 2  # [4 clean history | 2 imagined] latents x (1x2) tokens per latent frame
    assert len(cache) == model.mot.num_layers
    assert all(set(layer) == {"k", "v"} and layer["k"].shape == layer["v"].shape == (1, tokens, 24) for layer in cache)
    assert len(calls["action_cache"]) == 10 and all(item is cache for item in calls["action_cache"])
    # 4 partial video steps from sigma 1.0, then the prefill at the trained imagination sigma 0.9.
    assert len(calls["video_t"]) == 5 and calls["video_t"][0] == pytest.approx(1000.0)
    assert all(a > b for a, b in zip(calls["video_t"], calls["video_t"][1:]))
    assert calls["video_t"][-1] == pytest.approx(900.0)


def test_codenoise_denoises_video_and_action_jointly_without_prefill(monkeypatch):
    model = build_tiny_longwam(joint=True)
    joint_calls = []
    joint = model._predict_joint_noise

    def record_joint(**kwargs):
        joint_calls.append(kwargs["latents_video"].shape)
        return joint(**kwargs)

    monkeypatch.setattr(model, "_predict_joint_noise", record_joint)
    monkeypatch.setattr(model.mot, "prefill_video_cache", lambda **kwargs: pytest.fail("COD must not prefill"))
    LongWAMAdapter(model).predict_action(window_request(condition="codenoise", num_inference_steps=4))
    assert joint_calls == [torch.Size([1, Z, 6, 2, 4])] * 4


def test_build_observation_window_pads_with_oldest_and_samples_stride():
    frames = [torch.full((3, 2, 2), float(index)) for index in range(60)]
    first = build_observation_window(frames[:1], 48, 4)
    assert first.shape == (1, 3, 13, 2, 2) and torch.count_nonzero(first) == 0
    partial = build_observation_window(frames[:11], 48, 4)
    assert partial[0, 0, :, 0, 0].tolist() == [0.0] * 10 + [2.0, 6.0, 10.0]
    full = build_observation_window(frames, 48, 4)
    assert full[0, 0, :, 0, 0].tolist() == [float(i) for i in range(11, 60, 4)]
    assert build_observation_window(frames, 0, 4) is None


def test_libero_preprocessing_matches_upstream_formula():
    from PIL import Image

    adapter = LongWAMLiberoAdapter(build_tiny_longwam(), data_config=libero_data_config(), dataset_stats=libero_stats())
    observation = libero_observation(0)
    agent = observation.cameras["agentview"].copy()
    agent[:16, :16] = (255, 0, 0)  # raw top-left must land bottom-right after the 180-degree rotation
    observation = replace(observation, cameras={**observation.cameras, "agentview": agent})
    prepared = adapter.prepare_observation(observation)

    def reference(image):
        rotated = Image.fromarray(np.ascontiguousarray(image[::-1, ::-1]))
        return np.asarray(rotated.resize((224, 224), resample=Image.BILINEAR))

    rgb = np.concatenate([reference(agent), reference(observation.cameras["wrist"])], axis=1)
    expected = torch.tensor(rgb).permute(2, 0, 1).unsqueeze(0).float() * (2.0 / 255.0) - 1.0
    assert prepared.image.shape == (1, 3, 224, 448)
    torch.testing.assert_close(prepared.image, expected, rtol=0, atol=0)
    assert prepared.image[0, 0, -1, 223] > 0.9 and prepared.image[0, 1, -1, 223] < -0.9

    raw = np.concatenate([[0.1, -0.2, 0.3], [0.0, 0.0, 0.5], [0.02, -0.02]]).astype(np.float32)
    expected_state = (raw - 1.0) / 3.0  # min -2, max 4
    expected_state[0] = raw[0] - 0.5    # constant dimension: scale 1, offset -min
    np.testing.assert_allclose(prepared.proprio.numpy()[0], expected_state, atol=1e-6)
    assert prepared.proprio.dtype == torch.float32 and prepared.proprio.device.type == "cpu"


def test_libero_postprocess_denormalizes_and_maps_gripper():
    adapter = LongWAMLiberoAdapter(build_tiny_longwam(), data_config=libero_data_config(), dataset_stats=libero_stats())
    normalized = torch.linspace(-1, 1, 32 * 7).reshape(32, 7)
    commands = adapter.postprocess_actions(normalized, environment="libero")
    expected = normalized.numpy() * 3.0 + 1.0
    expected[:, 0] = normalized.numpy()[:, 0] + 0.5
    expected[:, -1] = np.sign(-(expected[:, -1] * 2 - 1))
    np.testing.assert_allclose(commands, expected, atol=1e-5)
    assert set(np.unique(commands[:, -1])) <= {-1.0, 0.0, 1.0}
    raw = LongWAMLiberoAdapter(build_tiny_longwam(), data_config=libero_data_config(), dataset_stats=libero_stats(),
                               binarize_gripper=False).postprocess_actions(normalized)
    np.testing.assert_allclose(raw[:, -1], -((normalized.numpy()[:, -1] * 3 + 1) * 2 - 1), atol=1e-5)
    with pytest.raises(ValueError):
        adapter.postprocess_actions(normalized, environment="robotwin")
    plus = LongWAMLiberoPlusAdapter(build_tiny_longwam(), data_config=libero_data_config(), dataset_stats=libero_stats())
    np.testing.assert_array_equal(plus.postprocess_actions(normalized, environment="libero_plus"), commands)


def test_libero_history_window_follows_episode_hooks(monkeypatch):
    model = build_tiny_longwam()
    adapter = LongWAMLiberoAdapter(model, data_config=libero_data_config((32, 32)), dataset_stats=libero_stats())
    seen = []
    infer = model.infer_joint_ar

    def record(**kwargs):
        seen.append(kwargs)
        return infer(**kwargs)

    monkeypatch.setattr(model, "infer_joint_ar", record)

    def request_for(observation):
        return InferenceRequest(adapter.prepare_observation(observation), condition="idm", num_inference_steps=2, seed=1)

    with pytest.raises(RuntimeError, match="begin_episode"):
        adapter.predict_action(request_for(libero_observation(0)))
    frames = [adapter.prepare_observation(libero_observation(step)).image[0] for step in range(61)]
    adapter.begin_episode(libero_observation(0))
    first = adapter.predict_action(request_for(libero_observation(0)))
    assert first.metadata["history"]["padded_frames"] == 48 and first.metadata["history"]["observed_frames"] == 1
    torch.testing.assert_close(seen[-1]["obs_window"], torch.stack([frames[0]] * 13, dim=1)[None], rtol=0, atol=0)
    for step in range(1, 61):
        adapter.observe_step(libero_observation(step), action=None)
        if step in (10, 60):
            result = adapter.predict_action(request_for(libero_observation(step)))
            indices = result.metadata["history"]["window_observation_indices"]
            assert indices == [max(0, step - 48 + offset) for offset in range(0, 49, 4)] and indices[-1] == step
            expected = torch.stack([frames[index] for index in indices], dim=1)[None]
            torch.testing.assert_close(seen[-1]["obs_window"], expected, rtol=0, atol=0)
            torch.testing.assert_close(seen[-1]["input_image"], frames[step][None], rtol=0, atol=0)
            assert result.actions.shape == (32, 7) and torch.isfinite(result.actions).all()
    with pytest.raises(ValueError, match="latest observed frame"):
        adapter.predict_action(request_for(libero_observation(59)))
    adapter.begin_episode(libero_observation(100))  # a new episode drops the old history
    assert adapter.predict_action(request_for(libero_observation(100))).metadata["history"]["padded_frames"] == 48


def test_shared_libero_rollout_calls_history_hooks():
    from wam_interp.evaluations._libero_rollout import run_libero_episode
    from wam_interp.evaluations.LIBERO.config import LiberoEvalConfig

    adapter = LongWAMLiberoAdapter(build_tiny_longwam(), data_config=libero_data_config((32, 32)), dataset_stats=libero_stats())
    counter = {"step": 0}

    def reset(env, initial_state, **kwargs):
        return libero_observation(0)

    def step(env, action, **kwargs):
        counter["step"] += 1
        return SimpleNamespace(observation=libero_observation(counter["step"]), reward=0.0, success=False, done=False)

    episode = run_libero_episode(
        model=adapter, env=SimpleNamespace(), task=SimpleNamespace(language="put the bowl on the plate"),
        task_id=0, trial_index=0, initial_state_index=0, initial_state=None, seed=0, max_steps=25,
        config=LiberoEvalConfig(num_steps_wait=0, replan_steps=10, inference_options={"num_inference_steps": 2}),
        reset_fn=reset, step_fn=step)
    assert episode.policy_steps == 25 and episode.inference_calls == 3
    history = [call["history"] for call in episode.inference_metadata]
    assert [item["observed_frames"] for item in history] == [1, 11, 21]
    assert [item["current_step"] for item in history] == [0, 10, 20]
    assert {call["condition"] for call in episode.inference_metadata} == {"idm"}


def test_robotwin_preprocessing_layout_state_and_decoding():
    import torch.nn.functional as F

    adapter = LongWAMRobotWinAdapter(build_tiny_longwam(action_dim=14, proprio_dim=14),
                                     data_config=robotwin_data_config(), dataset_stats=robotwin_stats())
    assert adapter.environment_action_type("robotwin") == "qpos"
    observation = robotwin_observation(0)
    prepared = adapter.prepare_observation(observation)
    assert prepared.image.shape == (1, 3, 384, 320)

    def resized(name, size):
        tensor = torch.from_numpy(observation.cameras[name]).permute(2, 0, 1).float() / 255.0
        return F.interpolate(tensor[None], size=size, mode="bilinear", align_corners=False, antialias=True)[0]

    # Raw 240x320 cameras: stage one (240x320) is the identity, stage two resizes to the mosaic slot.
    expected = torch.cat([resized("head", (256, 320)),
                          torch.cat([resized("left_wrist", (128, 160)), resized("right_wrist", (128, 160))], dim=2)], dim=1)
    torch.testing.assert_close(prepared.image[0], expected * 2.0 - 1.0, rtol=0, atol=1e-6)

    stats = robotwin_stats()["state"]["default"]
    joints = np.asarray(observation.state["joint_position"], dtype=np.float32)
    expected_state = (joints - np.float32(stats["global_mean"])) / (np.float32(stats["global_std"]) + 1e-8)
    np.testing.assert_allclose(prepared.proprio.numpy()[0], expected_state, atol=1e-5)
    normalized = torch.linspace(-1, 1, 32 * 14).reshape(32, 14)
    decoded = adapter.postprocess_actions(normalized, environment="robotwin")
    np.testing.assert_allclose(decoded, normalized.numpy() * np.float32(stats["global_std"]) + np.float32(stats["global_mean"]),
                               atol=1e-5)
    with pytest.raises(ValueError):
        adapter.environment_action_type("libero")


def test_robotwin_episode_inference_end_to_end():
    adapter = LongWAMRobotWinAdapter(build_tiny_longwam(action_dim=14, proprio_dim=14),
                                     data_config=robotwin_data_config(), dataset_stats=robotwin_stats())
    adapter.begin_episode(robotwin_observation(0))
    for step in range(1, 25):
        adapter.observe_step(robotwin_observation(step), action=None)
    request = InferenceRequest(adapter.prepare_observation(robotwin_observation(24)), condition="idm",
                               num_inference_steps=2, seed=0)
    result = adapter.predict_action(request)
    assert result.actions.shape == (32, 14) and torch.isfinite(result.actions).all()
    assert result.metadata["observation_window_shape"] == [1, 3, 13, 384, 320]
    assert result.metadata["history"]["observed_frames"] == 25
    assert adapter.postprocess_actions(result.actions, environment="robotwin").shape == (32, 14)


def test_adapter_rejects_mismatched_history_contract():
    config = libero_data_config()
    config["past_obs_size"] = 32
    with pytest.raises(ValueError, match="history"):
        LongWAMLiberoAdapter(build_tiny_longwam(), data_config=config, dataset_stats=libero_stats())
    with pytest.raises(ValueError, match="proprio_dim 8 != data state dim 14"):
        LongWAMRobotWinAdapter(build_tiny_longwam(), data_config=robotwin_data_config(), dataset_stats=robotwin_stats())


RELEASE_LIKE_CONFIG = """\
checkpoint: ./model.pt
mixed_precision: bf16
eval_num_inference_steps: 10
data:
  train:
    shape_meta:
      images:
      - {key: image, raw_shape: [3, 512, 512], shape: [3, 224, 224]}
      - {key: wrist_image, raw_shape: [3, 512, 512], shape: [3, 224, 224]}
      action: [{key: default, raw_shape: 7, shape: 7}]
      state: [{key: default, raw_shape: 8, shape: 8}]
    num_frames: 81
    past_obs_size: 48
    action_chunk: 32
    action_video_freq_ratio: 4
    video_size: [224, 448]
    concat_multi_camera: horizontal
    context_len: 128
    processor:
      _target_: fastwam.datasets.lerobot.processors.fastwam_processor.FastWAMProcessor
      num_output_cameras: 2
      action_state_transforms: null
      use_stepwise_action_norm: false
      norm_default_mode: min/max
      norm_exception_mode: null
model:
  _target_: fastwam.runtime.create_arwam
  model_id: Wan-AI/Wan2.2-TI2V-5B
  tokenizer_model_id: Wan-AI/Wan2.1-T2V-1.3B
  tokenizer_max_len: 128
  load_text_encoder: false
  proprio_dim: 8
  redirect_common_files: false
  mot_checkpoint_mixed_attn: true
  skip_dit_load_from_pretrain: true
  longlive_video_weights: null
  action_dit_pretrained_path: null
  arwam_num_frame_per_block: 1
  arwam_local_attn_size: -1
  arwam_sink_size: 0
  arwam_past_obs_size: 48
  arwam_num_imagine_frames: 2
  arwam_imagine_sigma: 0.9
  arwam_imagine_infer_steps: 4
  video_dit_config: {has_image_input: false, patch_size: [1, 2, 2], in_dim: 48, hidden_dim: 3072, text_dim: 4096,
                     video_attention_mask_mode: per_frame_causal, action_conditioned: false}
  action_dit_config: {action_dim: 7, hidden_dim: 1024}
  video_scheduler: {train_shift: 5.0, infer_shift: 5.0, num_train_timesteps: 1000}
  action_scheduler: {train_shift: 5.0, infer_shift: 5.0, num_train_timesteps: 1000}
"""


def test_config_name_migration_matches_upstream():
    from wam_interp.models.longwam.loader import migrate_config_names
    migrated = migrate_config_names({"_target_": "fastwam.runtime.create_arwam", "arwam_sink_size": 0,
                                     "x": ["fastwam.models.wan22.ar_wam.ARWAM", "${model.arwam_past_obs_size}", "./fastwam.pt"]})
    assert migrated == {"_target_": "longwam.runtime.create_longwam", "longwam_sink_size": 0,
                        "x": ["longwam.models.wan22.long_wam.LongWAM", "${model.longwam_past_obs_size}", "./fastwam.pt"]}


def test_loader_builds_release_config_and_loads_checkpoint_strictly(tmp_path, monkeypatch):
    from wam_interp.models import load_model_adapter
    from wam_interp.models.longwam.loader import load_longwam
    from wam_interp.models.longwam.model_code.models.wan22.long_wam import LongWAM

    config = tmp_path / "config.yaml"
    config.write_text(RELEASE_LIKE_CONFIG)
    source = build_tiny_longwam(seed=7)
    checkpoint = tmp_path / "model.pt"
    source.save_checkpoint(checkpoint, step=21700)
    captured = {}

    def fake_from_pretrained(**kwargs):
        captured.clear()
        captured.update(kwargs)
        return build_tiny_longwam(seed=1)

    monkeypatch.setattr(LongWAM, "from_wan22_pretrained", fake_from_pretrained)
    model = load_longwam(config, checkpoint, device="cpu", dtype="float32")
    assert captured["longwam_past_obs_size"] == 48 and captured["longwam_num_imagine_frames"] == 2
    assert captured["longwam_imagine_sigma"] == 0.9 and captured["longwam_joint_denoise"] is False
    assert captured["load_text_encoder"] is True and captured["tokenizer_max_len"] == 128
    assert captured["skip_video_dit_load_from_pretrain"] is True and captured["skip_action_dit_load_from_pretrain"] is True
    assert captured["redirect_common_files"] is False and captured["action_dit_pretrained_path"] is None
    assert captured["video_dit_config"]["hidden_dim"] == 3072 and captured["torch_dtype"] == torch.float32
    for name, tensor in source.mot.state_dict().items():
        torch.testing.assert_close(model.mot.state_dict()[name], tensor, rtol=0, atol=0)
    meta = model._wam_interp_metadata
    assert meta["upstream_revision"] == "242635807b4ebb89fa2609a84df7eacde6aed92f"
    assert meta["checkpoint_step"] == 21700 and meta["checkpoint_strict"] is True and meta["released_bundle"] is None
    assert meta["checkpoint_load"]["missing_keys"] == 0 and "proprio_encoder" in meta["checkpoint_load"]["modules_loaded"]
    assert meta["data_config"]["past_obs_size"] == 48 and meta["evaluation"]["num_inference_steps"] == 10
    assert meta["resolved_model_config"]["_target_"] == "wam_interp.models.longwam.runtime.create_longwam"
    assert LongWAMAdapter(model).inference_defaults() == {"condition": "idm", "num_inference_steps": 10}
    # The benchmark adapter defaults to the bundle's own data config.
    assert LongWAMLiberoAdapter(model, dataset_stats=libero_stats()).video_size == (224, 448)

    payload = torch.load(checkpoint)
    payload["mot"].pop(next(iter(payload["mot"])))
    partial = tmp_path / "partial.pt"
    torch.save(payload, partial)
    with pytest.raises(RuntimeError, match="Missing key"):
        load_longwam(config, partial, device="cpu", dtype="float32")
    loaded = load_longwam(config, partial, device="cpu", dtype="float32", allow_partial_checkpoint=True)
    assert loaded._wam_interp_metadata["checkpoint_load"]["problems"]
    payload = torch.load(checkpoint)
    del payload["proprio_encoder"]
    torch.save(payload, partial)
    with pytest.raises(ValueError, match="proprio_encoder"):
        load_longwam(config, partial, device="cpu", dtype="float32")
    with pytest.raises(FileNotFoundError):
        load_model_adapter("longwam", bundle=tmp_path / "missing")
