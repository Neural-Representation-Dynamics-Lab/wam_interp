"""CPU checks for the OpenWAM backend.

Light tests (preprocessing, action conversion, adapter contract, loader helpers) use stub models
and run in the shared environment. The ``test_tiny_*`` tests build a tiny random-weight OpenWAM
through the real loader and engine; they need the openwam extra (ftfy, modelscope, torchvision,
transformers) and are skipped without it.
"""

import io
import json
import pickle
from functools import partialmethod
from types import SimpleNamespace
from unittest import mock

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf
from PIL import Image

from wam_interp.models.openwam import loader
from wam_interp.models.openwam.adapter import OpenWAMAdapter, image_to_pil
from wam_interp.models.openwam.libero_adapter import OpenWAMLiberoAdapter, native_eef10_to_libero7d
from wam_interp.models.openwam.libero_plus_adapter import OpenWAMLiberoPlusAdapter
from wam_interp.models.openwam.model_code.benchmarks.utils import action_conversion as ac
from wam_interp.models.openwam.model_code.benchmarks.utils.client import resize_for_lshape_slot
from wam_interp.models.openwam.model_code.dataloader.transforms.normalize import Normalizer
from wam_interp.models.openwam.model_code.dataloader.utils.unify_action import (
    map_to_unify, parse_unify_spec, unmap_from_unify,
)
from wam_interp.models.openwam.robotwin_adapter import OpenWAMRobotWinAdapter
from wam_interp.models.base import EvaluationModelAdapter, ModelAdapter
from wam_interp.types import EnvironmentObservation, InferenceRequest, PreparedObservation

PROMPT_PREFIX = "A video recorded from a robot's point of view executing the following instruction: "


# --------------------------------------------------------------------------- helpers

def _axis_angle_to_quat_xyzw(rotvec):
    angle = np.linalg.norm(rotvec)
    if angle < 1e-12:
        return np.array([0.0, 0.0, 0.0, 1.0])
    axis = rotvec / angle
    return np.concatenate([axis * np.sin(angle / 2), [np.cos(angle / 2)]])


def _random_quats(rng, n):
    q = rng.normal(size=(n, 4))
    return q / np.linalg.norm(q, axis=1, keepdims=True)


def _same_rotation(q1, q2, atol=1e-5):
    return abs(abs(float(np.dot(q1, q2))) - 1.0) < atol


class UnifiedMinMax:
    """Same composition as upstream deploy/model_loader.py::_UnifyAwareNormalizer, for stubs."""

    def __init__(self, action_stats, state_stats, spec, unify_dim=80):
        self._inner = Normalizer(mode="min_max", stats={**action_stats, "_normalize_stats": state_stats})
        self._dst_index = parse_unify_spec(spec, unify_dim)
        self._unify_dim = unify_dim

    def normalize(self, x):
        return map_to_unify(self._inner.normalize(np.asarray(x)), self._dst_index, self._unify_dim)[0]

    def unnormalize(self, x):
        return self._inner.unnormalize(unmap_from_unify(np.asarray(x), self._dst_index))


def _stats(dim, low, high):
    return {"min": np.full(dim, low, np.float32), "max": np.full(dim, high, np.float32)}


def _cfg(dataset):
    layout = (["observation.images.image", "observation.images.image2", None] if dataset == "libero"
              else ["head_camera", "left_camera", "right_camera"])
    return OmegaConf.create({
        "model": {"architecture": {"use_proprioception": True, "attention_mask_mode": "mutual"}},
        "dataloader": {"type": dataset, "action_mode": "eef", "multiview": True, "camera_layout": layout,
                       "height": 384, "width": 320, "num_frames": 33, "video_stride": 4},
        "inference": {"num_frames": 33, "video_num_frames": 9, "denoise_steps": 10, "height": 384, "width": 320},
    })


class StubEngine:
    def __init__(self, architecture):
        self.architecture = architecture
        self.calls = []

    def generate(self, conditions):
        self.calls.append(conditions)
        proprio = self.architecture.normalize_deploy_proprio(conditions["proprio"])
        actions = np.tile(proprio.numpy(), (int(conditions.get("num_frames", 33)) - 1, 1))
        return {"video": None, "actions": self.architecture.normalizer.unnormalize(actions)}


def _stub_model(dataset):
    raw = 10 if dataset == "libero" else 20
    spec = ["0-9"] if dataset == "libero" else ["0-9", "34-43"]
    normalizer = UnifiedMinMax(_stats(raw, -0.9, 0.9), _stats(raw, -1.5, 1.5), spec)

    def normalize_deploy_proprio(proprio):  # upstream BaseWAMArchitecture.normalize_deploy_proprio
        return torch.from_numpy(architecture.normalizer.normalize(np.asarray(proprio, dtype=np.float32)))

    architecture = SimpleNamespace(normalizer=normalizer, action_dim=80, uses_proprioception=True,
                                   binary_command_dims=(), device=torch.device("cpu"), dtype=torch.bfloat16,
                                   normalize_deploy_proprio=normalize_deploy_proprio)
    return SimpleNamespace(cfg=_cfg(dataset), architecture=architecture, engine=StubEngine(architecture),
                           device=torch.device("cpu"), dtype=torch.bfloat16,
                           _wam_interp_metadata={"checkpoint": "stub", "upstream_revision": loader.UPSTREAM_REVISION})


def _libero_observation(rng, size=256):
    return EnvironmentObservation(
        cameras={"agentview": rng.integers(0, 256, (size, size, 3), dtype=np.uint8),
                 "wrist": rng.integers(0, 256, (size, size, 3), dtype=np.uint8)},
        state={"eef_position": np.array([0.1, -0.2, 1.0]), "eef_quaternion": _random_quats(rng, 1)[0],
               "gripper_position": np.array([0.03, -0.03])},
        instruction="put the bowl on the plate", metadata={"task_id": 0})


def _robotwin_observation(rng):
    def endpose():
        return np.concatenate([rng.uniform(-0.5, 0.5, 3), _random_quats(rng, 1)[0]])
    return EnvironmentObservation(
        cameras={"head": rng.integers(0, 256, (240, 320, 3), dtype=np.uint8),
                 "left_wrist": rng.integers(0, 256, (240, 320, 3), dtype=np.uint8),
                 "right_wrist": rng.integers(0, 256, (240, 320, 3), dtype=np.uint8)},
        state={"left_endpose": endpose(), "right_endpose": endpose(), "left_gripper": 0.7, "right_gripper": 0.1,
               "joint_position": np.zeros(14)},
        instruction="beat the block with the hammer")


def _pixels(image):
    return np.asarray(image_to_pil(image))


# --------------------------------------------------------------------------- action conversion

def test_rot6d_axis_angle_round_trip():
    rng = np.random.default_rng(0)
    for _ in range(200):
        axis = rng.normal(size=3)
        rotvec = axis / np.linalg.norm(axis) * rng.uniform(0.0, np.pi - 1e-3)
        rot6d = ac.quat_xyzw_to_rot6d(_axis_angle_to_quat_xyzw(rotvec))
        np.testing.assert_allclose(ac.rot6d_to_axis_angle(rot6d), rotvec, atol=2e-5)
    np.testing.assert_array_equal(ac.rot6d_to_axis_angle(np.array([1, 0, 0, 0, 1, 0.0])), np.zeros(3))


def test_quat_rot6d_round_trip_is_sign_invariant():
    rng = np.random.default_rng(1)
    for quat in _random_quats(rng, 200):
        rot6d = ac.quat_xyzw_to_rot6d(quat)
        assert _same_rotation(ac.rot6d_to_quat_xyzw(rot6d), quat)
        np.testing.assert_allclose(ac.quat_xyzw_to_rot6d(-quat), rot6d, atol=1e-6)


def test_robotwin_endpose_eef20_ee16_round_trip():
    rng = np.random.default_rng(2)
    left, right = np.concatenate([[0.1, 0.2, 0.9], _random_quats(rng, 1)[0]]), np.concatenate(
        [[-0.1, 0.3, 1.1], _random_quats(rng, 1)[0]])
    eef20 = ac.robotwin_endpose_to_eef20d(left, right, 0.25, [0.75])
    assert eef20.shape == (20,) and eef20.dtype == np.float32
    ee16 = ac.eef20d_to_ee16d(eef20)
    assert ee16.shape == (16,)
    np.testing.assert_allclose(ee16[[0, 1, 2, 7, 8, 9, 10, 15]], [0.1, 0.2, 0.9, 0.25, -0.1, 0.3, 1.1, 0.75], atol=1e-6)
    assert _same_rotation(ee16[3:7], left[3:]) and _same_rotation(ee16[11:15], right[3:])


def test_native_eef10_to_libero7d_matches_upstream_conventions():
    identity = np.array([0.5, -0.25, 2.0, 1, 0, 0, 0, 1, 0, 1.0], np.float32)
    np.testing.assert_allclose(native_eef10_to_libero7d(identity), [0.5, -0.25, 1.0, 0, 0, 0, -1.0])  # clip + open->-1
    closed = identity.copy()
    closed[9] = -1.0
    assert native_eef10_to_libero7d(closed)[6] == 1.0  # LIBERO +1 closes
    rotvec = np.array([0.3, -0.2, 0.1])
    action = np.concatenate([[0, 0, 0], ac.quat_xyzw_to_rot6d(_axis_angle_to_quat_xyzw(rotvec)), [0.0]])
    np.testing.assert_allclose(native_eef10_to_libero7d(action)[3:6], rotvec, atol=2e-5)
    with pytest.raises(ValueError):
        native_eef10_to_libero7d(np.full(10, np.nan))


def test_libero_state_eef10_layout():
    quat = np.array([0.0, 0.0, np.sin(0.2), np.cos(0.2)])
    state = ac.libero_obs_to_eef10([0.1, 0.2, 0.3], quat, [0.04, -0.04])
    np.testing.assert_allclose(state[:3], [0.1, 0.2, 0.3], atol=1e-7)
    np.testing.assert_allclose(state[3:9], ac.quat_xyzw_to_rot6d(quat), atol=1e-7)
    assert state[9] == pytest.approx(1.0, abs=1e-6)  # 0.08 m finger separation = fully open = +1 (float32)
    assert ac.libero_obs_to_eef10([0, 0, 0], quat, [0.0, 0.0])[9] == -1.0


# --------------------------------------------------------------------------- preprocessing (stub model)

@pytest.mark.parametrize("wrist", ["real", "black"])
def test_libero_composite_layout_state_and_decode(wrist):
    rng = np.random.default_rng(3)
    observation = _libero_observation(rng)
    adapter = OpenWAMLiberoAdapter(_stub_model("libero"), wrist=wrist)
    assert isinstance(adapter, EvaluationModelAdapter)
    prepared = adapter.prepare_observation(observation)
    assert tuple(prepared.image.shape) == (3, 384, 320) and prepared.image.dtype == torch.float32
    pixels = _pixels(prepared.image)
    head = resize_for_lshape_slot(observation.cameras["agentview"][::-1, ::-1], "head_camera")
    np.testing.assert_array_equal(pixels[:256], head)
    if wrist == "real":
        np.testing.assert_array_equal(pixels[256:, :160],
                                      resize_for_lshape_slot(observation.cameras["wrist"][::-1, ::-1], "left_wrist_camera"))
    else:
        assert not pixels[256:, :160].any()
    assert not pixels[256:, 160:].any()  # camera_layout[2] is null -> black
    raw = ac.libero_obs_to_eef10(observation.state["eef_position"], observation.state["eef_quaternion"],
                                 observation.state["gripper_position"])
    expected = np.zeros(80, np.float32)
    expected[:10] = np.clip((raw + 1.5) / 3.0 * 2 - 1, -1, 1)  # eef_state min-max into slots 0..9
    np.testing.assert_allclose(prepared.proprio.numpy(), expected, atol=1e-6)
    assert prepared.prompt == observation.instruction and prepared.metadata["wrist"] == wrist

    result = adapter.predict_action(InferenceRequest(prepared, **adapter.inference_defaults()))
    assert result.actions.shape == (32, 80) and result.metadata["action_space"] == "model (normalized)"
    commands = adapter.postprocess_actions(result.actions, environment="libero")
    assert commands.shape == (32, 7) and commands.dtype == np.float32
    np.testing.assert_allclose(commands[0], native_eef10_to_libero7d(adapter.denormalize(result.actions)[0]))
    with pytest.raises(ValueError):
        adapter.postprocess_actions(result.actions, environment="libero_plus")
    plus = OpenWAMLiberoPlusAdapter(_stub_model("libero"))
    assert plus.postprocess_actions(result.actions, environment="libero_plus").shape == (32, 7)


def test_robotwin_composite_prompt_state_and_ee_commands():
    rng = np.random.default_rng(4)
    observation = _robotwin_observation(rng)
    adapter = OpenWAMRobotWinAdapter(_stub_model("robotwin"))
    assert isinstance(adapter, EvaluationModelAdapter) and isinstance(OpenWAMAdapter(None), ModelAdapter)
    assert adapter.environment_action_type("robotwin") == "ee"
    prepared = adapter.prepare_observation(observation)
    pixels = _pixels(prepared.image)
    bilinear = lambda name, w, h: np.asarray(Image.fromarray(observation.cameras[name]).resize((w, h), Image.BILINEAR))
    np.testing.assert_array_equal(pixels[:256], bilinear("head", 320, 256))
    np.testing.assert_array_equal(pixels[256:, :160], bilinear("left_wrist", 160, 128))
    np.testing.assert_array_equal(pixels[256:, 160:], bilinear("right_wrist", 160, 128))
    assert prepared.prompt == PROMPT_PREFIX + observation.instruction
    eef20 = ac.robotwin_endpose_to_eef20d(observation.state["left_endpose"], observation.state["right_endpose"],
                                          0.7, 0.1)
    unified = prepared.proprio.numpy()
    np.testing.assert_allclose(unified[np.r_[0:10, 34:44]], np.clip((eef20 + 1.5) / 3.0 * 2 - 1, -1, 1), atol=1e-6)
    assert not np.delete(unified, np.r_[0:10, 34:44]).any()

    result = adapter.predict_action(InferenceRequest(prepared, **adapter.inference_defaults()))
    commands = adapter.postprocess_actions(result.actions, environment="robotwin")
    assert commands.shape == (32, 16)
    np.testing.assert_allclose(commands[0], ac.eef20d_to_ee16d(adapter.denormalize(result.actions)[0]))
    with pytest.raises(ValueError):
        adapter.environment_action_type("libero")


def test_predict_action_passes_model_space_proprio_and_returns_model_space_actions():
    adapter = OpenWAMLiberoAdapter(_stub_model("libero"))
    prepared = adapter.prepare_observation(_libero_observation(np.random.default_rng(5)))
    original = adapter.architecture.normalizer
    result = adapter.predict_action(InferenceRequest(prepared, **adapter.inference_defaults()))
    conditions = adapter.model.engine.calls[-1]
    assert adapter.architecture.normalizer is original  # restored after the call
    assert conditions["seed"] == 42 and conditions["denoise_steps"] == 10 and conditions["prompt"] == prepared.prompt
    np.testing.assert_array_equal(np.asarray(conditions["first_frame_image"][0]), _pixels(prepared.image))
    # The stub engine echoes normalized proprio as actions; with the pass-through normalizer they stay in model space.
    np.testing.assert_allclose(result.actions[0], prepared.proprio.numpy(), atol=1e-6)


def test_adapter_rejects_requests_upstream_cannot_honor():
    adapter = OpenWAMLiberoAdapter(_stub_model("libero"))
    prepared = adapter.prepare_observation(_libero_observation(np.random.default_rng(6)))
    defaults = adapter.inference_defaults()
    assert defaults == {"condition": "joint", "num_video_frames": 9, "num_inference_steps": 10, "seed": 42}
    bad = [{"condition": "first_frame"}, {"action_horizon": 16}, {"num_video_frames": 33}, {"rand_device": "cuda"},
           {"sigma_shift": 3.0}, {"text_cfg_scale": 2.0}, {"negative_prompt": "x"}, {"compile_action_infer": True},
           {"num_inference_steps": 0}]
    for override in bad:
        with pytest.raises(ValueError):
            adapter.predict_action(InferenceRequest(prepared, **{**defaults, **override}))
    without_state = PreparedObservation(prepared.image, None, prepared.prompt)
    with pytest.raises(ValueError, match="proprio"):
        adapter.predict_action(InferenceRequest(without_state, **defaults))


def test_condition_hook_edits_engine_conditions():
    adapter = OpenWAMLiberoAdapter(_stub_model("libero"))
    prepared = adapter.prepare_observation(_libero_observation(np.random.default_rng(7)))
    adapter.condition_hook = lambda conditions: {**conditions, "prompt": "edited", "seed": 7}
    result = adapter.predict_action(InferenceRequest(prepared, **adapter.inference_defaults()))
    assert adapter.model.engine.calls[-1]["prompt"] == "edited" and result.metadata["seed"] == 7
    assert result.metadata["condition_hook"] is True


def test_image_to_pil_round_trips_uint8_grid():
    pixels = np.random.default_rng(8).integers(0, 256, (384, 320, 3), dtype=np.uint8)
    tensor = torch.from_numpy(pixels).permute(2, 0, 1).float() / 127.5 - 1.0
    np.testing.assert_array_equal(np.asarray(image_to_pil(tensor)), pixels)
    np.testing.assert_array_equal(np.asarray(image_to_pil(pixels)), pixels)
    with pytest.raises(ValueError):
        image_to_pil(tensor * 3)


def test_adapters_reject_wrong_checkpoint_family():
    with pytest.raises(ValueError):
        OpenWAMLiberoAdapter(_stub_model("robotwin"))
    with pytest.raises(ValueError):
        OpenWAMRobotWinAdapter(_stub_model("libero"))


# --------------------------------------------------------------------------- loader helpers

def _npy_object(payload_pickle):
    buffer = io.BytesIO()
    header = {"descr": "|O", "fortran_order": False, "shape": ()}
    np.lib.format.write_array_header_1_0(buffer, header)
    return buffer.getvalue() + payload_pickle


def test_normalization_stats_reader_blocks_arbitrary_pickles(tmp_path):
    good = tmp_path / "good.npy"
    np.save(good, {"eef": {"min": np.zeros(3, np.float32), "max": np.ones(3, np.float32)}}, allow_pickle=True)
    assert set(loader.read_normalization_stats(good)["eef"]) == {"min", "max"}

    class Exploit:
        def __reduce__(self):
            return (print, ("pwned",))

    bad = tmp_path / "bad.npy"
    bad.write_bytes(_npy_object(pickle.dumps(Exploit(), protocol=3)))
    with pytest.raises(pickle.UnpicklingError, match="builtins.print"):
        loader.read_normalization_stats(bad)


def test_config_rewrite_and_research_overrides():
    config = {"model": {"video_backbone": {"components": [
        {"model_class": "openwam.model.video_backbone.wan.models.dit.WanModel"}],
        "tokenizer": {"class": "openwam.model.video_backbone.wan.models.text_encoder.HuggingfaceTokenizer"}}},
        "other": "openwam-serve"}
    rewritten = loader._rewrite_class_paths(config)
    assert rewritten["model"]["video_backbone"]["components"][0]["model_class"] == \
        "wam_interp.models.openwam.model_code.model.video_backbone.wan.models.dit.WanModel"
    assert rewritten["model"]["video_backbone"]["tokenizer"]["class"].startswith("wam_interp.models.openwam.model_code.")
    assert rewritten["other"] == "openwam-serve"
    training = OmegaConf.create({"dataloader": {"num_frames": 33, "video_stride": 4, "height": 384, "width": 320},
                                 "project": {"output_dir": "outputs/${now:%Y}"}})
    merged = loader.merge_deploy_config(training, loader.research_config_overrides())
    assert OmegaConf.to_container(merged.inference) == {
        "denoise_steps": 10, "denoise_mode": "sync", "lead_modality": "video", "variance_shift_alpha": 1.0,
        "linear_offset": 0.0, "num_frames": 33, "video_num_frames": 9, "height": 384, "width": 320}
    assert merged.optimization.dit_cache.enabled is False and merged.optimization.compile.enabled is False
    assert merged.optimization.decode_video is False
    upstream = loader.merge_deploy_config(training, loader.research_config_overrides(upstream_optimizations=True))
    assert OmegaConf.to_container(upstream.optimization.dit_cache) == {"enabled": True, "cosine_threshold": 0.99,
                                                                       "max_skips": 3}
    assert upstream.optimization.compile.enabled is True and upstream.optimization.compile.idm.video_loop.enabled
    assert upstream.optimization.decode_video is False and upstream.optimization.prompt_embed_cache.enabled


def test_upstream_optimizations_refuse_tracing_and_hooks():
    rng = np.random.default_rng(3)
    model = _stub_model("libero")
    model._wam_interp_metadata["upstream_optimizations"] = True
    adapter = OpenWAMLiberoAdapter(model)
    prepared = adapter.prepare_observation(_libero_observation(rng))
    defaults = adapter.inference_defaults()
    adapter.predict_action(InferenceRequest(prepared, **defaults))  # plain evaluation still works
    for change in ({"trace_modules": ("video_backbone",)}, {"condition": "isolated"}):
        with pytest.raises(ValueError, match="upstream_optimizations"):
            adapter.predict_action(InferenceRequest(prepared, **{**defaults, **change}))
    adapter.condition_hook = lambda conditions: conditions
    with pytest.raises(ValueError, match="upstream_optimizations"):
        adapter.predict_action(InferenceRequest(prepared, **defaults))


def test_checkpoint_file_resolution(tmp_path):
    with pytest.raises(FileNotFoundError):
        loader.resolve_checkpoint_file(tmp_path)
    for step in (9, 10690, 200):
        (tmp_path / f"checkpoint_step_{step}.safetensors").write_bytes(b"")
    (tmp_path / "checkpoint_step_x.safetensors").write_bytes(b"")
    assert loader.resolve_checkpoint_file(tmp_path).name == "checkpoint_step_10690.safetensors"
    assert loader.resolve_checkpoint_file(tmp_path, "checkpoint_step_9.safetensors").name.endswith("_9.safetensors")
    with pytest.raises(FileNotFoundError):
        loader.load_openwam(tmp_path, device="cpu")  # no config.yaml


# --------------------------------------------------------------------------- tiny random model (runtime)

def _require_runtime():
    for name in ("ftfy", "modelscope", "torchvision", "transformers", "tokenizers", "safetensors"):
        pytest.importorskip(name, reason=f"OpenWAM model code needs the 'openwam' extra ({name} missing)")


def _tiny_config(*, dataset, variant, unify, mask):
    raw = 10 if dataset == "libero" else 20
    action_dim = 80 if unify else raw
    dataloader = {"type": dataset, "action_mode": "eef", "unify_action": unify, "unify_state_map": None,
                  "num_frames": 33, "video_stride": 4, "height": 384, "width": 320, "multiview": True,
                  "normalize_mode": "min-max"}
    if dataset == "libero":
        dataloader.update(unify_action_map=["0-9"], target_camera="observation.images.image",
                          camera_layout=["observation.images.image", "observation.images.image2", None])
    else:
        dataloader.update(unify_action_map=["0-9", "34-43"], target_camera="head_camera",
                          camera_layout=["head_camera", "left_camera", "right_camera"])
    wan = "openwam.model.video_backbone.wan.models"
    return {
        "model": {
            "video_backbone": {
                "encoder": {"name": "wan22_vae", "model_path": "/path/to/Wan2.2-TI2V-5B"},
                "name": "wan22_ti2v_5b", "model_path": "/path/to/Wan2.2-TI2V-5B", "from_scratch": False,
                "shift_video": 5.0,
                "components": [
                    {"attr": "dit", "model_class": f"{wan}.dit.WanModel", "extra_kwargs": {
                        "has_image_input": False, "patch_size": [1, 2, 2], "in_dim": 48, "dim": 64, "ffn_dim": 128,
                        "freq_dim": 32, "text_dim": 32, "out_dim": 48, "num_heads": 2, "num_layers": 2, "eps": 1e-6,
                        "seperated_timestep": True, "require_clip_embedding": False, "require_vae_embedding": False,
                        "fuse_vae_embedding_in_latents": True}},
                    {"attr": "vae", "model_class": f"{wan}.vae.WanVideoVAE38", "extra_kwargs": {"dim": 8}},
                    {"attr": "text_encoder", "model_class": f"{wan}.text_encoder.WanTextEncoder", "extra_kwargs": {
                        "vocab": 64, "dim": 32, "dim_attn": 32, "dim_ffn": 64, "num_heads": 2, "num_layers": 1}}],
                "tokenizer": {"class": f"{wan}.text_encoder.HuggingfaceTokenizer", "attr": "tokenizer",
                              "subdir": "tokenizer/google/umt5-xxl", "path_kwarg": "name",
                              "kwargs": {"seq_len": 512, "clean": "whitespace"}}},
            "action_backbone": {"dim": 32, "ffn_dim": 64, "shift_action": 5.0},
            "architecture": {"framework": "dual_system", "variant": variant, "action_dim": action_dim,
                             "use_proprioception": True, "state_dim": action_dim, "bridge_layers": None,
                             "bridge_interval": 1, "mot_checkpoint_mixed_attn": True, "attention_mask_mode": mask,
                             "video_attention_mask_mode": "first_frame_causal", "detach_bridge": False,
                             "idm_video_cond_noise_prob": 0.5}},
        "dataloader": dataloader,
        "training": {"mixed_precision": "bf16"},
        "project": {"output_dir": "outputs/${now:%Y-%m-%d_%H-%M-%S}"},
    }, raw


def _write_tokenizer(directory):
    from tokenizers import Tokenizer, models, pre_tokenizers

    words = ("<pad> </s> <unk> a the put pick up on bowl plate beat block with hammer video recorded from robot "
             "point of view executing following instruction").split()
    tokenizer = Tokenizer(models.WordLevel({word: i for i, word in enumerate(words)}, unk_token="<unk>"))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    directory.mkdir(parents=True)
    tokenizer.save(str(directory / "tokenizer.json"))
    (directory / "tokenizer_config.json").write_text(json.dumps({
        "tokenizer_class": "PreTrainedTokenizerFast", "pad_token": "<pad>", "eos_token": "</s>",
        "unk_token": "<unk>", "model_max_length": 512}))


@pytest.fixture(scope="module")
def tiny_vae():
    """The Wan2.2 VAE wrapper hard-codes a 256-wide decoder; decode is off, so only shrink that."""
    _require_runtime()
    from wam_interp.models.openwam.model_code.model.video_backbone.wan.models.vae import VideoVAE38_

    with mock.patch.object(VideoVAE38_, "__init__", partialmethod(VideoVAE38_.__init__, dec_dim=8)):
        yield


def _tiny_checkpoint(root, *, dataset="libero", variant="joint_self_attn", unify=True, mask="mutual"):
    import yaml
    from safetensors.torch import save_file
    from wam_interp.models.openwam.model_code.model.architectures.base import BaseWAMArchitecture

    config, raw = _tiny_config(dataset=dataset, variant=variant, unify=unify, mask=mask)
    root.mkdir(parents=True)
    (root / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
    _write_tokenizer(root / "tokenizer/google/umt5-xxl")
    ones, zeros = np.ones(raw, np.float32), np.zeros(raw, np.float32)
    stats = {"eef": {"min": -0.9 * ones, "max": 0.9 * ones, "mean": zeros, "std": ones}}
    if dataset == "libero":  # the released LIBERO stats carry a separate proprio block
        stats["eef_state"] = {"min": -1.5 * ones, "max": 1.5 * ones, "mean": zeros, "std": ones}
    np.save(root / "normalization_stats.npy", stats, allow_pickle=True)
    weights = root / "checkpoint_step_1.safetensors"
    weights.write_bytes(b"")
    with mock.patch.object(BaseWAMArchitecture, "load_checkpoint", lambda self, path, strict=True: None), \
            mock.patch.object(loader, "check_architecture_load", lambda records, architecture, checkpoint: {}):
        architecture = loader.load_openwam(root, device="cpu").architecture
    generator = torch.Generator().manual_seed(0)
    with torch.no_grad():
        for parameter in architecture.parameters():
            parameter.add_((torch.randn(parameter.shape, generator=generator) * 0.02).to(parameter.dtype))
    save_file({key: value.contiguous().clone() for key, value in architecture.state_dict().items()}, str(weights))
    return root


@pytest.fixture(scope="module")
def tiny_libero(tiny_vae, tmp_path_factory):
    root = _tiny_checkpoint(tmp_path_factory.mktemp("openwam") / "libero")
    return root, loader.load_openwam(root, device="cpu")


def _upstream_engine_call(model, payload, seed=42):
    """What upstream's policy server sends to the engine: raw state, PIL composite, prompt."""
    return model.engine.generate({"first_frame_image": [payload["image"]], "prompt": payload["prompt"],
                                  "proprio": payload["state"], "seed": seed})["actions"]


def _payload(adapter, prepared, raw_state):
    return {"image": image_to_pil(prepared.image), "prompt": prepared.prompt, "state": np.asarray(raw_state, np.float32)}


def test_tiny_upstream_optimizations_enable_the_dit_cache(tiny_libero, monkeypatch):
    """The DiT cache module is vendored and wired; compile is left to the GPU run (slow on CPU)."""
    root, exact = tiny_libero
    no_compile = {**loader.UPSTREAM_DEPLOY_OPTIMIZATIONS, "compile": {"enabled": False}}
    monkeypatch.setattr(loader, "UPSTREAM_DEPLOY_OPTIMIZATIONS", no_compile)
    model = loader.load_openwam(root, device="cpu", upstream_optimizations=True)
    assert type(model.engine._dit_cache).__name__ == "DiTVelocityCache"
    assert model._wam_interp_metadata["upstream_optimizations"] is True
    adapter = OpenWAMLiberoAdapter(model)
    prepared = adapter.prepare_observation(_libero_observation(np.random.default_rng(5)))
    actions = adapter.predict_action(InferenceRequest(prepared, **adapter.inference_defaults())).actions
    reference = OpenWAMLiberoAdapter(exact).predict_action(
        InferenceRequest(prepared, **OpenWAMLiberoAdapter(exact).inference_defaults())).actions
    assert actions.shape == reference.shape and np.isfinite(actions).all()


def test_tiny_checkpoint_loads_strictly_with_research_settings(tiny_libero, tmp_path):
    root, model = tiny_libero
    summary = model._wam_interp_metadata["checkpoint_load"]
    assert summary["missing_keys"] == 0 and summary["unexpected_keys"] == 0
    assert set(summary["parameter_count_by_dtype"]) == {"torch.bfloat16"}
    assert model.engine._dit_cache is None and model.engine._decode_video is False
    assert model.cfg.model.video_backbone.model_path is None
    assert model._wam_interp_metadata["upstream_revision"] == loader.UPSTREAM_REVISION

    from safetensors.torch import load_file, save_file
    weights = load_file(str(root / "checkpoint_step_1.safetensors"))
    for name, edit in (("partial", lambda w: w.pop(sorted(w)[0])), ("extra", lambda w: w.update(extra=torch.zeros(1)))):
        broken = tmp_path / name
        broken.mkdir()
        for entry in root.iterdir():
            if entry.name != "checkpoint_step_1.safetensors":
                (broken / entry.name).symlink_to(entry)
        changed = dict(weights)
        edit(changed)
        save_file(changed, str(broken / "checkpoint_step_1.safetensors"))
        with pytest.raises(RuntimeError, match="Strict load failed|did not load completely"):
            loader.load_openwam(broken, device="cpu")


@pytest.mark.parametrize("wrist", ["real", "black"])
def test_tiny_libero_matches_upstream_engine_call_bitwise(tiny_libero, wrist):
    _, model = tiny_libero
    adapter = OpenWAMLiberoAdapter(model, wrist=wrist)
    observation = _libero_observation(np.random.default_rng(10))
    prepared = adapter.prepare_observation(observation)
    result = adapter.predict_action(InferenceRequest(prepared, **adapter.inference_defaults()))
    assert result.actions.shape == (32, 80) and result.actions.dtype == np.float32
    upstream = _upstream_engine_call(model, _payload(adapter, prepared, prepared.metadata["raw_state_eef10"]))
    np.testing.assert_array_equal(adapter.denormalize(result.actions), upstream)
    np.testing.assert_array_equal(adapter.postprocess_actions(result.actions),
                                  np.stack([native_eef10_to_libero7d(step) for step in upstream]))
    repeat = adapter.predict_action(InferenceRequest(prepared, **adapter.inference_defaults()))
    np.testing.assert_array_equal(repeat.actions, result.actions)


def test_tiny_mask_override_and_hook(tiny_libero):
    _, model = tiny_libero
    adapter = OpenWAMLiberoAdapter(model)
    prepared = adapter.prepare_observation(_libero_observation(np.random.default_rng(11)))
    defaults = adapter.inference_defaults()
    joint = adapter.predict_action(InferenceRequest(prepared, **defaults)).actions
    explicit = adapter.predict_action(InferenceRequest(prepared, **{**defaults, "condition": "mutual"})).actions
    np.testing.assert_array_equal(joint, explicit)  # tiny checkpoint was "trained" with the mutual mask
    isolated = adapter.predict_action(InferenceRequest(prepared, **{**defaults, "condition": "isolated"}))
    assert not np.array_equal(isolated.actions, joint)
    assert isolated.metadata["condition"] == "isolated" and isolated.metadata["training_attention_mask_mode"] == "mutual"

    layouts = []

    def identity(mask, layout):
        layouts.append(layout)
        return mask

    adapter.attention_mask_hook = identity
    np.testing.assert_array_equal(adapter.predict_action(InferenceRequest(prepared, **defaults)).actions, joint)
    assert len(layouts) == 10  # one mask per joint forward (10 denoising steps)
    assert layouts[0] == {"s_video": 3 * 12 * 10, "s_action": 32, "video_tokens_per_frame": 120, "mode": "mutual"}

    def cut_video_to_action(mask, layout):
        mask[:layout["s_video"], layout["s_video"]:] = False
        return mask

    adapter.attention_mask_hook = cut_video_to_action
    assert not np.array_equal(adapter.predict_action(InferenceRequest(prepared, **defaults)).actions, joint)
    driver = model.architecture.mot_driver
    assert driver.attention_mask_mode == "mutual" and "_build_attention_mask" not in vars(driver)
    adapter.attention_mask_hook = lambda mask, layout: mask[:-1]
    with pytest.raises(ValueError, match="attention_mask_hook"):
        adapter.predict_action(InferenceRequest(prepared, **defaults))
    assert "_build_attention_mask" not in vars(driver)


def test_tiny_tracing_sees_projections_not_blocks(tiny_libero):
    _, model = tiny_libero
    adapter = OpenWAMLiberoAdapter(model)
    prepared = adapter.prepare_observation(_libero_observation(np.random.default_rng(12)))
    defaults = adapter.inference_defaults()
    paths = ("video_backbone.dit.blocks.0.self_attn.q", "action_backbone.blocks.1.ffn")
    result = adapter.predict_action(InferenceRequest(prepared, **{**defaults, "trace_modules": paths}))
    assert {event.metadata["module"] for event in result.trace} == set(paths) and len(result.trace) == 20
    with pytest.raises(RuntimeError, match="never ran"):  # the MoT driver bypasses block.forward
        adapter.predict_action(InferenceRequest(prepared, **{**defaults, "trace_modules": ("video_backbone.dit.blocks.0",)}))


@pytest.mark.parametrize("variant,unify", [("joint_self_attn", True), ("joint_self_attn", False), ("idm", False),
                                           ("joint_cross_attn", False)])
def test_tiny_robotwin_variants_match_upstream_engine_call(tiny_vae, tmp_path, variant, unify):
    mask = "mutual" if unify else "action_sees_video"
    root = _tiny_checkpoint(tmp_path / "robotwin", dataset="robotwin", variant=variant, unify=unify, mask=mask)
    model = loader.load_openwam(root, device="cpu")
    adapter = OpenWAMRobotWinAdapter(model)
    prepared = adapter.prepare_observation(_robotwin_observation(np.random.default_rng(13)))
    assert prepared.proprio.shape == ((80,) if unify else (20,))
    result = adapter.predict_action(InferenceRequest(prepared, **adapter.inference_defaults()))
    upstream = _upstream_engine_call(model, _payload(adapter, prepared, prepared.metadata["raw_state_eef20"]))
    assert upstream.shape == (32, 20)
    np.testing.assert_array_equal(adapter.denormalize(result.actions), upstream)
    commands = adapter.postprocess_actions(result.actions, environment="robotwin")
    assert commands.shape == (32, 16) and np.isfinite(commands).all()
    if variant != "joint_self_attn":
        with pytest.raises(ValueError, match="joint self-attention"):
            adapter.predict_action(InferenceRequest(prepared, **{**adapter.inference_defaults(), "condition": "isolated"}))


def test_tiny_registry_round_trip(tiny_libero):
    from wam_interp.models.registry import load_model_adapter

    root, _ = tiny_libero
    adapter = load_model_adapter("openwam", checkpoint_dir=root, device="cpu")
    assert isinstance(adapter, OpenWAMAdapter) and adapter.inference_settings["action_horizon"] == 32
