"""CPU checks for dispatch, inference config, and LIBERO conventions."""

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
import json
import subprocess
import sys

import numpy as np
import pytest
import torch
from PIL import Image

from wam_interp.models import load_model_adapter, registered_model_adapters
from wam_interp.models.fasterwam.adapter import FasterWAMAdapter
from wam_interp.models.fasterwam.loader import compose_fasterwam_config, load_fasterwam
from wam_interp.models.fasterwam.libero_adapter import FasterWAMLiberoAdapter
from wam_interp.types import EnvironmentObservation, PreparedObservation, InferenceRequest


class TinyNative(torch.nn.Module):
    device = "cpu"
    torch_dtype = torch.float32

    def __init__(self):
        super().__init__()
        self.action_expert = SimpleNamespace(action_dim=7)
        self.projection = torch.nn.Identity()
        self.calls = []

    def infer_action_one_pass_future_cache(self, **kwargs):
        self.calls.append(("cache", kwargs))
        return {"action": self.projection(torch.ones(kwargs["action_horizon"], 7))}

    def infer_action(self, **kwargs):
        self.calls.append(("joint", kwargs))
        return {"action": self.projection(torch.ones(kwargs["action_horizon"], 7))}


def request():
    return InferenceRequest(PreparedObservation(torch.zeros(3, 16, 16), torch.zeros(8), "pick cup"),
                            condition="one_pass_future_cache", num_video_frames=9)


def test_config_is_inference_only_and_matches_release():
    from omegaconf import OmegaConf
    cfg = compose_fasterwam_config()
    cfg = OmegaConf.to_container(cfg, resolve=True)
    assert cfg["model"]["load_text_encoder"] is True
    assert cfg["model"]["skip_dit_load_from_pretrain"] is True
    assert cfg["model"]["action_dit_pretrained_path"] is None
    assert cfg["model"]["proprio_dim"] == 8
    assert cfg["model"]["action_dit_config"]["action_dim"] == 7
    assert cfg["model"]["condition_layers"] == list(range(0, 29, 4))
    assert cfg["model"]["video_kv_fusion"] == "interval_weighted_sum"
    assert cfg["model"]["action_scheduler"]["infer_shift"] == 5.0
    override = compose_fasterwam_config(overrides=["model.action_scheduler.infer_shift=3.0"])
    assert override.model.action_scheduler.infer_shift == 3.0


def test_registry_import_has_no_heavy_imports():
    code = "import sys; import wam_interp.models as m; assert 'fasterwam' in m.registered_model_adapters(); assert not {'torch','hydra','libero'} & set(sys.modules)"
    subprocess.run([sys.executable, "-c", code], check=True)
    assert "fasterwam" in registered_model_adapters()


def test_missing_checkpoint_fails_before_allocating(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_model_adapter("fasterwam", checkpoint=tmp_path / "missing.pt")


def test_native_loader_factory_and_provenance(tmp_path, monkeypatch):
    checkpoint = tmp_path / "release.pt"
    checkpoint.write_bytes(b"fixture")
    class Loadable(TinyNative):
        def load_checkpoint(self, path):
            self.loaded = path
    native = Loadable()
    import hydra.utils
    monkeypatch.setattr(hydra.utils, "instantiate", lambda *args, **kwargs: native)
    with pytest.raises(RuntimeError, match="did not load completely"):  # this fixture loads no weights
        load_fasterwam(compose_fasterwam_config(), checkpoint, device="cpu", dtype="float32")
    result = load_fasterwam(compose_fasterwam_config(), checkpoint, device="cpu", dtype="float32",
                            allow_partial_checkpoint=True)
    assert result._wam_interp_metadata["checkpoint_load"]["problems"]
    assert result.loaded == str(checkpoint.resolve())
    assert result.training is False
    assert result._wam_interp_metadata["resolved_config"]["model"]["condition_layers"] == list(range(0, 29, 4))
    json.dumps(result._wam_interp_metadata)


def test_hydra_factory_translates_native_constructor(monkeypatch):
    from hydra.utils import instantiate
    from wam_interp.models.fasterwam.model_code.fasterwam import FasterWAM
    captured = {}
    def capture(**kwargs):
        captured.update(kwargs)
        return TinyNative()
    monkeypatch.setattr(FasterWAM, "from_wan22_pretrained", capture)
    cfg = compose_fasterwam_config()
    instantiate(cfg.model, model_dtype=torch.float32, device="cpu")
    assert captured["condition_layers"] == list(range(0, 29, 4))
    assert captured["action_dit_config"]["condition_layers"] == captured["condition_layers"]
    assert captured["action_infer_shift"] == 5.0
    assert captured["video_train_shift"] == 5.0
    assert captured["skip_dit_load_from_pretrain"] is True
    assert captured["torch_dtype"] == torch.float32


@pytest.mark.parametrize("condition,expected", [("one_pass_future_cache", "cache"), ("joint_rollout", "joint")])
def test_dispatch_and_seed(condition, expected):
    native = TinyNative()
    result = FasterWAMAdapter(native).predict_action(replace(request(), condition=condition, seed=21, sigma_shift=3.0))
    kind, kwargs = native.calls[0]
    assert kind == expected
    assert kwargs["seed"] == 21 and kwargs["sigma_shift"] == 3.0
    assert kwargs["num_video_frames"] == 9
    assert kwargs["prompt"].endswith("instruction: pick cup")
    assert result.actions.shape == (32, 7)
    assert result.metadata["action_dtype"] == "torch.float32"


@pytest.mark.parametrize("changes", [{"condition": "first_frame"}, {"num_video_frames": 8}, {"num_video_frames": 0}, {"compile_action_infer": True}, {"action_horizon": 0}])
def test_unsupported_requests(changes):
    with pytest.raises(ValueError):
        FasterWAMAdapter(TinyNative()).predict_action(replace(request(), **changes))


def test_trace_and_intervention_cleanup():
    from wam_interp.interventions.module_output import replace_module_output
    native = TinyNative()
    adapter = FasterWAMAdapter(native)
    with replace_module_output(native, "projection", lambda output: output * 0):
        result = adapter.predict_action(replace(request(), trace_modules=("projection",)))
    assert torch.count_nonzero(result.actions) == 0
    assert len(result.trace) == 1
    assert not native.projection._forward_hooks
    assert adapter.predict_action(request()).actions.sum() > 0
    def fail(_output):
        raise RuntimeError("test cleanup")
    with pytest.raises(RuntimeError):
        with replace_module_output(native, "projection", fail):
            adapter.predict_action(replace(request(), trace_modules=("projection",)))
    assert not native.projection._forward_hooks


def test_libero_matches_upstream_formula_and_camera_order():
    from omegaconf import OmegaConf
    cfg = compose_fasterwam_config()
    # Asymmetric ranges make accidental identity normalization visible.
    stats = {kind: {"default": {"global_min": [-2.0] * n, "global_max": [4.0] * n}}
             for kind, n in (("state", 8), ("action", 7))}
    adapter = FasterWAMLiberoAdapter(TinyNative(), data_config=OmegaConf.to_container(cfg.data.train, resolve=True), dataset_stats=stats)
    camera = np.arange(256 * 256 * 3, dtype=np.uint8).reshape(256, 256, 3)
    raw = EnvironmentObservation({"agentview": camera, "wrist": 255 - camera},
        {"eef_position": [1, 0, -1], "eef_quaternion": [0, 0, 0, 1], "gripper_position": [0.2, 0.4]}, "pick cup")
    prepared = adapter.prepare_observation(raw)
    # Independent upstream center crop/resize formula at the default square resolution.
    first = np.asarray(Image.fromarray(camera[::-1, ::-1]).resize((224, 224), Image.Resampling.BILINEAR))
    second = np.asarray(Image.fromarray((255-camera)[::-1, ::-1]).resize((224, 224), Image.Resampling.BILINEAR))
    expected = torch.from_numpy(np.concatenate([first, second], axis=1).copy()).permute(2, 0, 1).float() * (2.0 / 255.0) - 1.0
    torch.testing.assert_close(prepared.image[0], expected)
    proprio = np.array([1, 0, -1, 0, 0, 0, 0.2, 0.4], dtype=np.float32)
    np.testing.assert_allclose(prepared.proprio.numpy()[0], (proprio + 2) / 3 - 1, atol=1e-7)
    actions = torch.zeros(32, 7)
    actions[:, -1] = -0.5
    commands = adapter.postprocess_actions(actions, environment="libero")
    expected_actions = actions.numpy() * 3 + 1
    expected_actions[:, -1] = np.sign(-(expected_actions[:, -1] * 2 - 1))
    np.testing.assert_allclose(commands, expected_actions)
    assert adapter.inference_defaults() == {"num_video_frames": 9, "condition": "one_pass_future_cache"}


def test_native_sparse_config_validation():
    from wam_interp.models.fasterwam.model_code.sparse_action_dit import SparseActionDiT
    from wam_interp.models.fasterwam.model_code.sparse_mot import SparseMoT
    assert SparseActionDiT._normalize_condition_layers([0, 2], 3) == (0, 2)
    with pytest.raises(ValueError):
        SparseActionDiT._normalize_condition_layers([3], 3)
    assert callable(SparseMoT.prefill_video_cache)


def test_shared_evaluator_uses_fasterwam_defaults_and_saves_result(monkeypatch, tmp_path):
    from omegaconf import OmegaConf
    from wam_interp.evaluations.LIBERO import runner, LiberoEvalConfig
    cfg = compose_fasterwam_config()
    native = TinyNative()
    stats = {kind: {"default": {"global_min": [-1.0] * n, "global_max": [1.0] * n}}
             for kind, n in (("state", 8), ("action", 7))}
    adapter = FasterWAMLiberoAdapter(native, data_config=OmegaConf.to_container(cfg.data.train, resolve=True), dataset_stats=stats)
    observation = EnvironmentObservation({"agentview": np.zeros((256, 256, 3), np.uint8),
        "wrist": np.zeros((256, 256, 3), np.uint8)}, {"eef_position": [0,0,0],
        "eef_quaternion": [0,0,0,1], "gripper_position": [0,0]}, "pick cup")
    task = SimpleNamespace(language="pick cup", name="fixture")
    env = SimpleNamespace(close=lambda: None)
    monkeypatch.setattr(runner, "_get_suite", lambda name: SimpleNamespace(n_tasks=1, get_task=lambda i: task))
    monkeypatch.setattr(runner, "initial_states_for_task", lambda *args: [0])
    monkeypatch.setattr(runner, "create_libero_environment", lambda *args: env)
    monkeypatch.setattr(runner, "reset_to_initial_state", lambda *args, **kwargs: observation)
    commands = []
    def step(env, action, **kwargs):
        commands.append(action)
        return SimpleNamespace(observation=observation, reward=1.0, success=True, done=True)
    monkeypatch.setattr(runner, "step_libero", step)
    result = runner.evaluate_libero(adapter, LiberoEvalConfig(num_trials=1, num_steps_wait=0, output_dir=tmp_path))
    assert native.calls[0][0] == "cache"
    assert native.calls[0][1]["num_video_frames"] == 9
    assert np.asarray(commands).shape == (1, 7)
    assert result.model_name == "fasterwam" and result.episodes == 1
    saved = json.loads(Path(result.output_path).read_text())
    assert saved["model_name"] == "fasterwam"


@pytest.mark.parametrize("condition", ["one_pass_future_cache", "joint_rollout"])
def test_tiny_native_inference_and_checkpoint_roundtrip(condition, tmp_path):
    from wam_interp.models.fasterwam.model_code.fasterwam import FasterWAM
    from wam_interp.models.fasterwam.model_code.sparse_action_dit import SparseActionDiT
    from wam_interp.models.fasterwam.model_code.sparse_mot import SparseMoT
    from wam_interp.models.fasterwam.model_code.wan_video_dit import WanVideoDiT

    class TinyVAE(torch.nn.Module):
        temporal_downsample_factor = 4
        upsampling_factor = 8
        model = SimpleNamespace(z_dim=4)

        def encode(self, images, **kwargs):
            return [torch.zeros(4, 1, 2, 2)]

    def build():
        common = dict(hidden_dim=48, ffn_dim=64, text_dim=16, freq_dim=16,
                      eps=1e-6, num_heads=2, attn_head_dim=24, num_layers=3)
        video = WanVideoDiT(**common, in_dim=4, out_dim=4, patch_size=(1, 2, 2),
                           has_image_input=False, seperated_timestep=True,
                           video_attention_mask_mode="first_frame_causal")
        action = SparseActionDiT(**common, action_dim=7, noncondition_num_heads=1, condition_layers=[0, 2])
        mot = SparseMoT({"video": video, "action": action}, condition_layers=[0, 2],
                        video_kv_fusion="interval_weighted_sum", mot_checkpoint_mixed_attn=False)
        native = FasterWAM(video, action, mot, TinyVAE(), text_dim=16, proprio_dim=8,
                          mot_action_video_attention_layers=[0, 2])
        native.encode_prompt = lambda prompt: (torch.ones(1, 2, 16), torch.ones(1, 2, dtype=torch.bool))
        return native

    torch.manual_seed(12)
    native = build()
    req = replace(request(), condition=condition, action_horizon=4, num_inference_steps=2)
    first = FasterWAMAdapter(native).predict_action(req).actions
    torch.testing.assert_close(first, FasterWAMAdapter(native).predict_action(req).actions, rtol=0, atol=0)
    checkpoint = tmp_path / "tiny.pt"
    native.save_checkpoint(checkpoint)
    restored = build()
    restored.load_checkpoint(checkpoint)
    second = FasterWAMAdapter(restored).predict_action(req).actions
    torch.testing.assert_close(first, second, rtol=0, atol=0)
    assert first.shape == (4, 7) and torch.isfinite(first).all()
