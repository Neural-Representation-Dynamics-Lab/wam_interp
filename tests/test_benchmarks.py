"""Protocol and transform tests; no simulator assets or checkpoints required."""

from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
import json
import subprocess
import sys

import numpy as np
import pytest
import torch

from wam_interp.types import EnvironmentObservation, InferenceRequest, InferenceResult, PreparedObservation
from wam_interp.evaluations.LIBERO_PLUS import LiberoPlusEvalConfig
from wam_interp.evaluations.LIBERO_PLUS import runner as plus_runner
from wam_interp.evaluations.LIBERO_PLUS.environment import initial_state_path
from wam_interp.evaluations.ROBOTWIN import RoboTwinEvalConfig
from wam_interp.evaluations.ROBOTWIN import runner as robot_runner


def raw_robot():
    return EnvironmentObservation(
        {"head": np.full((48, 64, 3), 255, np.uint8),
         "left_wrist": np.full((48, 64, 3), 127, np.uint8),
         "right_wrist": np.zeros((48, 64, 3), np.uint8)},
        {"joint_position": np.arange(14, dtype=np.float32),
         "left_endpose": [1, 2, 3, 0, 0, 0, 1], "left_gripper": 0.5,
         "right_endpose": [4, 5, 6, 0, 0, 0, 1], "right_gripper": 0.2}, "pick object")


class Native(torch.nn.Module):
    device = "cpu"
    torch_dtype = torch.float32

    def __init__(self, dimensions=14):
        super().__init__()
        self.action_expert = SimpleNamespace(action_dim=dimensions)
        self.last = None

    def infer_action(self, prompt, input_image, action_horizon, **kwargs):
        self.last = kwargs
        return {"action": torch.zeros(action_horizon, self.action_expert.action_dim)}

    def infer_action_one_pass_future_cache(self, **kwargs):
        return self.infer_action(**kwargs)


def stats(dim=14, mode="z-score"):
    return {kind: {"default": {"global_min": [-10] * dim, "global_max": [10] * dim,
              "global_mean": [1] * dim, "global_std": [2] * dim}} for kind in ("state", "action")}


def test_importing_evaluators_keeps_simulators_and_torch_lazy():
    code = "import sys; from wam_interp.evaluations.LIBERO_PLUS import evaluate_libero_plus; from wam_interp.evaluations.ROBOTWIN import evaluate_robotwin; assert not {'libero','sapien','torch','envs'} & set(sys.modules)"
    subprocess.run([sys.executable, "-c", code], check=True)


@pytest.mark.parametrize("backend", ["fastwam", "fasterwam"])
def test_robotwin_joint_config_transform_and_inference(backend):
    from omegaconf import OmegaConf
    if backend == "fastwam":
        from wam_interp.models.fastwam.loader import compose_fastwam_config
        from wam_interp.models.fastwam.robotwin_adapter import FastWAMRobotWinAdapter as Adapter
        cfg = compose_fastwam_config("robotwin_uncond_3cam_384_1e-4", benchmark="robotwin")
    else:
        from wam_interp.models.fasterwam.loader import compose_fasterwam_config
        from wam_interp.models.fasterwam.robotwin_adapter import FasterWAMRobotWinAdapter as Adapter
        cfg = compose_fasterwam_config("robotwin_fasterwam_3cam_384_1e-4", benchmark="robotwin")
    native = Native()
    assert cfg.model.proprio_dim == 14
    adapter = Adapter(native, data_config=OmegaConf.to_container(cfg.data.train, resolve=True), dataset_stats=stats())
    prepared = adapter.prepare_observation(raw_robot())
    assert prepared.image.shape == (1, 3, 384, 320)
    assert prepared.image[0, :, :256].min() == 1  # Head above both wrists; no flip.
    assert prepared.image[0, :, 256:, 160:].max() == -1
    np.testing.assert_allclose(prepared.proprio[0].numpy(), np.clip((np.arange(14) - 1) / 2, -5, 5), atol=1e-6)
    decoded = adapter.postprocess_actions(torch.zeros(32, 14), environment="robotwin")
    np.testing.assert_allclose(decoded, 1, atol=1e-6)  # Native qpos grippers are not sign-flipped.
    assert adapter.environment_action_type("robotwin") == "qpos"
    result = adapter.predict_action(InferenceRequest(prepared, **adapter.inference_defaults()))
    assert result.actions.shape == (32, 14)


@pytest.mark.parametrize("name,expected", [
    ("task_table_12.pruned_init", "libero_10/task.pruned_init"),
    ("task_tb_12.pruned_init", "libero_10/task.pruned_init"),
    ("task_language_3.pruned_init", "libero_10/task.pruned_init"),
    ("task_view_2.pruned_init", "libero_10/task.pruned_init"),
    ("task_light_2.pruned_init", "libero_10/task.pruned_init"),
    ("task_level3.pruned_init", "libero_newobj/libero_10/task_level3.pruned_init"),
    ("task_add_2.pruned_init", "libero_newobj/libero_10/task_add_2.pruned_init"),
])
def test_plus_initial_state_remapping(name, expected):
    task = SimpleNamespace(problem_folder="libero_10", init_states_file=name)
    assert initial_state_path(task, Path("init_states")) == Path("init_states") / expected


def fake_plus_suite():
    tasks = [SimpleNamespace(name="variant_b", language="upstream rewritten instruction"),
             SimpleNamespace(name="variant_a", language="pick cup")]
    return SimpleNamespace(n_tasks=2, get_task=lambda index: tasks[index])


def classifications():
    # Deliberately reorder labels; metadata id is not a zero-based task index.
    return [{"id": 8, "name": "variant_a", "category": "Camera Viewpoints", "difficulty_level": 2},
            {"id": 1, "name": "variant_b", "category": "Language Instructions", "difficulty_level": 3}]


def test_plus_task_filters_join_by_variant_name():
    config = LiberoPlusEvalConfig(task_ids=None, categories=("Camera Viewpoints",), difficulty_levels=(2,))
    selected = plus_runner.select_variants(fake_plus_suite(), classifications(), config)
    assert [item["task_id"] for item in selected] == [1]
    assert selected[0]["classification_id"] == 8
    with pytest.raises(ValueError, match="Unknown perturbation"):
        plus_runner.select_variants(fake_plus_suite(), classifications(), replace(config, categories=("typo",)))
    with pytest.raises(ValueError, match="No variants"):
        plus_runner.select_variants(fake_plus_suite(), classifications(), replace(config, difficulty_levels=(5,)))


def test_plus_rejects_wrong_import_and_suite(tmp_path, monkeypatch):
    from wam_interp.evaluations.LIBERO_PLUS import environment as env
    root = tmp_path / "plus"
    folder = root / "libero/libero/benchmark"
    folder.mkdir(parents=True)
    benchmark = SimpleNamespace(__file__=str(folder / "__init__.py"),
                               get_benchmark_dict=lambda: {"libero_spatial": lambda: SimpleNamespace(n_tasks=10)})
    monkeypatch.setattr(env, "get_benchmark_module", lambda: (benchmark, None, None))
    monkeypatch.setattr(env, "checkout_revision", lambda *args: env.LIBERO_PLUS_REVISION)
    config = LiberoPlusEvalConfig(benchmark_root=root)
    with pytest.raises(RuntimeError, match="classification"):
        env.identify_benchmark(config)
    (folder / "task_classification.json").write_text(json.dumps({"libero_spatial": []}))
    with pytest.raises(RuntimeError, match="suite size"):
        env.identify_benchmark(config)
    with pytest.raises(RuntimeError, match="outside"):
        env.identify_benchmark(replace(config, benchmark_root=tmp_path / "other"))


class LoopAdapter:
    name = "fixture"

    def __init__(self, dim=7):
        self.dim = dim
        self.events = []
        self.model = SimpleNamespace(device="cpu", dtype=torch.float32)

    def inference_defaults(self):
        return {"condition": "fixture"}

    def prepare_observation(self, observation):
        return PreparedObservation(None, None, observation.instruction)

    def predict_action(self, request):
        self.events.append("predict")
        return InferenceResult(np.zeros((request.action_horizon, self.dim)), {"dtype": "float32"})

    def postprocess_actions(self, actions, *, environment):
        return actions

    def begin_episode(self, observation):
        self.events.append("begin")

    def observe_step(self, observation, action):
        self.events.append("observe")

    def environment_action_type(self, environment):
        return "qpos" if self.dim == 14 else "ee"


def test_plus_shared_loop_hooks_aggregation_and_save(tmp_path, monkeypatch):
    classification = tmp_path / "classification.json"
    classification.write_text("{}")
    monkeypatch.setattr(plus_runner, "identify_benchmark", lambda config: (fake_plus_suite(), classifications(), tmp_path, classification))
    monkeypatch.setattr(plus_runner, "initial_states_for_variant", lambda *args: [0])
    closes = []
    monkeypatch.setattr(plus_runner, "create_libero_environment", lambda *args: SimpleNamespace(close=lambda: closes.append(1)))
    obs = EnvironmentObservation({}, {}, "upstream rewritten instruction")
    monkeypatch.setattr(plus_runner, "reset_to_initial_state", lambda *args, **kwargs: obs)
    monkeypatch.setattr(plus_runner, "step_libero", lambda *args, **kwargs: SimpleNamespace(observation=obs, reward=1, done=True, success=True))
    adapter = LoopAdapter()
    result = plus_runner.evaluate_libero_plus(adapter, LiberoPlusEvalConfig(num_steps_wait=0, output_dir=tmp_path))
    assert adapter.events == ["begin", "predict", "observe"]
    assert result["partial_selection"] and result["successes"] == result["episodes"] == 1
    assert result["by_category"]["Language Instructions"]["success_rate"] == 1.0
    assert len(closes) == 1 and Path(result["output_path"]).is_file()


def test_plus_cleanup_and_partial_report_on_failure(tmp_path, monkeypatch):
    classification = tmp_path / "classification.json"
    classification.write_text("{}")
    monkeypatch.setattr(plus_runner, "identify_benchmark", lambda config: (fake_plus_suite(), classifications(), tmp_path, classification))
    monkeypatch.setattr(plus_runner, "initial_states_for_variant", lambda *args: [0])
    closes = []
    monkeypatch.setattr(plus_runner, "create_libero_environment", lambda *args: SimpleNamespace(close=lambda: closes.append(1)))
    def fail(*args, **kwargs):
        raise RuntimeError("fixture reset failure")
    monkeypatch.setattr(plus_runner, "reset_to_initial_state", fail)
    with pytest.raises(RuntimeError, match="fixture reset"):
        plus_runner.evaluate_libero_plus(LoopAdapter(), LiberoPlusEvalConfig(num_steps_wait=0, output_dir=tmp_path))
    assert closes == [1]
    report = json.loads(next(tmp_path.glob("libero_plus_*.json")).read_text())
    assert report["error"]["type"] == "RuntimeError" and report["episodes"] == 0


class Frontend:
    def __init__(self, rejected=0):
        self.rejected = rejected
        self.attempts = 0
        self.action_count = 0
        self.step_limit = 4
        self.commands = []
        self.closed = 0

    def screen_seed(self, seed, episode):
        self.attempts += 1
        return self.attempts > self.rejected, {}, "expert_failed"

    def begin(self, *args):
        return raw_robot()

    def step(self, action, action_type):
        self.commands.append((action, action_type))
        self.action_count += 1
        return raw_robot(), self.action_count == 2

    def close(self, clear_cache=False):
        self.closed += 1


def patch_frontend(monkeypatch, frontend):
    @contextmanager
    def open_env(config):
        try:
            yield frontend
        finally:
            frontend.close()
    monkeypatch.setattr(robot_runner, "open_environment", open_env)


@pytest.mark.parametrize("dim,action_type", [(14, "qpos"), (16, "ee")])
def test_robotwin_runner_seed_screening_hooks_and_commands(tmp_path, monkeypatch, dim, action_type):
    frontend = Frontend(rejected=1)
    patch_frontend(monkeypatch, frontend)
    adapter = LoopAdapter(dim)
    result = robot_runner.evaluate_robotwin(adapter, RoboTwinEvalConfig("fixture", tmp_path, output_dir=tmp_path))
    assert result["accepted_seeds"] == [100001]
    assert result["rejected_seeds"][0]["seed"] == 100000
    assert result["success_rate"] == 1.0 and result["completed"]
    assert adapter.events == ["begin", "predict", "observe", "observe"]
    assert len(frontend.commands) == 2 and frontend.commands[0][1] == action_type
    assert Path(result["output_path"]).is_file()


def test_robotwin_seed_limit_and_failure_report(tmp_path, monkeypatch):
    frontend = Frontend(rejected=20)
    patch_frontend(monkeypatch, frontend)
    with pytest.raises(RuntimeError, match="Seed search exhausted"):
        robot_runner.evaluate_robotwin(LoopAdapter(14), RoboTwinEvalConfig("fixture", tmp_path,
            max_seed_attempts=2, output_dir=tmp_path))
    report = json.loads(next(tmp_path.glob("robotwin_*.json")).read_text())
    assert report["attempts"] == 2 and len(report["rejected_seeds"]) == 2
    assert report["success_rate"] is None and frontend.closed


@pytest.mark.parametrize("invalid", ["nonfinite", "dimension"])
def test_robotwin_invalid_commands_cleanup(tmp_path, monkeypatch, invalid):
    frontend = Frontend()
    patch_frontend(monkeypatch, frontend)
    adapter = LoopAdapter(14)
    adapter.postprocess_actions = lambda *args, **kwargs: (
        np.full((3, 14), np.nan) if invalid == "nonfinite" else np.zeros((3, 7)))
    with pytest.raises(ValueError, match="nonfinite" if invalid == "nonfinite" else "Expected qpos"):
        robot_runner.evaluate_robotwin(adapter, RoboTwinEvalConfig("fixture", tmp_path, output_dir=tmp_path))
    assert frontend.closed and not frontend.commands


def test_robotwin_context_restores_process_after_failure(tmp_path, monkeypatch):
    from wam_interp.evaluations.ROBOTWIN import environment as env
    for path in ("envs/_base_task.py", "description/utils/generate_episode_instructions.py"):
        item = tmp_path / path
        item.parent.mkdir(parents=True, exist_ok=True)
        item.touch()
    (tmp_path / "task_config").mkdir()
    (tmp_path / "assets").mkdir()
    monkeypatch.setattr(env, "checkout_revision", lambda *args: env.ROBOTWIN_REVISION)
    monkeypatch.delitem(sys.modules, "envs", raising=False)
    frontend = Frontend()
    monkeypatch.setattr(env, "RobotWinFrontend", lambda config: frontend)
    previous = Path.cwd()
    with pytest.raises(RuntimeError):
        with env.open_environment(RoboTwinEvalConfig("fixture", tmp_path)):
            assert Path.cwd() == tmp_path
            raise RuntimeError("fixture")
    assert Path.cwd() == previous and frontend.closed


class LingNative:
    def __init__(self, robot=True):
        self.config = SimpleNamespace(env_type="robotwin_tshape" if robot else "none",
            action_per_frame=16 if robot else 4, frame_chunk_size=2 if robot else 4,
            num_inference_steps=25 if robot else 20, action_num_inference_steps=50,
            video_exec_step=-1, guidance_scale=5, height=256 if robot else 128,
            width=320 if robot else 128)
        self.device = "cpu"
        self.prompt_embeds = None
        self.transformer = torch.nn.Identity()
        self.frame_st_id = 0
        self.updates = []
        self.robot = robot

    def reset(self, prompt):
        self.prompt_embeds = torch.ones(1)
        self.frame_st_id = 0
        self.updates = []

    def infer_chunk(self, *args, **kwargs):
        return torch.zeros(1), None

    def postprocess_action(self, actions):
        cfg = self.config
        chunk = np.zeros((16 if self.robot else 7, cfg.frame_chunk_size, cfg.action_per_frame), np.float32)
        if self.robot:
            chunk[[6, 14]] = 1  # Identity XYZW quaternion for each arm.
        return chunk

    def compute_kv_cache(self, frames, actions):
        self.updates.append((frames, actions))
        self.frame_st_id += self.config.frame_chunk_size

    def checkpoint_state(self):
        return {"frame_st_id": self.frame_st_id}

    def restore_state(self, state):
        assert state["frame_st_id"] == self.frame_st_id

    def saving_overwritten_rows(self, state):
        from contextlib import nullcontext
        return nullcontext()


def test_lingbot_robotwin_chunk_cadence_relative_pose_and_reset():
    from wam_interp.models.lingbot_va.robotwin_adapter import LingBotVARobotWinAdapter
    native = LingNative()
    adapter = LingBotVARobotWinAdapter(native)
    obs = raw_robot()
    adapter.begin_episode(obs)
    prepared = adapter.prepare_observation(obs)
    assert [tuple(view.shape) for view in prepared.image] == [(3,256,320), (3,128,160), (3,128,160)]
    request = InferenceRequest(prepared, action_horizon=32, **adapter.inference_defaults())
    prediction = adapter.predict_action(request)
    commands = adapter.postprocess_actions(prediction.actions, environment="robotwin")
    assert commands.shape == (16, 16)  # First latent action frame is skipped.
    np.testing.assert_allclose(commands[0, :3], [1,2,3])
    np.testing.assert_allclose(commands[0, 8:11], [4,5,6])
    adapter.predict_action(request)  # nothing executed yet: a repeat prediction, no cache update
    assert not native.updates
    for command in commands[:3]:
        adapter.observe_step(obs, command)
    with pytest.raises(ValueError, match="complete chunk"):
        adapter.predict_action(request)
    for command in commands[3:]:
        adapter.observe_step(obs, command)
    adapter.predict_action(request)
    assert len(native.updates[0][0]) == 4  # One observation every four executed actions.
    commands = adapter.postprocess_actions(prediction.actions, environment="robotwin")
    assert commands.shape == (32, 16)
    for command in commands:
        adapter.observe_step(obs, command)
    adapter.predict_action(request)
    assert len(native.updates[1][0]) == 8
    adapter.begin_episode(obs)
    assert adapter._last_chunk is None and not adapter._frames


def test_relative_pose_matches_scipy_reference():
    from scipy.spatial.transform import Rotation
    from wam_interp.models.lingbot_va.robotwin_adapter import compose_relative_pose
    relative = np.array([0.1,0.2,0.3,0.2,0.1,0.3,0.9,0.5] * 2)
    initial = np.array([1,2,3,0.4,0.3,0.2,0.8,0.9] * 2)
    result = compose_relative_pose(relative, initial)
    reference = (Rotation.from_quat(initial[3:7]) * Rotation.from_quat(relative[3:7])).as_quat()
    np.testing.assert_allclose(result[3:7], reference, atol=1e-7)
    np.testing.assert_allclose(result[:3], relative[:3] + initial[:3])
    assert result[7] == relative[7]
    with pytest.raises(ValueError, match="zero-norm"):
        compose_relative_pose(np.zeros(16), initial)


def test_lingbot_robotwin_streaming_latent_layout():
    from wam_interp.models.lingbot_va.runtime import LingBotVARuntime
    native = LingBotVARuntime.__new__(LingBotVARuntime)
    native.config = SimpleNamespace(env_type="robotwin_tshape")
    native.device = "cpu"
    native.dtype = torch.float32
    native.vae = SimpleNamespace(config=SimpleNamespace(latents_mean=[0.0] * 48, latents_std=[1.0] * 48))
    class Encoder:
        def __init__(self, head=False):
            self.head = head
            self.inputs = []
        def encode_chunk(self, value):
            self.inputs.append(value.shape)
            b, c, t, h, w = value.shape
            output = torch.ones(b, 96, 1, h//16, w//16)
            if self.head:
                output *= 3
            else:
                output[1] *= 2
            return output
    native.streaming_vae = Encoder(head=True)
    native.streaming_vae_half = Encoder()
    frame = (torch.ones(3,256,320), torch.ones(3,128,160), torch.ones(3,128,160))
    latent = native.encode_frames([frame])
    assert latent.shape == (1,48,1,24,20)
    assert latent[..., :8, :10].mean() == 1
    assert latent[..., :8, 10:].mean() == 2
    assert latent[..., 8:, :].mean() == 3
    assert native.streaming_vae.inputs != native.streaming_vae_half.inputs


@pytest.mark.parametrize("backend", ["fastwam", "fasterwam", "lingbot_va"])
def test_explicit_plus_adapter_action_semantics(backend):
    from omegaconf import OmegaConf
    from wam_interp.models.fasterwam.loader import compose_fasterwam_config
    if backend == "lingbot_va":
        from wam_interp.models.lingbot_va.libero_plus_adapter import LingBotVALiberoPlusAdapter
        adapter = LingBotVALiberoPlusAdapter(LingNative(robot=False))
        adapter.begin_episode(EnvironmentObservation({}, {}, "pick cup"))
        commands = adapter.postprocess_actions(torch.zeros(1), environment="libero_plus")
        assert commands.shape == (12,7)
    else:
        if backend == "fastwam":
            from wam_interp.models.fastwam.libero_plus_adapter import FastWAMLiberoPlusAdapter as Adapter
        else:
            from wam_interp.models.fasterwam.libero_plus_adapter import FasterWAMLiberoPlusAdapter as Adapter
        cfg = compose_fasterwam_config()
        data = OmegaConf.to_container(cfg.data.train, resolve=True)
        adapter = Adapter(Native(7), data_config=data, dataset_stats=stats(7))
        commands = adapter.postprocess_actions(torch.zeros(32,7), environment="libero_plus")
        assert commands.shape == (32,7)
    with pytest.raises(ValueError):
        adapter.postprocess_actions(torch.zeros(1), environment="robotwin")


def load_setup_script():
    import importlib.util
    path = Path(__file__).resolve().parents[1] / "scripts/prepare_robotwin.py"
    spec = importlib.util.spec_from_file_location("prepare_robotwin", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_robotwin_setup_preserves_native_patches_and_runtime_pins():
    setup = load_setup_script()
    requirements = setup.filter_requirements(
        "torch==2.4.1\ntorchvision\nhuggingface_hub==0.25.0\nazure==4.0.0\n"
        "azure-ai-inference\nsapien==3.0.0b1\nscipy==1.10.1\nzarr\npyglet<2\n")
    assert "torch==2.4.1" not in requirements and "torchvision" not in requirements
    assert "huggingface-hub==0.36.2" in requirements and "azure==" not in requirements
    assert "azure-ai-inference" in requirements and "sapien==3.0.0b1" in requirements
    assert "scipy==1.10.1" in requirements and "zarr==2.18.3" in requirements
    source = ('pip install -r script/requirements.txt\n# SAPIEN/MPLib patches\n'
        'cd envs\ngit clone --branch v0.7.8 --depth 1 https://github.com/NVlabs/curobo.git\n'
        'cd curobo\npip install -e . --no-build-isolation\n')
    installer = setup.installer_body(source)
    assert installer.startswith("set -euo pipefail")
    assert "# SAPIEN/MPLib patches" in installer and "if [ ! -d curobo/.git ]" in installer
    assert "pip install -e . --no-build-isolation" in installer
    with pytest.raises(ValueError, match="installer changed"):
        setup.installer_body("unexpected source")


def test_robotwin_full_chunk_requirement_fails_before_simulator(tmp_path):
    adapter = LoopAdapter(16)
    adapter.requires_full_action_chunk = True
    with pytest.raises(ValueError, match="full action chunks"):
        robot_runner.evaluate_robotwin(adapter, RoboTwinEvalConfig("fixture", tmp_path))


@pytest.mark.parametrize("expert_ok", [True, False])
def test_robotwin_frontend_native_expert_protocol_and_cleanup(tmp_path, expert_ok):
    from wam_interp.evaluations.ROBOTWIN.environment import RobotWinFrontend
    events = []
    class NativeTask:
        plan_success = expert_ok
        def setup_demo(self, **kwargs):
            events.append(("setup", kwargs))
        def play_once(self):
            events.append("expert")
            return {"info": {"object": "fixture"}}
        def check_success(self):
            events.append("success")
            return True
        def close_env(self, clear_cache=False):
            events.append("close")
    frontend = RobotWinFrontend.__new__(RobotWinFrontend)
    frontend.config = RoboTwinEvalConfig("fixture", tmp_path)
    frontend.env = NativeTask()
    frontend.active = False
    frontend.args = {"render_freq": 0}
    accepted, info, reason = frontend.screen_seed(100001, 0)
    assert accepted == expert_ok
    assert events[0] == ("setup", dict(render_freq=0, now_ep_num=0, seed=100001, is_test=True))
    assert events[-1] == "close" and not frontend.active
    assert info["info"]["object"] == "fixture"
    assert reason == (None if expert_ok else "expert_failed")


def test_model_provenance_hashes_normalization_statistics():
    from wam_interp.evaluations.provenance import model_provenance
    adapter = LoopAdapter(14)
    adapter.dataset_stats = stats()
    first = model_provenance(adapter)["statistics_content_sha256"]
    adapter.dataset_stats["state"]["default"]["global_mean"][0] = 2
    second = model_provenance(adapter)["statistics_content_sha256"]
    assert first != second and len(first) == 64
