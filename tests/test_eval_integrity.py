"""Regression tests for evaluation-integrity fixes (sampler shift, LIBERO-Plus names, RoboTwin
protocol, partial results, checkpoint completeness, silent hooks, provenance)."""

import importlib.util
import json
import os
import signal
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------- FastWAM released-checkpoint sampler shift


@pytest.mark.parametrize("task,expected", [
    ("libero_uncond_2cam224_1e-4", 5.0),
    ("libero_optional_idm_2cam224_1e-4", 1.0),
    ("libero_idm_2cam224_1e-4", None),
])
def test_released_checkpoint_sampler_shift_is_applied(task, expected):
    from wam_interp.models.fastwam.loader import compose_fastwam_config

    cfg = compose_fastwam_config(task)
    assert cfg.EVALUATION.sigma_shift == expected
    assert compose_fastwam_config(task, released_checkpoint_defaults=False).EVALUATION.sigma_shift is None
    assert compose_fastwam_config(task, overrides=["EVALUATION.sigma_shift=2.0"]).EVALUATION.sigma_shift == 2.0


def test_robotwin_uncond_shift_and_adapter_defaults():
    from wam_interp.models.fastwam.libero_adapter import evaluation_sampler_defaults
    from wam_interp.models.fastwam.loader import compose_fastwam_config
    from wam_interp.models.fastwam.robotwin_adapter import FastWAMRobotWinAdapter

    assert compose_fastwam_config("robotwin_uncond_3cam_384_1e-4", benchmark="robotwin").EVALUATION.sigma_shift == 5.0
    model = SimpleNamespace(_wam_interp_metadata={"evaluation": {"sigma_shift": 5.0}})
    assert evaluation_sampler_defaults(model) == {"sigma_shift": 5.0}
    assert evaluation_sampler_defaults(SimpleNamespace()) == {}
    adapter = object.__new__(FastWAMRobotWinAdapter)
    adapter.model = model
    assert adapter.inference_defaults()["sigma_shift"] == 5.0


# ---------------------------------------------------------------- LIBERO-Plus variant names


def test_libero_plus_variant_names_resolve_to_their_base_bddl(monkeypatch, tmp_path):
    from wam_interp.evaluations.LIBERO import environment

    variant = "pick_up_the_bowl_view_0_0_100_0_0_initstate_3_noise_2.bddl"
    assert environment.bddl_source_name(variant) == "pick_up_the_bowl.bddl"
    assert environment.bddl_source_name("pick_up_the_bowl_table_1.bddl") == "pick_up_the_bowl_table_1.bddl"
    (tmp_path / "suite").mkdir()
    (tmp_path / "suite" / "pick_up_the_bowl.bddl").write_text("(define)")
    opened = []

    class FakeEnv:
        def __init__(self, bddl_file_name, **kwargs):
            opened.append(bddl_file_name)

        def seed(self, value):
            pass

    monkeypatch.setattr(environment, "get_benchmark_module",
                        lambda: (None, lambda key: str(tmp_path), FakeEnv))
    environment.create_libero_environment(SimpleNamespace(problem_folder="suite", bddl_file=variant), 128, 0)
    assert opened == [str(tmp_path / "suite" / variant)]   # the simulator still gets the full variant name
    with pytest.raises(FileNotFoundError):
        environment.create_libero_environment(
            SimpleNamespace(problem_folder="suite", bddl_file="missing_view_0_0_100_0_0_initstate_1.bddl"), 128, 0)


# ---------------------------------------------------------------- RoboTwin per-backend protocol


def _script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_robotwin_protocol_matches_each_backends_published_eval():
    protocol = _script("evaluate_benchmark").robotwin_protocol
    assert protocol("fastwam", 0) == (24, None)
    assert protocol("fasterwam", 0) == (28, None)
    assert protocol("lingbot_va", 0) == (32, 10000)
    assert protocol("lingbot_va", 2) == (32, 30000)
    assert protocol("openwam", 0) == (32, None)
    assert protocol("longwam", 0) == (24, None)


def test_libero_protocol_per_backend():
    protocol = _script("evaluate_benchmark").libero_protocol
    fastwam = {"action_horizon": 32, "replan_steps": 10, "num_steps_wait": 30, "camera_resolution": 256}
    for backend in ("fastwam", "fasterwam", "longwam"):
        assert protocol(backend, "libero_spatial") == {**fastwam, "max_steps": None}  # evaluator: 400 / 700
    assert protocol("openwam", "libero_spatial") == {**fastwam, "max_steps": 600}
    assert protocol("openwam", "libero_10") == {**fastwam, "max_steps": None}
    assert protocol("lingbot_va", "libero_10") == {"action_horizon": 16, "replan_steps": 16, "num_steps_wait": 5,
                                                    "camera_resolution": 128, "max_steps": 796}


def test_fastwam_task_flag_is_checked(tmp_path, capsys):
    script = _script("evaluate_benchmark")
    checkpoint, stats = tmp_path / "model.pt", tmp_path / "stats.json"
    checkpoint.write_bytes(b"0")
    stats.write_text("{}")
    common = ["--checkpoint", str(checkpoint), "--dataset-stats", str(stats), "--benchmark-root", str(tmp_path),
              "--device", "cpu"]
    for argv in (["--benchmark", "libero", "--backend", "fasterwam", "--fastwam-task", "libero_uncond_2cam224_1e-4"],
                 ["--benchmark", "robotwin", "--backend", "fastwam", "--fastwam-task", "libero_uncond_2cam224_1e-4"]):
        with pytest.raises(SystemExit):
            script.main(argv + common)
        assert "--fastwam-task" in capsys.readouterr().err


def test_robotwin_instruction_choice_is_seeded(monkeypatch):
    import random

    from wam_interp.evaluations.ROBOTWIN.environment import RobotWinFrontend

    fake = SimpleNamespace(generate_episode_descriptions=lambda task, infos, size: [
        {"unseen": random.sample([f"instruction {i}" for i in range(50)], 20), "seen": []}])
    monkeypatch.setitem(sys.modules, "generate_episode_instructions", fake)
    rgb = {"rgb": np.zeros((4, 4, 3), np.uint8)}
    env = SimpleNamespace(setup_demo=lambda **kwargs: np.random.seed(kwargs["seed"]),
                          set_instruction=lambda instruction: None,
                          get_obs=lambda: {"observation": {"head_camera": rgb, "left_camera": rgb, "right_camera": rgb},
                                           "joint_action": {"vector": np.zeros(14)}})
    frontend = object.__new__(RobotWinFrontend)
    frontend.env, frontend.args, frontend.active = env, {}, False
    frontend.config = SimpleNamespace(task_name="fixture", instruction_pool_size=20)
    chosen = []
    for _ in range(2):
        random.seed(12345 + len(chosen))   # different global state each time
        chosen.append(frontend.begin(100000, 0, {"info": {}}, "unseen").instruction)
    assert chosen[0] == chosen[1]


# ---------------------------------------------------------------- partial results survive errors and SIGTERM


def test_libero_results_are_saved_after_each_episode(monkeypatch, tmp_path):
    from wam_interp.evaluations.LIBERO import runner
    from wam_interp.evaluations.LIBERO.config import LiberoEvalConfig
    from wam_interp.evaluations.LIBERO.results import EpisodeResult

    task = SimpleNamespace(language="pick cup", name="fixture")
    monkeypatch.setattr(runner, "_get_suite", lambda name: SimpleNamespace(n_tasks=10, get_task=lambda i: task))
    monkeypatch.setattr(runner, "initial_states_for_task", lambda *args: [0])
    monkeypatch.setattr(runner, "create_libero_environment", lambda *args: SimpleNamespace(close=lambda: None))
    calls = []

    def episode(**kwargs):
        calls.append(kwargs["trial_index"])
        if len(calls) == 3:
            raise RuntimeError("CUDA out of memory")
        return EpisodeResult(task_id=kwargs["task_id"], trial_index=kwargs["trial_index"], initial_state_index=0,
                             seed=kwargs["seed"], success=True, termination_reason="success", policy_steps=5,
                             warmup_steps=0, inference_calls=1, total_reward=1.0)

    monkeypatch.setattr(runner, "_run_episode", episode)
    model = SimpleNamespace(name="fixture", inference_defaults=dict, prepare_observation=lambda o: o,
                            predict_action=lambda r: None, postprocess_actions=lambda a, environment: a)
    with pytest.raises(RuntimeError, match="out of memory"):
        runner.evaluate_libero(model, LiberoEvalConfig(num_trials=5, task_ids=(0,), output_dir=tmp_path))
    saved = json.loads(next(tmp_path.glob("libero_*.json")).read_text())
    assert saved["completed"] is False and saved["error"]["type"] == "RuntimeError"
    assert saved["episodes"] == 2 and saved["partial_selection"] is True
    assert "package_versions" in saved["provenance"]
    assert not list(tmp_path.glob("*.tmp"))


def test_termination_handler_turns_sigterm_into_system_exit():
    from wam_interp.evaluations.provenance import install_termination_handler

    previous = install_termination_handler()
    try:
        with pytest.raises(SystemExit) as caught:
            os.kill(os.getpid(), signal.SIGTERM)
            for _ in range(1000):  # Python runs the handler at the next bytecode boundary
                pass
        assert caught.value.code == 128 + signal.SIGTERM
    finally:
        signal.signal(signal.SIGTERM, previous)


# ---------------------------------------------------------------- incomplete checkpoints are rejected


class TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.mot = torch.nn.Sequential(torch.nn.Linear(3, 3), torch.nn.Linear(3, 3))
        self.video_expert = torch.nn.Linear(2, 2)
        self.proprio_encoder = torch.nn.Linear(4, 4)

    def load_checkpoint(self, payload):
        # Mirrors the vendored FastWAM loader: strict=False, result discarded.
        self.mot.load_state_dict(payload["mot"], strict=False)
        if "proprio_encoder" in payload:
            self.proprio_encoder.load_state_dict(payload["proprio_encoder"], strict=True)


def _load(model, payload, **kwargs):
    from wam_interp.models.checkpoints import check_checkpoint_loads, record_state_dict_loads

    modules = {"mot": model.mot, "video_expert": model.video_expert, "proprio_encoder": model.proprio_encoder}
    with record_state_dict_loads(modules) as loads:
        model.load_checkpoint(payload)
    assert "load_state_dict" not in vars(model.mot)   # patch removed
    return check_checkpoint_loads(loads, model, "fixture.pt", **kwargs)


def test_partial_checkpoints_raise_and_complete_ones_load():
    source = TinyModel()
    full = {"mot": source.mot.state_dict(), "proprio_encoder": source.proprio_encoder.state_dict()}
    summary = _load(TinyModel(), full)
    assert summary["missing_keys"] == 0 and summary["modules_loaded"] == ["mot", "proprio_encoder"]
    partial = {"mot": {k: v for k, v in full["mot"].items() if not k.startswith("1.")},
               "proprio_encoder": full["proprio_encoder"]}
    with pytest.raises(RuntimeError, match="2 missing keys"):
        _load(TinyModel(), partial)
    assert _load(TinyModel(), partial, allow_partial=True)["missing_keys"] == 2
    with pytest.raises(RuntimeError, match="proprio_encoder"):
        _load(TinyModel(), {"mot": full["mot"]})


# ---------------------------------------------------------------- hooks on modules that never run


class Block(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.q = torch.nn.Linear(2, 2)

    def forward(self, x):  # never called: the parent calls self.block.q directly, like MoT code
        return self.q(x)


class Parent(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.block = Block()

    def forward(self, x):
        return self.block.q(x)


def test_tracing_and_interventions_fail_on_modules_that_never_run():
    from wam_interp.interventions.module_output import replace_module_output
    from wam_interp.tracing.hooks import capture_module_outputs

    model, x = Parent(), torch.ones(1, 2)
    recorder = SimpleNamespace(events=[], record=lambda *a, **k: recorder.events.append(a))
    with capture_module_outputs(model, ("block.q",), recorder):
        model(x)
    assert len(recorder.events) == 1
    with pytest.raises(RuntimeError, match="never ran"):
        with capture_module_outputs(model, ("block",), recorder):
            model(x)
    with replace_module_output(model, "block.q", lambda out: out * 0):
        assert torch.count_nonzero(model(x)) == 0
    with pytest.raises(RuntimeError, match="never ran"):
        with replace_module_output(model, "block", lambda out: out * 0):
            model(x)


# ---------------------------------------------------------------- provenance


def test_statistics_hash_finds_transform_stats_and_ignores_the_file_path():
    from wam_interp.evaluations.provenance import model_provenance

    stats = {"action": {"default": {"global_min": [0.0]}}}
    a = SimpleNamespace(name="x", model=None, transforms=SimpleNamespace(dataset_stats={**stats, "_source": "/a.json"}))
    b = SimpleNamespace(name="x", model=None, transforms=SimpleNamespace(dataset_stats={**stats, "_source": "/b.json"}))
    first, second = model_provenance(a), model_provenance(b)
    assert first["statistics_content_sha256"] is not None
    assert first["statistics_content_sha256"] == second["statistics_content_sha256"]
    assert "source_dirty" in first and "status_and_diff_sha256" in first["source_state"]


# ---------------------------------------------------------------- RoboTwin cache clearing and systematic screening errors


def _robotwin_run(monkeypatch, tmp_path, screen, episodes=6):
    from contextlib import contextmanager

    from wam_interp.evaluations.ROBOTWIN import runner
    from wam_interp.evaluations.ROBOTWIN.config import RoboTwinEvalConfig
    from wam_interp.types import EnvironmentObservation, InferenceResult

    rgb = np.zeros((4, 4, 3), np.uint8)
    obs = EnvironmentObservation({"head": rgb, "left_wrist": rgb, "right_wrist": rgb},
                                 {"joint_position": np.zeros(14)}, "do it", {})
    closes = []

    class Frontend:
        step_limit, action_count = 5, 0

        def screen_seed(self, seed, episode):
            return screen(seed)

        def begin(self, seed, episode, info, instruction_type):
            self.action_count = 0
            return obs

        def step(self, action, action_type):
            self.action_count += 1
            return obs, True

        def close(self, clear_cache=False):
            closes.append(bool(clear_cache))

    @contextmanager
    def open_env(config):
        yield Frontend()

    monkeypatch.setattr(runner, "open_environment", open_env)
    model = SimpleNamespace(name="fixture", model=None, inference_defaults=dict, prepare_observation=lambda o: o,
                            predict_action=lambda r: InferenceResult(np.zeros((4, 14))),
                            postprocess_actions=lambda a, environment: a,
                            environment_action_type=lambda environment: "qpos")
    config = RoboTwinEvalConfig("fixture", tmp_path, num_episodes=episodes, output_dir=tmp_path,
                                max_repeated_screen_errors=3)
    return runner, model, config, closes


def test_robotwin_clears_render_cache_every_five_episodes(monkeypatch, tmp_path):
    runner, model, config, closes = _robotwin_run(monkeypatch, tmp_path, lambda seed: (True, {"info": {}}, None))
    report = runner.evaluate_robotwin(model, config)
    assert report["completed"] and report["episodes"] == 6
    assert closes == [False, False, False, False, True, False]


def test_robotwin_repeated_identical_screen_errors_stop_the_search(monkeypatch, tmp_path):
    from wam_interp.evaluations.ROBOTWIN.config import RoboTwinEvalConfig

    reasons = iter(["expert_failed", "OSError: missing asset", "OSError: missing asset", "OSError: missing asset"])
    runner, model, config, _ = _robotwin_run(monkeypatch, tmp_path, lambda seed: (False, None, next(reasons)))
    with pytest.raises(RuntimeError, match="3 times in a row"):
        runner.evaluate_robotwin(model, config)
    saved = json.loads(next(tmp_path.glob("robotwin_*.json")).read_text())
    assert saved["completed"] is False and len(saved["rejected_seeds"]) == 4
    # ordinary expert failures never trigger it
    runner, model, config, _ = _robotwin_run(monkeypatch, tmp_path, lambda seed: (False, None, "expert_failed"))
    with pytest.raises(RuntimeError, match="exhausted"):
        runner.evaluate_robotwin(model, RoboTwinEvalConfig("fixture", tmp_path, num_episodes=1, max_seed_attempts=10,
                                                           max_repeated_screen_errors=3))
