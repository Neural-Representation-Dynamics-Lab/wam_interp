"""Concrete pilot restoration and review tests without LIBERO or checkpoints."""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from wam_interp.types import EnvironmentObservation, PreparedObservation
from wam_interp.experiments.libero_pilot import LiberoPilot, rng_snapshot, rng_restore, validate_config
from wam_interp.experiments.annotation import make_annotation_page, analyze_annotations
from wam_interp.experiments.futures import BOUNDARY, capture_future, apply_future_hook


def config():
    return json.loads((Path(__file__).parents[1] / "configs/experiments/e1_libero.json").read_text())


def observation(x=0):
    return EnvironmentObservation({"agentview": np.zeros((8, 8, 3), np.uint8),
        "wrist": np.zeros((8, 8, 3), np.uint8)},
        {"eef_position": np.array([x, 0., 0.]), "eef_quaternion": np.array([0, 0, 0, 1]),
         "gripper_position": np.zeros(2)}, "pick")


def test_recreate_controller_and_replay():
    envs = []
    def create(*args):
        env = SimpleNamespace(x=0., closed=False, controller_steps=0)
        env.close = lambda: setattr(env, "closed", True)
        envs.append(env)
        return env
    def reset(env, state, **kw): return observation()
    def step(env, command, **kw):
        env.controller_steps += 1
        env.x += float(command[0])
        return SimpleNamespace(observation=observation(env.x), done=False, success=False)
    adapter = SimpleNamespace(inference_defaults=lambda: {"num_video_frames": 9},
        prepare_observation=lambda o: PreparedObservation(None, None, o.instruction))
    pilot = LiberoPilot(config(), adapter, task=SimpleNamespace(language="pick"), initial_state=np.zeros(3),
        create_env=create, reset_env=reset, step_env=step)
    try:
        pilot.verify_replay()
        pilot.rollout(np.array([[.1, 0, 0, 0, 0, 0, -1]]))
        assert pilot.last_rollout["eef_position"][-1, 0] == .1
        pilot.restore(pilot.saved_rng)
        assert pilot.env.controller_steps == config()["warmup_steps"]
        assert all(env.closed for env in envs[:-1])
    finally:
        pilot.close()
    assert all(env.closed for env in envs)


def test_rng_restores_cpu():
    import random
    before = rng_snapshot()
    expected = (random.random(), np.random.rand(), torch.rand(1))
    rng_restore(before)
    assert random.random() == expected[0]
    assert np.random.rand() == expected[1]
    assert torch.equal(torch.rand(1), expected[2])


def test_counterfactual_donor_preserves_recipient_request(monkeypatch):
    import wam_interp.experiments.libero_pilot as module
    from wam_interp.types import InferenceRequest
    request = InferenceRequest(PreparedObservation(torch.zeros(1), torch.zeros(1), "original task"),
                               condition="idm", seed=7)
    seen = []
    def predict(adapter, received, donor):
        seen.append(received)
        return None, {}
    monkeypatch.setattr(module, "predict_with_future", predict)
    pilot = SimpleNamespace(request=request, saved_rng={}, restore=lambda state: None, adapter=None)
    LiberoPilot.infer(pilot, seed=2, prompt="alternate task")
    assert pilot.request is request and request.observation.prompt == "original task" and request.seed == 7
    assert seen[0].observation.prompt == "alternate task" and seen[0].seed == 2
    assert seen[0].observation.image is request.observation.image
    assert seen[0].observation.proprio is request.observation.proprio


def test_rng_restores_local_cuda():
    if not torch.cuda.is_available(): pytest.skip("CUDA unavailable")
    torch.empty(1, device="cuda")
    state = rng_snapshot()
    expected = torch.rand(4, device="cuda")
    rng_restore(state)
    assert torch.equal(expected, torch.rand(4, device="cuda"))


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_future_boundary_on_available_device(device):
    if device == "cuda" and not torch.cuda.is_available(): pytest.skip("CUDA unavailable")
    model = SimpleNamespace(research_future_boundary=BOUNDARY)
    original = torch.ones(1, 2, 3, 2, 2, dtype=torch.bfloat16, device=device)
    with capture_future(model, torch.zeros_like(original)) as captured:
        result = apply_future_hook(model, original)
    assert result.dtype == torch.bfloat16
    assert torch.equal(result[:, :, :1], original[:, :, :1])
    assert captured["used"].device.type == "cpu"


def test_annotation_roundtrip_and_missing(tmp_path):
    spec = {"synthetic": True, "condition": "baseline", "state_id": "fixture",
            "coordinate_frame": "agentview", "units": "normalized_image_xy",
            "pairs": [{"step": i, "imagined": "a.png", "executed": "b.png"} for i in (0, 4)]}
    path = tmp_path / "pairs.json"
    path.write_text(json.dumps(spec))
    html = make_annotation_page(path)
    assert "canvas" in html.read_text()
    spec["records"] = [{"step": i, "gripper": {"imagined": [0., 0.], "executed": [0., .1]},
                        "object": {"imagined": None, "executed": None}} for i in (0, 4)]
    path.write_text(json.dumps(spec))
    result = analyze_annotations(path)
    assert result["metrics"]["gripper"]["ade"] == .1
    assert result["metrics"]["object"]["ade"] is None
    spec["records"][1]["step"] = 5
    path.write_text(json.dumps(spec))
    with pytest.raises(ValueError, match="timestamps"):
        analyze_annotations(path)


def test_config_and_review_guard(tmp_path):
    validate_config(config())
    path = Path(__file__).parents[1] / "scripts/e1_libero.py"
    spec = importlib.util.spec_from_file_location("pilot_cli", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    (tmp_path / "context.json").write_text('{}')
    (tmp_path / "status.json").write_text('{"status":"awaiting_semantic_review"}')
    (tmp_path / "review.json").write_text(json.dumps({"reviewed_before_execution": False}))
    with pytest.raises(ValueError, match="Review semantics"):
        module.reviewed_manifest(tmp_path, {})


def test_full_prepare_review_run_pipeline(tmp_path, monkeypatch):
    from PIL import Image
    from wam_interp.types import InferenceResult
    from wam_interp.experiments.e1 import context_id
    path = Path(__file__).parents[1] / "scripts/e1_libero.py"
    spec = importlib.util.spec_from_file_location("pilot_pipeline", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    c = config()
    c.update(candidate_seeds=[1, 2], frame_steps=[0, 1, 2], execute_steps=2)
    model = SimpleNamespace(research_future_boundary=BOUNDARY, device="cpu", torch_dtype=torch.float32,
        _decode_latents=lambda latents: [Image.new("RGB", (16, 8)) for _ in range(3)])
    class Adapter:
        def __init__(self): self.model = model
        def inference_defaults(self): return {"num_video_frames": 9}
        def prepare_observation(self, o):
            return PreparedObservation(torch.zeros(1, 3, 8, 16), torch.zeros(8), o.instruction)
        def predict_action(self, request):
            future = torch.ones(1, 1, 3, 1, 1) * (-1 if request.seed == 2 else 1)
            future[:, :, 0] = 0
            future = apply_future_hook(model, future)
            actions = torch.zeros(2, 7)
            actions[:, 0] = future[0, 0, 1:, 0, 0] * .01
            return InferenceResult(actions)
        def postprocess_actions(self, actions, **kwargs): return actions
    def create(*args):
        return SimpleNamespace(x=0., close=lambda: None)
    def step(env, action, **kwargs):
        env.x += float(action[0])
        return SimpleNamespace(observation=observation(env.x), done=False, success=False)
    pilot = LiberoPilot(c, Adapter(), task=SimpleNamespace(language="pick"), initial_state=np.zeros(3),
        create_env=create, reset_env=lambda *a, **k: observation(), step_env=step)
    context = dict.fromkeys(("repo_revision", "backend", "checkpoint", "statistics", "resolved_config",
        "observation_sha256", "state_sha256", "prompt", "action_seed", "precision",
        "behavior_measure", "state_id"), "fixture")
    monkeypatch.setattr(module, "provenance", lambda p: context)
    prepared, output = tmp_path / "prepared", tmp_path / "run"
    try:
        module.prepare(pilot, prepared)
        review = json.loads((prepared / "review.json").read_text())
        review.update(reviewed_before_execution=True, frame_mapping_verified=True,
            recipient_label="right", expected_sign=-1,
            same_property="candidate_1", opposite_property="candidate_2")
        for item, label in zip(review["candidates"], ["right", "left"]):
            item.update(acceptable=True, label=label, nuisance_notes="synthetic fixture")
        tampered = json.loads(json.dumps(review))
        tampered["candidates"][0]["generation_request"]["prompt"] = "silently changed"
        (prepared / "review.json").write_text(json.dumps(tampered))
        with pytest.raises(ValueError, match="instruction provenance"):
            module.reviewed_manifest(prepared, context)
        (prepared / "review.json").write_text(json.dumps(review))
        summary = module.run(pilot, prepared, output)
        assert summary["opposite_minus_same"] < 0
        assert len(list(output.glob("*_frame_pairs.html"))) == 4
        assert len(list(output.glob("*_execution.npz"))) == 4
        assert json.loads((output / "status.json").read_text())["status"] == "complete"
    finally:
        pilot.close()
