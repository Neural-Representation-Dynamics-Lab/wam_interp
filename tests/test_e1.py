"""CPU-only E1 isolation, failure recovery and trajectory measurement checks."""

import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from wam_interp.experiments.e1 import Branch, context_id, run_e1
from wam_interp.experiments.futures import BOUNDARY, apply_future_hook, capture_future
from wam_interp.experiments.trajectories import trajectory_metrics


def inputs():
    context = dict.fromkeys(("repo_revision", "backend", "checkpoint", "statistics", "resolved_config",
        "observation_sha256", "state_sha256", "prompt", "action_seed", "precision",
        "behavior_measure", "state_id"), "fixture")
    manifest = {"context": context, "property": "direction", "recipient_label": "right",
                "expected_sign": -1, "synthetic": True, "donors": {}}
    for arm, label in (("same_property", "right"), ("opposite_property", "left")):
        manifest["donors"][arm] = dict(context_id=context_id(context), label=label,
            reviewed_before_execution=True, nuisance_notes="fixture")
    return manifest, {"same_property": np.ones((2, 1)), "opposite_property": -np.ones((2, 1))}


class Session:
    def __init__(self, fail=False):
        self.state, self.starts, self.fail = 0., [], fail
    def snapshot(self):
        return self.state
    def restore(self, state):
        self.state = state
    def run(self, donor):
        self.starts.append(self.state)
        self.state += 1
        if self.fail:
            raise RuntimeError("fixture failure")
        actions = np.ones((2, 1)) if donor is None else donor.copy()
        return Branch(actions, actions, actions, actions, float(actions.sum()), {})


def test_all_branches_restore_and_save(tmp_path):
    manifest, donors = inputs()
    session = Session()
    result = run_e1(session, donors=donors, manifest=manifest, output=tmp_path / "run")
    assert session.starts == [0., 0., 0., 0.]
    assert session.state == 0.
    assert result["identity_max_action_error"] == 0
    assert result["opposite_minus_same"] == -4
    assert len(list((tmp_path / "run").glob("*.npz"))) == 5


def test_failed_run_restores_and_marks_failed(tmp_path):
    manifest, donors = inputs()
    session = Session(fail=True)
    with pytest.raises(RuntimeError, match="fixture failure"):
        run_e1(session, donors=donors, manifest=manifest, output=tmp_path / "run")
    assert session.state == 0.
    assert json.loads((tmp_path / "run/status.json").read_text())["status"] == "failed"


@pytest.mark.parametrize("field,value", [("context_id", "wrong"), ("label", "left"),
    ("reviewed_before_execution", False), ("nuisance_notes", "")])
def test_invalid_donors_rejected_before_execution(tmp_path, field, value):
    manifest, donors = inputs()
    manifest["donors"]["same_property"][field] = value
    session = Session()
    with pytest.raises(ValueError):
        run_e1(session, donors=donors, manifest=manifest, output=tmp_path / "run")
    assert session.starts == []


def test_hook_preserves_anchor_and_cleans_up():
    model = SimpleNamespace(research_future_boundary=BOUNDARY)
    original = torch.ones(1, 2, 3, 2, 2)
    with capture_future(model, torch.zeros_like(original)) as capture:
        patched = apply_future_hook(model, original)
    assert torch.equal(patched[:, :, :1], original[:, :, :1])
    assert patched[:, :, 1:].sum() == 0
    assert torch.equal(original, torch.ones_like(original))
    assert capture["used"].device.type == "cpu"
    assert not hasattr(model, "_research_future_hook")


def test_hook_error_cleans_up():
    model = SimpleNamespace(research_future_boundary=BOUNDARY)
    with pytest.raises(ValueError):
        with capture_future(model, torch.zeros(3)):
            apply_future_hook(model, torch.ones(1, 2, 3, 2, 2))
    assert not hasattr(model, "_research_future_hook")


def test_missing_boundary_and_nested_hooks_rejected():
    model = SimpleNamespace(research_future_boundary=BOUNDARY)
    with pytest.raises(RuntimeError, match="not reached"):
        with capture_future(model):
            pass
    with capture_future(model):
        with pytest.raises(RuntimeError, match="already active"):
            with capture_future(model):
                pass
        apply_future_hook(model, torch.ones(1, 2, 3, 2, 2))


def test_anchor_and_nonfinite_rejected():
    tensor = torch.ones(1, 2, 3, 2, 2)
    for value in (0., float("nan")):
        model = SimpleNamespace(_research_future_hook=lambda x: torch.full_like(x, value))
        with pytest.raises(ValueError):
            apply_future_hook(model, tensor)


def test_trajectory_metrics_known_geometry():
    result = trajectory_metrics([0, 1, 2], [[0, 0], [1, 0], [2, 0]],
                                [[0, 0], [1, 1], [2, 2]])
    assert result["ade"] == 1
    assert result["fde"] == 2
    assert result["displacement_error"] == 2


def test_missing_endpoint_is_not_last_visible_point():
    result = trajectory_metrics([0, 1, 2], [[0, 0], [1, 0], [np.nan, 0]],
                                [[0, 0], [1, 0], [2, 0]])
    assert result["fde"] is None
    assert result["coverage"] == 2 / 3


def test_replanning_prefix_excludes_unexecuted_future():
    result = trajectory_metrics([0, 1, 2], [[0, 0], [1, 0], [100, 0]],
                                [[0, 0], [1, 0], [2, 0]], executed_until=1)
    assert result["fde"] == result["ade"] == 0
    assert result["eligible_frames"] == 2


@pytest.mark.parametrize("times", [[0, 0], [1, 0], [0, np.nan], []])
def test_invalid_timestamps(times):
    with pytest.raises(ValueError):
        trajectory_metrics(times, np.zeros((len(times), 2)), np.zeros((len(times), 2)))


def test_all_missing_tracks_have_no_score():
    result = trajectory_metrics([0, 1], [[np.nan, 0], [np.nan, 0]], [[0, 0], [1, 1]])
    assert result["coverage"] == 0
    assert result["ade"] is result["fde"] is result["stationary_ade"] is None


def test_real_bridge_uses_adapter_decoding_and_prefix():
    from wam_interp.experiments.fastwam_session import FastWAMChunkSession
    from wam_interp.types import InferenceRequest, PreparedObservation, InferenceResult
    class Adapter:
        model = SimpleNamespace(research_future_boundary=BOUNDARY)
        def predict_action(self, request):
            future = apply_future_hook(self.model, torch.ones(1, 1, 3, 1, 1))
            return InferenceResult(future[0, 0, 1:, 0, 0].reshape(2, 1))
        def postprocess_actions(self, actions, *, environment):
            return actions * 2
    request = InferenceRequest(PreparedObservation(None, None, "test"), condition="idm")
    commands_seen = []
    def rollout(commands):
        commands_seen.append(commands)
        return {"executed_commands": commands, "behavior": 4.}
    session = FastWAMChunkSession(Adapter(), request, snapshot_fn=lambda: {},
        restore_fn=lambda state: None, rollout_fn=rollout, execute_steps=1)
    branch = session.run(None)
    assert commands_seen[0].tolist() == [[2.]]
    assert branch.actions.tolist() == [[1.], [1.]]
    assert branch.behavior == 4.
    assert branch.future.shape == (1, 1, 3, 1, 1)


def test_restore_failure_does_not_report_complete(tmp_path):
    manifest, donors = inputs()
    class BrokenRestore(Session):
        count = 0
        def restore(self, state):
            self.count += 1
            if self.count == 5:
                raise RuntimeError("restore failed")
            super().restore(state)
    with pytest.raises(RuntimeError, match="restore failed"):
        run_e1(BrokenRestore(), donors=donors, manifest=manifest, output=tmp_path / "run")
    assert json.loads((tmp_path / "run/status.json").read_text())["status"] == "failed"


def test_cpu_analysis_import_does_not_load_models():
    import subprocess
    import sys
    subprocess.run([sys.executable, "-c",
        "import sys; import wam_interp.experiments.e1; import wam_interp.experiments.trajectories; "
        "assert 'torch' not in sys.modules; assert 'mujoco' not in sys.modules"], check=True)
