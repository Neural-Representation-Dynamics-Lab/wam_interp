"""Regression checks for the native pilot's physical and matched-control gates."""

import hashlib
import sys
from pathlib import Path

import numpy as np
import pytest
from wam_interp.experiments.obstacle import Barrier


@pytest.fixture
def pilot(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    import e1_native_avoidance

    return e1_native_avoidance


def record(points, hits=()):
    return {
        "path": np.asarray(points).tolist(),
        "decisions": [0],
        "robot_barrier_contacts": list(hits),
    }


def test_gate_needs_progress_no_contact_and_small_offroute(pilot):
    box = Barrier((0.0, 0.0, 0.0), (1.0, 0.0), (0.01, 0.02, 0.01), 1, False)
    clear = record([[-0.1, 0, 0], [0, 0, 0], [0.05, 0, 0]])
    live = record([[-0.1, 0, 0], [0, -0.05, 0], [0.05, -0.05, 0]])
    replay = {"robot_barrier_contacts": [2]}
    off = {"robot_barrier_contacts": []}
    witness = [{"reached_bar_plane": True, "robot_contacts": 0}]
    cfg = {"min_clearance_gain_mm": 5, "max_offroute_rms_mm": 5}

    def score(tested, shifted=clear, collision_test=replay, witnesses=witness):
        return pilot.compare(
            clear, tested, shifted, collision_test, off, witnesses, box, cfg
        )

    assert score(live)["native_pass"]
    assert not score(dict(live, robot_barrier_contacts=[2]))["native_pass"]
    assert not score(record([[-0.1, 0, 0], [-0.07, -0.05, 0]]))["native_pass"]
    assert not score(live, shifted=live)["native_pass"]
    assert not score(live, collision_test=off)["fixture_pass"]
    assert not score(live, witnesses=[])["fixture_pass"]
    assert score(live, shifted=record(clear["path"][:2]))["off_route_rms_mm"] is None


def test_witness_bias_is_mirrored_and_preserves_input(pilot):
    commands = np.zeros((48, 7))
    for side in (-1, 1):
        box = Barrier((0, 0, 0), (1, 0), (1, 1, 1), side, False)
        biased = pilot.witness_actions(commands, box, 0.35)
        assert np.allclose(biased[:20, 1], -0.35 * side)
        assert not biased[20:].any()
    assert not commands.any()


def test_score_only_accepts_windows_newlines_and_requires_receipts(
    tmp_path, pilot, monkeypatch
):
    design = (
        Path(__file__).resolve().parents[1]
        / "configs/experiments/e1_native_avoidance.json"
    )
    lf = design.read_bytes().replace(b"\r\n", b"\n")
    spec = tmp_path / "design.json"
    spec.write_bytes(lf.replace(b"\n", b"\r\n"))
    (tmp_path / "design.sha256").write_text(hashlib.sha256(lf).hexdigest())
    monkeypatch.setattr(
        sys,
        "argv",
        ["pilot", "--score-only", "--design", str(spec), "--out", str(tmp_path)],
    )
    monkeypatch.setattr(
        pilot, "load", lambda *_: pytest.fail("CPU scoring loaded the policy")
    )
    monkeypatch.setattr(
        pilot, "_get_suite", lambda *_: pytest.fail("CPU scoring loaded the simulator")
    )
    with pytest.raises(FileNotFoundError, match="Missing pilot episode"):
        pilot.main()
