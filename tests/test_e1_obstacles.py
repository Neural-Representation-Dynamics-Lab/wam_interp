import json
from pathlib import Path
from xml.etree import ElementTree as ET

import numpy as np
import pytest
import torch
from wam_interp.experiments.clip_cache import _replace_future
from wam_interp.experiments.obstacle import placement, score
from wam_interp.experiments.obstacle_libero import (
    recipient_clip,
    settle_without_barrier,
)


def path():
    return np.column_stack(
        (np.linspace(0, 0.16, 33), np.zeros(33), np.linspace(0.7, 0.61, 33))
    )


def test_short_rollout_records_exclusion_before_readout(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    from e1_obstacle_report import summarize

    case = tmp_path / "task_0/state_0/seed_1"
    case.mkdir(parents=True)
    (case / "sources.json").write_text(
        json.dumps({"task": 0, "state": 0, "seed": 1, "passed": True, "placements": {}})
    )
    (case / "behavior.json").write_text(
        json.dumps({"passed": False, "clear_sham_drift_mm": 0, "metrics": {}})
    )
    arms = ("clear", "minus", "plus", "off_minus", "off_plus")
    np.savez(case / "paths.npz", **{"clear__" + arm: np.zeros((5, 3)) for arm in arms})
    summary = summarize(tmp_path)
    assert summary["by_task"]["0"] == {"count": 0, "attempted": 1}
    assert "future_only_mm_at_8" not in summary["rows"][0]


def test_held_report_flags_missing_side_feature_for_eligible_start(
    tmp_path, monkeypatch
):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    import e1_obstacle_held_report as held

    configs = Path(__file__).resolve().parents[1] / "configs/experiments"
    manifest = configs / "e1_obstacle_fastwam_task0.json"
    spec = json.loads(manifest.read_text())
    for state in spec["side_template_states"]:
        case = tmp_path / f"dev/fastwam/task_0/state_{state}/seed_0"
        case.mkdir(parents=True)
        np.save(case / "side_difference.npy", [1.0, 0.0])
    row = {"task": 0, "state": spec["states"]["0"][0], "seed": 1, "behavior_pass": True}
    monkeypatch.setattr(held, "summarize", lambda folder: {"rows": [row]})
    with pytest.raises(
        FileNotFoundError, match="Missing side feature for eligible start"
    ):
        held.report(manifest, tmp_path / "dev", tmp_path / "held")


def test_mirrored_box_has_collision_mask_and_signed_side():
    trajectory = path()
    for side in (-1, 1):
        box = placement(trajectory, side)
        off = placement(trajectory, side, off_route=True)
        assert box.clearance(trajectory)[28] < 0 < off.clearance(trajectory)[28]
        xml = ET.fromstring(box.xml("<mujoco><worldbody/></mujoco>"))
        bar = xml.find(".//geom")
        assert bar.get("contype") == bar.get("conaffinity") == "1"
        assert not xml.findall(".//joint")
        bypass = trajectory.copy()
        bypass[16, 1] -= side * 0.012
        assert score(bypass, trajectory, box, 16)["open_side_mm"] == pytest.approx(12)
    with pytest.raises(ValueError):
        placement(np.zeros((33, 3)), -1)


def test_recipient_anchor_is_independent_copy():
    live = np.zeros((9, 3, 4, 3), np.uint8)
    obstacle = np.ones_like(live)
    for first, future in ((live, obstacle), (obstacle, live)):
        clip = recipient_clip(future, first)
        assert np.array_equal(clip[0], first[0])
        assert np.array_equal(clip[1:], future[1:])
        assert not np.shares_memory(clip, future)


def test_restore_collision_mask_on_failure():
    from types import SimpleNamespace

    model = SimpleNamespace(
        geom_name2id=lambda name: 0,
        geom_contype=np.array([3]),
        geom_conaffinity=np.array([7]),
    )
    env = SimpleNamespace(sim=SimpleNamespace(model=model))
    with pytest.raises(RuntimeError), settle_without_barrier(env):
        assert not model.geom_contype[0] and not model.geom_conaffinity[0]
        raise RuntimeError()
    assert (model.geom_contype[0], model.geom_conaffinity[0]) == (3, 7)


def test_cache_donor_preserves_each_cfg_anchor_and_inputs():
    native = torch.randn(2, 3, 4, 5, 6)
    future = torch.randn(1, 3, 3, 5, 6)
    copy = native.clone()
    patched = _replace_future(native, future)
    assert torch.equal(patched[:, :, :1], native[:, :, :1])
    assert torch.equal(patched[:, :, 1:], future.expand(2, -1, -1, -1, -1))
    assert torch.equal(native, copy)
    with pytest.raises(ValueError):
        _replace_future(native, future[:, :, :1])


def test_fasterwam_one_pass_has_zero_clean_signal():
    from wam_interp.models.fasterwam.model_code.schedulers.scheduler_continuous import (
        WanContinuousFlowMatchScheduler,
    )

    scheduler = WanContinuousFlowMatchScheduler()
    t = scheduler.build_inference_schedule(10, torch.device("cpu"), torch.bfloat16)[0][
        0
    ]
    noise = torch.tensor([0.25])
    assert t.item() == scheduler.num_train_timesteps
    assert torch.equal(
        scheduler.add_noise(torch.zeros(1), noise, t),
        scheduler.add_noise(torch.ones(1), noise, t),
    )


def test_hold_last_sample_excludes_contact_frame():
    from wam_interp.experiments.obstacle_libero import hold_after

    raw = np.arange(33)
    first_contact = 31
    prefix = 4 * ((first_contact - 1) // 4)
    held = hold_after(raw, prefix)
    assert held[28:].tolist() == [28] * 5
    assert np.array_equal(held[:29], raw[:29])
    assert raw[-1] == 32
    with pytest.raises(ValueError):
        hold_after(raw, -1)


def test_second_placement_uses_the_later_reach_and_same_side_convention():
    trajectory = path()
    early = placement(trajectory, 1)
    later = placement(trajectory, 1, station=32)
    off = placement(
        trajectory,
        1,
        station=32,
        off_route=True,
        off_distance=0.024,
        forward_offset=-0.13,
    )
    assert np.allclose(
        np.array(later.center) - early.center, trajectory[32] - trajectory[28]
    )
    assert np.allclose(
        np.array(off.center)[:2] - later.center[:2], -0.13 * np.array(later.forward)
    )
    with pytest.raises(ValueError):
        placement(trajectory, 1, station=5)
