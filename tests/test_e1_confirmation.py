import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from wam_interp.experiments.confirm_sources import tint_pair
from wam_interp.experiments.confirmation import measured_effects, reuse_fastwam_video
from wam_interp.experiments.future_adapter import FutureAdapter
from wam_interp.models.longwam.transforms import robotwin_model_image


def analysis():
    path = Path(__file__).parents[1] / "scripts/e1_confirm_analysis.py"
    spec = importlib.util.spec_from_file_location("tests_e1_analysis", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_edits_share_anchor_geometry_and_symmetric_headroom():
    rgb = np.tile(np.array([100, 150, 200], np.uint8), (8, 8, 1))
    rgb[1, 1, 0] = 0
    mask = np.zeros((8, 8), bool)
    mask[1:4, 1:4] = True
    assert all(np.array_equal(rgb, v) for v in tint_pair(rgb, mask, 0))
    warm, cool = tint_pair(rgb, mask, 3)
    np.testing.assert_array_equal(warm.astype(int) - rgb, rgb.astype(int) - cool)
    np.testing.assert_array_equal(rgb[~mask], warm[~mask])
    np.testing.assert_array_equal(warm[1, 1], rgb[1, 1])
    assert np.any(warm[mask] != rgb[mask])


def test_background_decision_includes_opposing_trajectories():
    paths = {"native": np.zeros((33, 3)), "sham": np.zeros((33, 3))}
    for d, s in (("left", -1), ("right", 1)):
        neutral = paths["native"].copy()
        neutral[1:, 0] = s * 0.010
        for name, offset in (("neutral", 0), ("warm", 0.009), ("cool", -0.009)):
            p = neutral.copy()
            p[1:, 1] += offset * s
            paths[f"{d}_{name}"] = p
    m = measured_effects(paths)
    assert m["motion_32_mm"] == pytest.approx(10)
    assert m["background_rms_max_mm"] == pytest.approx(9)
    assert m["signed_background_half_mm"] == pytest.approx(0)
    assert m["selectivity_mm"] == pytest.approx(10)


def test_raw_robotwin_views_use_native_float_transform_at_all_times():
    a = SimpleNamespace(
        benchmark="robotwin",
        image_meta=[
            {"key": k, "shape": [3, 224, 224]} for k in ("cam_high", "cam_left_wrist", "cam_right_wrist")
        ],
    )
    api = object.__new__(FutureAdapter)
    api.adapter = a
    api.owner = SimpleNamespace(device="cpu", torch_dtype=torch.float32)
    api.backend = "longwam"
    api._from_prepared_video = lambda req, video, noise: (video, noise)
    rng = np.random.default_rng(1)
    frames = [
        {
            k: rng.integers(0, 256, (61, 83, 3), dtype=np.uint8)
            for k in ("head", "left_wrist", "right_wrist")
        }
        for _ in range(3)
    ]
    vid, seed = api.from_camera_frames(None, frames, noise_seed=3)
    assert vid.shape == (1, 3, 3, 384, 320) and seed == 3
    for t, f in enumerate(frames):
        expected = robotwin_model_image(f, a.image_meta, device="cpu", dtype=torch.float32)
        assert torch.equal(vid[:, :, t], expected)


def test_preintervention_cache_checks_all_inputs_and_cleans_up_on_error():
    calls = []

    def denoise(**kw):
        calls.append(1)
        return kw["context"] + 2

    api = SimpleNamespace(backend="fastwam", owner=SimpleNamespace(_denoise_video=denoise))
    args = {"timestep_video": torch.tensor(1.0), "context": torch.tensor(3.0)}
    with pytest.raises(ValueError, match="input changed"), reuse_fastwam_video(api):
        api.owner._denoise_video(**args)
        api.owner._denoise_video(**args)
        api.owner._denoise_video(**{**args, "context": torch.tensor(4.0)})
    assert api.owner._denoise_video is denoise and len(calls) == 1


def test_state_level_gate_and_missing_seeds(tmp_path):
    a = analysis()
    for split, ids in (("dev", [1, 2]), ("held", list(range(10, 20)))):
        spec = {
            "tasks": [5],
            "states": {"5": ids},
            "seeds": [1, 2],
            "minimum_complete_states_per_task": 4,
            "minimum_task_coverage": 0.5,
        }
        a.write(tmp_path / f"{split}.json", spec)
        for idx in ids:
            for seed in (1, 2):
                p = tmp_path / split / f"task_5/state_{idx}/seed_{seed}"
                a.write(p / "fastwam_source.json", {"passed": not (idx == 19 and seed == 2)})
                np.savez(
                    p / "fastwam_features.npz",
                    arm=np.array([1.0, 0]),
                    colour=np.array([0.0, 1.0 if split == "dev" or idx % 2 else -1.0]),
                )
    g = a.gate(
        tmp_path / "dev.json", tmp_path / "held.json", tmp_path / "dev", tmp_path / "held", "fastwam"
    )
    assert g["properties"]["arm"]["n"] == 9 and g["properties"]["arm"]["passed"]
    assert not g["properties"]["colour"]["passed"]
    assert a.power_count(8) == 34


def test_analysis_uses_independent_starts_and_retains_sham_failures(tmp_path):
    a = analysis()
    spec = {
        "tasks": [3, 5],
        "states": {"3": list(range(10)), "5": list(range(10))},
        "seeds": [1, 2],
        "minimum_complete_states_per_task": 4,
        "minimum_task_coverage": 0.5,
        "suite": "libero_10",
        "background_rms_margin_mm": 5,
        "sham_margin_mm": 20,
    }
    manifest = tmp_path / "held.json"
    a.write(manifest, spec)
    root = tmp_path / "held"
    for task, idx, folders in a.cases(spec, root):
        for seed, folder in zip(spec["seeds"], folders):
            a.write(folder / "fastwam_source.json", {"passed": True, "sha256": "abc"})
            vals = {k: 1.0 for k in a.METRICS}
            vals.update(
                motion_32_mm=6.0 + idx,
                background_rms_max_mm=2.0,
                selectivity_mm=5.0,
                sham_drift_mm=30.0 if idx == 0 else 1.0,
            )
            a.write(
                folder / "fastwam_behavior.json",
                {
                    "eligible": idx != 0,
                    "metrics": vals,
                    "depicted_half_mm": 30.0,
                    "source_hash": "abc",
                    "action_seed": seed,
                },
            )
    check = {
        "manifest_sha256": a.digest(manifest),
        "properties": {k: {"passed": True} for k in a.PROPERTIES},
    }
    a.write(root / "fastwam_feature_gate.json", check)
    r = a.summarize(manifest, root, "fastwam")
    assert r["usable"] == {"3": 9, "5": 9} and r["attempted"] == 20
    assert r["quality"]["sampled_seeds"] == 40 and r["quality"]["largest_sham_drift_mm"] == 30.0
    assert r["motion_generalizes"] and r["background_decision"] == "bounded mean response"
    assert r["sensitivity_all_complete_starts"]["motion_32_mm"]["mean"] == 10.5
    a.write(tmp_path / "result.json", r)


def test_own_source_reuses_only_sealed_native_chunks(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).parents[1] / "scripts"))
    from e1_longwam_reference import copy_native

    a = analysis()
    shared, dest = tmp_path / "original", tmp_path / "own"
    a.write(dest / "held.json", {"tasks": [5], "states": {"5": [34]}, "seeds": [1, 2]})
    native = np.arange(224, dtype="float32").reshape(32, 7)
    for seed in (1, 2):
        folder = shared / f"held/task_5/state_34/seed_{seed}"
        folder.mkdir(parents=True)
        np.savez(folder / "longwam_commands.npz", native=native, sham=native + 10)
    a.write(shared / "held/task_5/state_34/seed_1/longwam_commands.json", {"order": ["sham", "native"]})
    copy_native(shared, dest, "held")
    rows = a.read(dest / "held_reference_ledger.json")
    assert [r["copied"] for r in rows] == [True, False]
    np.testing.assert_array_equal(
        np.load(dest / "held/task_5/state_34/seed_1/longwam_reference.npy"), native
    )
    target = dest / "held/task_5/state_34/seed_2/longwam_reference.npy"
    assert not target.exists()
