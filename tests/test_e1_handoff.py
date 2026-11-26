"""Review, resume, aggregation and transfer guards; no model or simulator needed."""
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).parents[1]


def load_script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_checkout_init_can_be_collected_as_standalone_module():
    # Pytest uses this import form for a checkout such as wam-e1-20261119.
    spec = importlib.util.spec_from_file_location("__init__", ROOT / "__init__.py",
                                                 submodule_search_locations=None)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    from wam_interp.types import InferenceRequest
    assert module.InferenceRequest is InferenceRequest


@pytest.mark.parametrize("key,value", [("action_seed", -1), ("identity_atol", float("nan")),
    ("execute_steps", True), ("candidate_seeds", [1, "2"]), ("frame_steps", [0, 4, 3]),
    ("suite", "unknown"), ("camera_resolution", 0)])
def test_bad_config_rejected(key, value):
    from wam_interp.experiments.libero_pilot import validate_config
    config = json.loads((ROOT / "configs/experiments/e1_libero.json").read_text())
    config[key] = value
    with pytest.raises(ValueError): validate_config(config)


def test_campaign_skips_complete_and_rejects_partial(tmp_path, monkeypatch):
    campaign = load_script("e1_campaign")
    cases = tmp_path / "cases.json"
    cases.write_text('[{"task_id":0,"initial_state_index":0}]')
    folder = tmp_path / "campaign/task_0_state_0/prepare"
    folder.mkdir(parents=True)
    status = folder / "status.json"
    status.write_text('{"status":"awaiting_semantic_review"}')
    monkeypatch.setattr(sys, "argv", ["e1_campaign", "prepare", "--config", str(ROOT / "configs/experiments/e1_libero.json"),
        "--cases", str(cases), "--output", str(tmp_path / "campaign")])
    monkeypatch.setattr(campaign.subprocess, "run", lambda *a, **kw: pytest.fail("Completed case reran"))
    campaign.main()
    status.write_text('{"status":"failed"}')
    with pytest.raises(RuntimeError, match="Incomplete run"): campaign.main()


def test_analysis_rejects_mixed_protocols(tmp_path, monkeypatch):
    analysis = load_script("analyze_e1")
    paths = []
    for i in range(2):
        path = tmp_path / f"{i}.json"
        path.write_text(json.dumps({"synthetic": True, "state_id": str(i), "opposite_minus_same": 1,
                                   "comparison_id": str(i), "expected_sign": 1}))
        paths.append(str(path))
    monkeypatch.setattr(sys, "argv", ["analyze", "--e1-summaries", *paths, "--output", str(tmp_path / "report.json")])
    with pytest.raises(ValueError, match="incompatible"): analysis.main()
    assert not (tmp_path / "report.json").exists()


def test_handoff_manifest_detects_changed_source(tmp_path, monkeypatch):
    verifier = load_script("verify_e1_handoff")
    monkeypatch.setattr(verifier.subprocess, "check_output", lambda *a, **kw: "revision\n")
    (tmp_path / "HANDOFF_SOURCE.json").write_text(json.dumps({"base_revision": "revision", "files": {"missing.py": "abc"}}))
    with pytest.raises(ValueError, match="missing or changed"): verifier.verify(tmp_path)


def test_review_rejects_incomplete_preparation(tmp_path):
    cli = load_script("e1_libero")
    (tmp_path / "status.json").write_text('{"status":"failed"}')
    with pytest.raises(ValueError, match="not complete"): cli.reviewed_manifest(tmp_path, {})


def test_counterfactual_candidate_recipe_and_validation():
    from wam_interp.experiments.libero_pilot import validate_config
    cli = load_script("e1_libero")
    config = json.loads((ROOT / "configs/experiments/e1_counterfactual_libero.json").read_text())
    validate_config(config)
    requests = cli.candidate_requests(config, "recipient")
    assert len(requests) == 8 and len({r[0] for r in requests}) == 8
    assert requests[0] == ("soup_tomato_seed_0", 0, config["donor_prompts"]["soup_tomato"])
    config["donor_prompts"]["../escape"] = "bad"
    with pytest.raises(ValueError, match="safe identifiers"): validate_config(config)
