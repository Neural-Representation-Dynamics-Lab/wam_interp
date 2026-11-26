"""Keep CPU planning honest about native evaluation versus E1 implementation."""

import json
from pathlib import Path
import subprocess
import sys

import pytest

from wam_interp.experiments.model_capabilities import (
    CAPABILITIES, baseline_argv, capability, require_future_transplant, validate_local_assets,
)

ROOT = Path(__file__).resolve().parents[1]


def command(variant, **changes):
    values = dict(python="python", checkpoint="/weights/model", benchmark_root="/sim/LIBERO",
                  output="results/new run", dataset_stats="/weights/stats.json")
    values.update(changes)
    return baseline_argv(variant, **values)


def test_import_is_lightweight_in_fresh_process():
    code = "from wam_interp.experiments.model_capabilities import capability_report; import sys; " \
           "assert not {'torch', 'mujoco', 'libero', 'transformers'} & sys.modules.keys(); " \
           "assert len(capability_report()['models']) == 6"
    subprocess.run([sys.executable, "-c", code], cwd=ROOT, check=True)


@pytest.mark.parametrize("entry", CAPABILITIES, ids=lambda c: c.variant)
def test_native_baseline_preserves_model_specific_condition(entry):
    argv = command(entry.variant)
    assert argv[argv.index("--backend") + 1] == entry.backend
    assert argv[argv.index("--condition") + 1] == entry.condition
    assert argv[argv.index("--dtype") + 1] == "bfloat16"
    assert argv[argv.index("--suite") + 1] == "libero_10"
    assert "--openwam-upstream-optimizations" not in argv
    assert "results/new run" in argv  # argv, not an interpolated shell command
    assert all((ROOT / path).is_file() for path in entry.source_files)


def test_optional_idm_comparison_reuses_same_checkpoint_and_task():
    a, b = command("fastwam_optional_idm"), command("fastwam_optional_first_frame")
    a[a.index("--condition") + 1] = b[b.index("--condition") + 1]
    assert a == b
    assert a[a.index("--fastwam-task") + 1] == "libero_optional_idm_2cam224_1e-4"


def test_only_native_future_modes_offer_a_transplant_boundary():
    assert require_future_transplant("fastwam_optional_idm").boundary.startswith("fastwam.idm")
    with pytest.raises(NotImplementedError):
        require_future_transplant("fastwam_optional_first_frame")
    for entry in CAPABILITIES[2:]:
        assert require_future_transplant(entry.variant).e1_status == "native_feature_hook_implemented"
    with pytest.raises(ValueError):
        capability("unknown_model")


@pytest.mark.parametrize("changes", [dict(episodes=0), dict(episodes=True), dict(seed=-1),
    dict(task_ids=[]), dict(task_ids=[0, 0]), dict(task_ids=[10]), dict(dataset_stats=None)])
def test_invalid_baseline_specs_fail_before_execution(changes):
    with pytest.raises(ValueError):
        command("fastwam_optional_idm", **changes)


def test_asset_kind_and_longwam_config_checks(tmp_path):
    checkpoint, stats = tmp_path / "model.pt", tmp_path / "dataset_stats.json"
    checkpoint.write_bytes(b"test fixture, not weights")
    stats.write_text("{}", encoding="utf-8")
    with pytest.raises(FileNotFoundError, match="config.yaml"):
        validate_local_assets("longwam_idm", checkpoint=checkpoint, dataset_stats=stats)
    (tmp_path / "config.yaml").write_text("{}", encoding="utf-8")
    validate_local_assets("longwam_idm", checkpoint=checkpoint, dataset_stats=stats)
    with pytest.raises(FileNotFoundError, match="directory"):
        validate_local_assets("lingbot_va", checkpoint=checkpoint)
    validate_local_assets("lingbot_va", checkpoint=tmp_path)


def test_cli_saves_portable_plan_without_loading_assets(tmp_path):
    target = tmp_path / "plan.json"
    argv = [sys.executable, str(ROOT / "scripts/e1_model_matrix.py"), "baseline",
            "--variant", "lingbot_va", "--checkpoint", "/not/downloaded/weights",
            "--benchmark-root", "/not/downloaded/LIBERO", "--output", "results/new run",
            "--save-plan", str(target)]
    subprocess.run(argv, cwd=ROOT, check=True, capture_output=True)
    result = json.loads(target.read_text(encoding="utf-8"))
    assert result["status"] == "planned_not_executed"
    assert result["e1_intervention"] is False
    assert "'results/new run'" in result["linux_command"]
    assert subprocess.run(argv, cwd=ROOT, capture_output=True).returncode != 0
