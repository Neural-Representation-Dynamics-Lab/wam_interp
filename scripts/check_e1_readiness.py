"""Write a readiness report without loading model weights or simulator assets."""
import argparse
from datetime import datetime, timezone
import importlib.metadata
import json
from pathlib import Path
import subprocess
import sys
import xml.etree.ElementTree as ET

from wam_interp.experiments.libero_pilot import validate_config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/experiments/e1_libero.json"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tests", action="store_true", help="Run CPU tests and optional tiny CUDA tests")
    args = parser.parse_args()
    if args.output.exists(): parser.error("Choose a new report path")
    root = Path(__file__).resolve().parents[1]
    config = json.loads(args.config.read_text())
    validate_config(config)
    report = {"timestamp_utc": datetime.now(timezone.utc).isoformat(), "python": sys.version,
        "repository": str(root), "config_valid": True, "versions": {},
        "assets": {k: {"path": config[k], "present": Path(config[k]).is_file()} for k in ("checkpoint", "statistics")},
        "pending": ["Real checkpoint extraction and identity validation", "Real LIBERO replay validation",
                    "Semantic donor review", "E1 executions", "Trajectory annotation and analysis"],
        "full_model_or_simulator_loaded": False}
    for name in ("torch", "numpy", "hydra-core", "mujoco", "robosuite", "transformers", "pytest"):
        try: report["versions"][name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError: report["versions"][name] = None
    report["target_environment"] = {"python": "3.10", "torch": "2.7.1+cu128", "mujoco": "3.3.2",
        "note": "Local test environment may differ; setup_e1_linux.sh creates the target environment."}
    try:
        import torch
        report["cuda"] = {"available": torch.cuda.is_available()}
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            report["cuda"].update(name=props.name, total_bytes=props.total_memory,
                                   bf16_supported=torch.cuda.is_bf16_supported())
    except ImportError:
        report["cuda"] = {"available": False, "reason": "torch is not installed"}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    code = 0
    if args.tests:
        junit = args.output.with_suffix(".junit.xml")
        code = subprocess.run([sys.executable, "-m", "pytest", "tests/test_e1.py", "tests/test_e1_libero.py",
            "tests/test_e1_handoff.py", "tests/test_fasterwam.py", "-q", f"--junitxml={junit.resolve()}"], cwd=root).returncode
        counts = {"tests": 0, "failures": 0, "errors": 0, "skipped": 0}
        if junit.exists():
            for suite in ET.parse(junit).getroot().iter("testsuite"):
                for key in counts: counts[key] += int(suite.get(key, 0))
        report["tests"] = {"exit_code": code, **counts, "junit": str(junit.resolve())}
    report["status"] = "local_checks_passed_gpu_experiments_pending" if args.tests and code == 0 else (
        "local_tests_failed" if code else "inventory_only_tests_not_run")
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    raise SystemExit(code)


if __name__ == "__main__": main()
