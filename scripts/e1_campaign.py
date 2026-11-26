"""Resume prepare/run stages across explicit cases, one GPU process at a time."""
import argparse
import json
from pathlib import Path
import subprocess
import sys


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage", choices=("prepare", "run"))
    p.add_argument("--config", type=Path, default=Path("configs/experiments/e1_libero.json"))
    p.add_argument("--cases", type=Path, default=Path("configs/experiments/e1_cases.json"))
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    base = json.loads(args.config.read_text())
    cases = json.loads(args.cases.read_text())
    seen = set()
    for case in cases:
        key = (case["task_id"], case["initial_state_index"])
        if key in seen or any(not isinstance(x, int) or x < 0 for x in key):
            raise ValueError("Cases must have unique, nonnegative task and initial-state indices")
        seen.add(key)
    for case in cases:
        folder = args.output / f"task_{case['task_id']}_state_{case['initial_state_index']}"
        folder.mkdir(parents=True, exist_ok=True)
        config = {**base, **case}
        config_path = folder / "config.json"
        if config_path.exists() and json.loads(config_path.read_text()) != config:
            raise ValueError(f"Configuration changed for {folder}; use a fresh campaign")
        config_path.write_text(json.dumps(config, indent=2))
        destination = folder / args.stage
        status = destination / "status.json"
        if status.exists():
            state = json.loads(status.read_text())["status"]
            if state == ("awaiting_semantic_review" if args.stage == "prepare" else "complete"):
                print(f"Already finished: {destination}")
                continue
            raise RuntimeError(f"Incomplete run at {destination}; inspect it and choose a fresh campaign directory")
        command = [sys.executable, str(Path(__file__).with_name("e1_libero.py")), args.stage,
                   "--config", str(config_path), "--output", str(destination)]
        if args.stage == "run": command += ["--prepared", str(folder / "prepare")]
        subprocess.run(command, check=True)


if __name__ == "__main__": main()
