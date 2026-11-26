"""Calibrate first; execute the paired development cases that retain the native unedited reach."""

import argparse
import json
from pathlib import Path
from time import monotonic, sleep

from e1_confirm import load
from e1_obstacle import behavior
from e1_obstacle_calibrate import calibrate
from wam_interp.evaluations.LIBERO.environment import initial_states_for_task
from wam_interp.evaluations.LIBERO.runner import _get_suite


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--backend", choices=["fastwam", "longwam"], required=True)
    p.add_argument("--wait-sources", action="store_true")
    a = p.parse_args()
    m = json.loads(a.manifest.read_text())
    if m["split"] != "development":
        raise ValueError("Development only")
    suite = _get_suite("libero_10")
    api = None
    for t in m["tasks"]:
        task = suite.get_task(t)
        states = initial_states_for_task(suite, t)
        for i in m["states"][str(t)]:
            for s in m["seeds"]:
                dest = a.out / a.backend / f"task_{t}/state_{i}/seed_{s}"
                marker = dest / "sources.json"
                deadline = monotonic() + 1800
                while (
                    a.wait_sources and not marker.is_file() and monotonic() < deadline
                ):
                    sleep(3)
                meta = json.loads(marker.read_text())
                if not meta["passed"]:
                    continue
                if api is None:
                    api = load(a.backend)
                if not (dest / "calibration.json").is_file():
                    calibrate(api, task, states[i], dest, meta)
                cal = json.loads((dest / "calibration.json").read_text())
                if cal["eligible"] and not (dest / "behavior.json").is_file():
                    behavior(api, task, t, states[i], i, s, dest)
                print(a.backend, t, i, s, cal, flush=True)


if __name__ == "__main__":
    main()
