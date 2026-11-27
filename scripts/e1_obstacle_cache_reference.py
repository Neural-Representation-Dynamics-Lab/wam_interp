"""Render a cache backend's own 32-control LIBERO reference, preserving native chunk boundaries."""

import argparse
import json
from pathlib import Path

import numpy as np
from e1_obstacle_cache import commands, load, make_request
from wam_interp.evaluations.LIBERO.environment import (
    initial_states_for_task,
    step_libero,
)
from wam_interp.evaluations.LIBERO.runner import _get_suite
from wam_interp.experiments.clip_study import start


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--backend", required=True, choices=["lingbot_va", "openwam"])
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    api = load(a.backend)
    suite = _get_suite("libero_10")
    m = json.loads(a.manifest.read_text())
    for tid in m["tasks"]:
        task = suite.get_task(tid)
        states = initial_states_for_task(suite, tid)
        for idx in m["states"][str(tid)]:
            for seed in m["seeds"]:
                dest = a.out / f"task_{tid}/state_{idx}/seed_{seed}"
                dest.mkdir(parents=True, exist_ok=True)
                out = dest / f"{a.backend}_reference.npy"
                if out.is_file():
                    continue
                with start(None, task, states[idx], idx, task_id=tid) as (
                    env,
                    meta,
                    obs,
                    _,
                ):
                    from dataclasses import replace

                    req = make_request(api, obs, seed)
                    chunks = []
                    steps = 0
                    while steps < 32:
                        plan = commands(api, api.adapter.predict_action(req))[
                            : 32 - steps
                        ]
                        if not len(plan):
                            raise RuntimeError("Empty native action chunk")
                        for cmd in plan:
                            result = step_libero(env, cmd, **meta)
                            if result.done or result.success:
                                raise RuntimeError(
                                    "Reference terminated before 32 controls"
                                )
                            obs = result.observation
                            if hasattr(api.adapter, "observe_step"):
                                api.adapter.observe_step(obs, cmd)
                        chunks.append(plan)
                        steps += len(plan)
                        req = replace(
                            req,
                            observation=api.adapter.prepare_observation(obs),
                            seed=seed + len(chunks),
                        )
                    np.save(out, np.concatenate(chunks))
                    print(out, flush=True)


if __name__ == "__main__":
    main()
