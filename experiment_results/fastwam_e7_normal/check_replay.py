"""Replay recorded rollouts and compare every simulator state with the recording.

    python check_replay.py <report.json> [<report.json> ...]

For each trial of the reports, rebuilds its environment, resets to the same initial state, steps the
recorded actions and prints the largest state difference and whether check_success() agrees with the
recorded outcome. Run in the LIBERO-Plus environment with scripts/ on the path.
"""

import json
import sys
from pathlib import Path

import numpy as np

import e7_future_ablation as e7
from wam_interp.evaluations._libero_rollout import _episode_seed
from wam_interp.evaluations.LIBERO.environment import create_libero_environment
from wam_interp.evaluations.LIBERO_PLUS import LiberoPlusEvalConfig
from wam_interp.evaluations.LIBERO_PLUS.environment import identify_benchmark, initial_states_for_variant


def episodes(report):
    base_seed = report["config"]["seed"]
    rollouts = Path(report["output_path"]).parent / "rollouts"
    if report["benchmark"] == "libero_plus":
        suite = identify_benchmark(LiberoPlusEvalConfig(task_suite_name="libero_10"))[0]
        for t in report["trials"]:
            states = initial_states_for_variant(suite, t["task_id"])
            yield t, suite.get_task(t["task_id"]), states[t["initial_state_index"] % len(states)], base_seed, rollouts
    else:
        tasks = e7.original_tasks()
        for t in report["trials"]:
            task, states = tasks[t["task_id"]]
            yield t, task, states[t["initial_state_index"]], base_seed, rollouts


def replay(trial, task, initial_state, base_seed, rollouts):
    recorded = np.load(rollouts / f"seed{base_seed}_{trial['task_id']}_{trial['trial_index']}.npz")
    actions, states = recorded["actions"], recorded["states"]
    if len(actions) != trial["warmup_steps"] + trial["policy_steps"]:
        raise SystemExit(f"{trial['task_id']}: {len(actions)} actions for "
                         f"{trial['warmup_steps']} + {trial['policy_steps']} steps")
    env = create_libero_environment(task, e7.PROTOCOL["camera_resolution"],
                                    _episode_seed(base_seed, trial["task_id"], 0))
    try:
        env.seed(trial["seed"])
        env.reset()
        env.set_init_state(initial_state)
        worst = np.abs(env.get_sim_state() - states[0]).max()
        for action, state in zip(actions, states[1:]):
            env.step(action)
            worst = max(worst, np.abs(env.get_sim_state() - state).max())
        return worst, bool(env.check_success())
    finally:
        env.close()


def main():
    rows = []
    for path in sys.argv[1:]:
        for args in episodes(json.loads(Path(path).read_text())):
            worst, success = replay(*args)
            trial = args[0]
            rows.append((trial["task_id"], trial["trial_index"], worst, success == trial["success"]))
            print(f"{trial['task_id']:5d} {trial['trial_index']} max |state diff| {worst:.3g} "
                  f"success {success} recorded {trial['success']}", flush=True)
    print(f"{len(rows)} replayed, largest diff {max(r[2] for r in rows):.3g}, "
          f"outcome agrees {sum(r[3] for r in rows)}/{len(rows)}")


if __name__ == "__main__":
    main()
