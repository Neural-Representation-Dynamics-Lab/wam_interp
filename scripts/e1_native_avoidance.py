"""Test native closed-loop FastWAM steering around two fixed LIBERO barrier layouts."""

import argparse
import hashlib
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
from e1_confirm import commands, load
from wam_interp.evaluations.LIBERO.environment import initial_states_for_task
from wam_interp.evaluations.LIBERO.runner import _get_suite
from wam_interp.experiments.clip_study import save
from wam_interp.experiments.confirm_replay import anchor_hash
from wam_interp.experiments.obstacle import Barrier
from wam_interp.experiments.obstacle_libero import rollout, scene


def run_episode(api, task, state, case, seed, placement, design, replay=None):
    record = case["placements"].get(placement)
    box = Barrier(**record["geometry"]) if record else None
    expected = record["anchor"] if record else case["clear_anchor"]
    horizon = design["max_controls"] if replay is None else len(replay)
    path, actions, contacts, decisions = [], [], [], []
    success = done = False
    with scene(
        api if replay is None else None,
        task,
        state,
        case["state"],
        case["task"],
        seed,
        box,
    ) as (env, meta, obs, request):
        if anchor_hash(obs) != expected:
            raise ValueError(f"Unexpected starting observation for {placement}")
        path.append(obs.state["eef_position"].copy())

        def observe(step, action):
            nonlocal obs, success, done
            obs, success, done = step.observation, bool(step.success), bool(step.done)
            if replay is None and hasattr(api.adapter, "observe_step"):
                api.adapter.observe_step(obs, action)

        while len(actions) < horizon and not (done or success):
            start = len(actions)
            if replay is None:
                if start:
                    request = replace(
                        request,
                        observation=api.adapter.prepare_observation(obs),
                        seed=seed + len(decisions),
                    )
                decisions.append(start)
                sequence = commands(api, api.adapter.predict_action(request))
            else:
                sequence = replay[start:]
            chunk = sequence[: min(design["replan_every"], horizon - start)]
            segment, _, _, hits = rollout(
                env,
                meta,
                obs,
                chunk,
                barrier=box is not None,
                robot_only=box is not None,
                on_step=observe,
            )
            path.extend(segment[1:])
            actions.extend(chunk[: len(segment) - 1])
            contacts.extend(start + k for k in hits)
    return {
        "path": np.asarray(path).tolist(),
        "commands": np.asarray(actions).tolist(),
        "robot_barrier_contacts": contacts,
        "decisions": decisions,
        "success": success,
        "done": done,
    }


def outcome(record, box):
    path = np.asarray(record["path"])
    forward = (path[:, :2] - box.center[:2]) @ box.forward
    return {
        "reached_bar_plane": bool(forward.max() >= 0),
        "forward_progress_past_plane_mm": float(1000 * forward.max()),
        "min_point_clearance_mm": float(1000 * box.clearance(path).min()),
        "robot_contacts": len(record["robot_barrier_contacts"]),
    }


def witness_actions(actions, box, bias):
    # Steer to the open side during the approach, then continue moving forward.
    adjusted = np.asarray(actions).copy()
    adjusted[:20, :2] = np.clip(
        adjusted[:20, :2] - bias * box.side * box.lateral, -1, 1
    )
    return adjusted


def compare(clear, live, off, replay, off_replay, witnesses, box, design):
    baseline, barrier = outcome(clear, box), outcome(live, box)
    delta = barrier["min_point_clearance_mm"] - baseline["min_point_clearance_mm"]
    a, b = np.asarray(clear["path"]), np.asarray(off["path"])
    rms = None
    if len(a) == len(b):
        rms = float(1000 * np.sqrt(np.square(a - b).sum(axis=1).mean()))
    feasible = bool(
        replay["robot_barrier_contacts"]
        and not off_replay["robot_barrier_contacts"]
        and any(w["reached_bar_plane"] and w["robot_contacts"] == 0 for w in witnesses)
    )
    passed = bool(
        feasible
        and barrier["reached_bar_plane"]
        and barrier["robot_contacts"] == 0
        and delta >= design["min_clearance_gain_mm"]
        and rms is not None
        and rms <= design["max_offroute_rms_mm"]
    )
    return {
        "fixture_pass": feasible,
        "native_pass": passed,
        "native": barrier,
        "clear": baseline,
        "clear_commands_hit_on_route": bool(replay["robot_barrier_contacts"]),
        "clear_commands_hit_off_route": bool(off_replay["robot_barrier_contacts"]),
        "witnesses": witnesses,
        "point_clearance_gain_mm": delta,
        "off_route_rms_mm": rms,
        "first_native_contact_control": next(
            iter(live["robot_barrier_contacts"]), None
        ),
        "native_replan_controls": live["decisions"],
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--design",
        type=Path,
        default=Path("configs/experiments/e1_native_avoidance.json"),
    )
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    spec = json.loads(args.design.read_bytes())
    args.out.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256(args.design.read_bytes()).hexdigest()
    manifest = args.out / "design.sha256"
    if manifest.exists() and manifest.read_text().strip() != digest:
        raise ValueError("This directory contains a different pilot design")
    manifest.write_text(digest + "\n")
    suite = _get_suite("libero_10")
    api = load("fastwam")
    summary = {"design_sha256": digest, "cases": [], "positive_control": False}
    for case in spec["cases"]:
        task = suite.get_task(case["task"])
        state = initial_states_for_task(suite, case["task"])[case["state"]]
        result = {"task": case["task"], "state": case["state"], "comparisons": []}
        for seed in spec["policy_seeds"]:
            folder = args.out / f"task_{case['task']}_state_{case['state']}_seed_{seed}"
            folder.mkdir(exist_ok=True)

            def execute(name, placement, replay=None):
                receipt = folder / (name + ".json")
                if not receipt.exists():
                    values = run_episode(
                        api, task, state, case, seed, placement, spec, replay
                    )
                    save(receipt, values)
                return json.loads(receipt.read_bytes())

            clear = execute("clear", "clear")
            for side in ("minus", "plus"):
                box = Barrier(**case["placements"][side]["geometry"])
                off = "off_" + side
                original = np.asarray(clear["commands"])
                replay = execute("replay_" + side, side, original)
                off_replay = execute("replay_" + off, off, original)
                witnesses = []
                for strength in spec["witness_biases"]:
                    plan = witness_actions(original, box, strength)
                    witness = execute(f"witness_{side}_{strength}", side, plan)
                    trial = outcome(witness, box)
                    witnesses.append(dict(bias=strength, **trial))
                    if trial["reached_bar_plane"] and not trial["robot_contacts"]:
                        break
                live = execute("native_" + side, side)
                shifted = execute("native_" + off, off)
                row = compare(
                    clear, live, shifted, replay, off_replay, witnesses, box, spec
                )
                result["comparisons"].append(dict(seed=seed, side=side, **row))
                print(
                    case["task"],
                    seed,
                    side,
                    row["fixture_pass"],
                    row["native_pass"],
                    flush=True,
                )
        result["passed"] = all(r["native_pass"] for r in result["comparisons"])
        summary["cases"].append(result)
        summary["positive_control"] = result["passed"]
        save(args.out / "summary.json", summary)
        if result["passed"]:
            break


if __name__ == "__main__":
    main()
