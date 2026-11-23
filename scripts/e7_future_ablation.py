"""Roadmap E7 pilot: ablate Faster-WAM's future conditioning in and out of distribution.

ID is the ten original libero_10 tasks, run through the LIBERO-Plus checkout so that both arms share
one simulator build and environment. OOD is LIBERO-Plus libero_10 variants sampled per factor and
difficulty level. Every condition sees the same variants, initial states and seeds.

Run from the repo root in the LIBERO-Plus environment, with LIBERO_CONFIG_PATH set as in
docs/benchmark_handoff.md:

    python scripts/e7_future_ablation.py --arm checks
    python scripts/e7_future_ablation.py --arm id --conditions normal,future_mask
    python scripts/e7_future_ablation.py --arm ood --conditions normal,future_mask
"""

import argparse
import json
import random
from pathlib import Path
from time import perf_counter

import torch
from libero.libero import get_libero_path
from libero.libero.benchmark import Task, grab_language_from_filename
from omegaconf import OmegaConf

from wam_interp.evaluations._libero_rollout import (
    _episode_seed, _json_safe, _max_steps, _request_for_call, run_libero_episode,
)
from wam_interp.evaluations.LIBERO import LiberoEvalConfig
from wam_interp.evaluations.LIBERO.environment import (
    LIBERO_DUMMY_ACTION, create_libero_environment, reset_to_initial_state, step_libero,
)
from wam_interp.evaluations.LIBERO_PLUS import LiberoPlusEvalConfig, evaluate_libero_plus
from wam_interp.evaluations.LIBERO_PLUS.environment import identify_benchmark
from wam_interp.evaluations.provenance import model_provenance, package_versions, report_path, write_report
from wam_interp.interventions.future_conditioning import (
    CONDITIONS, MASKS, FasterWAMFutureAblation, FutureAblationPolicy,
)
from wam_interp.models.fasterwam.libero_plus_adapter import FasterWAMLiberoPlusAdapter
from wam_interp.models.fasterwam.loader import compose_fasterwam_config, load_fasterwam

ORIGINAL_LIBERO_10 = (
    "LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket",
    "LIVING_ROOM_SCENE2_put_both_the_cream_cheese_box_and_the_butter_in_the_basket",
    "KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it",
    "KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it",
    "LIVING_ROOM_SCENE5_put_the_white_mug_on_the_left_plate_and_put_the_yellow_and_white_mug_on_the_right_plate",
    "STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_back_compartment_of_the_caddy",
    "LIVING_ROOM_SCENE6_put_the_white_mug_on_the_plate_and_put_the_chocolate_pudding_to_the_right_of_the_plate",
    "LIVING_ROOM_SCENE1_put_both_the_alphabet_soup_and_the_cream_cheese_box_in_the_basket",
    "KITCHEN_SCENE8_put_both_moka_pots_on_the_stove",
    "KITCHEN_SCENE6_put_the_yellow_and_white_mug_in_the_microwave_and_close_it",
)
CATEGORIES = ("Background Textures", "Camera Viewpoints", "Language Instructions", "Light Conditions",
              "Objects Layout", "Robot Initial States", "Sensor Noise")
# Faster-WAM's LIBERO episode settings, as in scripts/evaluate_benchmark.py.
PROTOCOL = {"action_horizon": 32, "replan_steps": 10, "num_steps_wait": 30, "camera_resolution": 256}
MASK_SPANS = {"current_mask": "latent frame 0", "future_mask": "latent frames 1-2",
              "future1_mask": "latent frame 1", "future2_mask": "latent frame 2"}


def load_adapter(args):
    cfg = compose_fasterwam_config(args.task_config, benchmark="libero_plus")
    model = load_fasterwam(cfg, args.checkpoint, device="cuda", dtype="bfloat16")
    adapter = FasterWAMLiberoPlusAdapter(model, data_config=OmegaConf.to_container(cfg.data.train, resolve=True),
                                         dataset_stats=args.dataset_stats)
    return adapter, (int(cfg.EVALUATION.num_video_frames) - 1) // 4 + 1  # latent frames; the VAE compresses time 4x


def original_tasks():
    """The ten original libero_10 tasks and their fixed initial states, from the LIBERO-Plus checkout."""
    tasks = []
    for name in ORIGINAL_LIBERO_10:
        task = Task(name=name, language=grab_language_from_filename("libero_10", f"{name}.bddl"), problem="Libero",
                    problem_folder="libero_10", bddl_file=f"{name}.bddl", init_states_file=f"{name}.pruned_init")
        path = Path(get_libero_path("init_states")) / "libero_10" / task.init_states_file
        tasks.append((task, torch.load(path, weights_only=False)))
    return tasks


def evaluate_original(policy, config, num_states):
    """Same loop and record format as evaluate_libero_plus, over the original tasks."""
    config.validate()
    path = report_path(config.output_dir, "libero_10_original")
    report = {"benchmark": "libero_10_original_via_libero_plus", "completed": False, "config": _json_safe(config),
              "model": model_provenance(policy), "package_versions": package_versions(), "trials": []}
    start = perf_counter()
    for task_id, (task, states) in enumerate(original_tasks()):
        env = create_libero_environment(task, config.camera_resolution, _episode_seed(config.seed, task_id, 0))
        try:
            for trial in range(num_states):
                result = run_libero_episode(
                    model=policy, env=env, task=task, task_id=task_id, trial_index=trial, initial_state_index=trial,
                    initial_state=states[trial], config=config, seed=_episode_seed(config.seed, task_id, trial),
                    max_steps=_max_steps(config), reset_fn=reset_to_initial_state, step_fn=step_libero,
                    environment="libero_plus")
                report["trials"].append({"task_id": task_id, "name": task.name, **_json_safe(result)})
                report["duration_seconds"] = perf_counter() - start
                write_report(report, path)
        finally:
            env.close()
    report["completed"] = True
    report["successes"] = sum(trial["success"] for trial in report["trials"])
    report["episodes"] = len(report["trials"])
    write_report(report, path)
    return report


def sample_variants(args):
    """Variant indices per "category:level" cell, the same for every condition."""
    suite, records, _, _ = identify_benchmark(LiberoPlusEvalConfig(task_suite_name="libero_10"))
    index = {suite.get_task(i).name: i for i in range(int(suite.n_tasks))}
    cells = {}
    for category in CATEGORIES:
        for level in args.levels:
            names = sorted(r["name"] for r in records if r["category"] == category and r["difficulty_level"] == level)
            picked = random.Random(f"{args.sample_seed}:{category}:{level}").sample(names, args.variants_per_cell)
            cells[f"{category}:{level}"] = sorted(index[name] for name in picked)
    return cells


def run_checks(adapter, latent_frames, args):
    """Patched normal and same-noise resample reproduce the unpatched model; masks leave the video cache alone."""
    task, states = original_tasks()[0]
    env = create_libero_environment(task, PROTOCOL["camera_resolution"], seed=0)
    observation = reset_to_initial_state(env, states[0], instruction=task.language, metadata={})
    for _ in range(PROTOCOL["num_steps_wait"]):
        observation = step_libero(env, LIBERO_DUMMY_ACTION, instruction=task.language, metadata={}).observation
    env.close()
    request = _request_for_call(LiberoEvalConfig(task_suite_name="libero_10", **PROTOCOL), adapter,
                                adapter.prepare_observation(observation), 7, 0)
    reference = adapter.predict_action(request).actions

    ablation = FasterWAMFutureAblation(adapter.model, latent_frames)
    policy = FutureAblationPolicy(adapter, ablation, "normal", probes=("normal", *MASKS, "resample"),
                                  executed_steps=PROTOCOL["replan_steps"], resample_offset=0)
    checks = {"patched_normal_bitwise": torch.equal(reference, policy.run(request, "normal").actions)}
    normal_cache = ablation.last_cache
    checks["same_noise_resample_bitwise"] = torch.equal(reference, policy.run(request, "resample").actions)
    policy.run(request, "future_mask")
    checks["mask_leaves_video_cache"] = all(
        a is None and b is None or torch.equal(a["k"], b["k"]) and torch.equal(a["v"], b["v"])
        for a, b in zip(normal_cache, ablation.last_cache))
    passed = all(checks.values())
    policy.resample_offset = args.resample_offset
    checks["action_abs_diff"] = policy.predict_action(request).metadata["future_ablation"]["action_abs_diff"]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "checks.json").write_text(json.dumps(checks, indent=2))
    print("CHECKS", json.dumps(checks), flush=True)
    if not passed:
        raise SystemExit("checks failed")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--arm", choices=("checks", "id", "ood"), required=True)
    parser.add_argument("--conditions", default="normal", help=f"comma-separated, from {', '.join(CONDITIONS)}")
    parser.add_argument("--probes", default="normal,future_mask,current_mask,future1_mask,future2_mask,resample",
                        help="conditions rerun at every call of the normal runs to measure action change")
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/fasterwam/libero/step_021700.pt"))
    parser.add_argument("--checkpoint-revision", default="6bf9471ced6919a15ab8fded89f7772f5060c44b")
    parser.add_argument("--dataset-stats", type=Path, default=Path("checkpoints/fasterwam/libero/dataset_stats.json"))
    parser.add_argument("--task-config", default="libero_fasterwam_2cam224_1e-4")
    parser.add_argument("--output-dir", type=Path, default=Path("results/e7"))
    parser.add_argument("--id-states", type=int, default=5, help="initial states per original task")
    parser.add_argument("--levels", type=lambda text: tuple(int(x) for x in text.split(",")), default=(2, 4))
    parser.add_argument("--variants-per-cell", type=int, default=10)
    parser.add_argument("--sample-seed", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resample-offset", type=int, default=10_000)
    args = parser.parse_args(argv)
    conditions = args.conditions.split(",")
    probes = tuple(args.probes.split(",")) if args.probes else ()

    adapter, latent_frames = load_adapter(args)
    if args.arm == "checks":
        run_checks(adapter, latent_frames, args)
        return
    ablation = FasterWAMFutureAblation(adapter.model, latent_frames)
    cells = sample_variants(args) if args.arm == "ood" else None
    for condition in conditions:
        policy = FutureAblationPolicy(adapter, ablation, condition, probes=probes if condition == "normal" else (),
                                      executed_steps=PROTOCOL["replan_steps"], resample_offset=args.resample_offset)
        metadata = {"experiment": "E7 future-conditioning ablation", "arm": args.arm, "condition": condition,
                    "probes": list(policy.probes), "mask_spans": MASK_SPANS, "resample_offset": args.resample_offset,
                    "checkpoint_revision": args.checkpoint_revision, "fasterwam_task_config": args.task_config,
                    "id_states": args.id_states if args.arm == "id" else None, "ood_cells": cells,
                    "sample_seed": args.sample_seed if cells else None}
        shared = {"seed": args.seed, "output_dir": args.output_dir / args.arm / condition, "inference_options": {},
                  "run_metadata": metadata, **PROTOCOL}
        if args.arm == "id":
            report = evaluate_original(policy, LiberoEvalConfig(task_suite_name="libero_10", **shared), args.id_states)
        else:
            task_ids = tuple(sorted(i for ids in cells.values() for i in ids))
            report = evaluate_libero_plus(policy, LiberoPlusEvalConfig(task_suite_name="libero_10", task_ids=task_ids,
                                                                       num_trials=1, **shared))
        successes = sum(trial["success"] for trial in report["trials"])
        print(f"RESULT {args.arm} {condition} {successes}/{len(report['trials'])} {report['output_path']}", flush=True)


if __name__ == "__main__":
    main()
