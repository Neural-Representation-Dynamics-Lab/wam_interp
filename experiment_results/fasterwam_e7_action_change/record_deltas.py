"""Normal E7 rollouts that store, at every call, how each probe changes the executed steps.

The pilot probes kept only the mean |change|. This keeps the per-step, per-dimension change of the
normalized actions, plus the decoded gripper flips, so a perturbation can be matched to it. Run from
the repository root on feat/e7-partial-mask (c1fb12e) in the LIBERO-Plus environment, with the same
--checkpoint and --dataset-stats as the e7 runs and --arm smoke first.
"""

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, "scripts")
import e7_future_ablation as e7  # noqa: E402
from wam_interp.evaluations._libero_rollout import _request_for_call  # noqa: E402
from wam_interp.evaluations.LIBERO import LiberoEvalConfig  # noqa: E402
from wam_interp.evaluations.LIBERO.environment import (  # noqa: E402
    LIBERO_DUMMY_ACTION, create_libero_environment, reset_to_initial_state, step_libero,
)
from wam_interp.evaluations.LIBERO_PLUS import LiberoPlusEvalConfig, evaluate_libero_plus  # noqa: E402
from wam_interp.interventions.future_conditioning import FasterWAMFutureAblation, FutureAblationPolicy  # noqa: E402

PROBES = ("future1_mask_90", "current_mask_90")


class DeltaRecorder(FutureAblationPolicy):
    def predict_action(self, request, recorder=None):
        result = self.run(request, "normal", recorder)
        steps = self.executed_steps
        base = result.actions[:steps].float()
        gripper = self.adapter.postprocess_actions(result.actions, environment="libero_plus")[:steps, -1]
        record = {"condition": "normal", "base": base.tolist(), "delta": {}, "gripper_flips": {}}
        for probe in self.probes:
            actions = self.run(request, probe).actions
            record["delta"][probe] = (actions[:steps].float() - base).tolist()
            decoded = self.adapter.postprocess_actions(actions, environment="libero_plus")[:steps, -1]
            record["gripper_flips"][probe] = int((decoded != gripper).sum())
        self.ablation.condition = "normal"
        result.metadata = {key: value for key, value in result.metadata.items() if key not in self.run_level_keys}
        result.metadata["future_ablation"] = record
        return result


def smoke(adapter, policy, out):
    """One call: the recorded run must equal the plain model, and the record must have the right shape."""
    task, states = e7.original_tasks()[0]
    env = create_libero_environment(task, e7.PROTOCOL["camera_resolution"], seed=0)
    obs = reset_to_initial_state(env, states[0], instruction=task.language, metadata={})
    for _ in range(e7.PROTOCOL["num_steps_wait"]):
        obs = step_libero(env, LIBERO_DUMMY_ACTION, instruction=task.language, metadata={}).observation
    env.close()
    request = _request_for_call(LiberoEvalConfig(task_suite_name="libero_10", **e7.PROTOCOL), adapter,
                                adapter.prepare_observation(obs), episode_seed=7, inference_call=0)
    policy.ablation.condition = "normal"
    reference = adapter.predict_action(request).actions
    result = policy.predict_action(request)
    record = result.metadata["future_ablation"]
    checks = {"normal_bitwise": torch.equal(reference, result.actions),
              "delta_shape": {p: [len(d), len(d[0])] for p, d in record["delta"].items()},
              "mean_abs": {p: float(torch.tensor(d).abs().mean()) for p, d in record["delta"].items()},
              "gripper_flips": record["gripper_flips"]}
    (out / "smoke.json").write_text(json.dumps(checks, indent=2))
    print("SMOKE", json.dumps(checks), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", choices=("smoke", "id", "ood"), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset-stats", type=Path, required=True)
    args = parser.parse_args()
    args.task_config, args.levels, args.variants_per_cell, args.sample_seed = "libero_fasterwam_2cam224_1e-4", (2, 4), 10, 0
    args.output_dir.mkdir(parents=True, exist_ok=True)

    adapter, latent_frames = e7.load_adapter(args)
    ablation = FasterWAMFutureAblation(adapter.model, latent_frames)
    policy = DeltaRecorder(adapter, ablation, "normal", probes=PROBES, executed_steps=e7.PROTOCOL["replan_steps"])
    if args.arm == "smoke":
        smoke(adapter, policy, args.output_dir)
        return
    cells = e7.sample_variants(args) if args.arm == "ood" else None
    metadata = {"experiment": "E7 executed-step deltas", "arm": args.arm, "condition": "normal",
                "probes": list(PROBES), "partial_subset_seed": ablation.subset_seed, "ood_cells": cells}
    shared = {"seed": 42, "output_dir": args.output_dir / args.arm / "normal_deltas", "inference_options": {},
              "run_metadata": metadata, **e7.PROTOCOL}
    if args.arm == "id":
        report = e7.evaluate_original(policy, LiberoEvalConfig(task_suite_name="libero_10", **shared), 5)
    else:
        task_ids = tuple(sorted(i for ids in cells.values() for i in ids))
        report = evaluate_libero_plus(policy, LiberoPlusEvalConfig(task_suite_name="libero_10", task_ids=task_ids,
                                                                   num_trials=1, **shared))
    successes = sum(trial["success"] for trial in report["trials"])
    print(f"RESULT {args.arm} normal_deltas {successes}/{len(report['trials'])} {report['output_path']}", flush=True)


if __name__ == "__main__":
    main()
