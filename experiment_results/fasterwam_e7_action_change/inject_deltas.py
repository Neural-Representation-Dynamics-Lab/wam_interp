"""Closed-loop control: add a recorded future1_mask_90 change to the normal chunk, without any mask.

Every call adds one executed-step change from the bank written by record_deltas.py (normalized
actions, steps x 7) to the first steps of the normal chunk. By default each call draws its change
independently. With --sequence an episode takes one bank episode that the normal run solved and
replays its changes in call order, cycling if it runs longer, so consecutive calls keep the recorded
carry-over between calls. Draws depend only on the episode's first request seed and the call index,
so reruns are identical. The draw ignores the recipient's state (a same-arm bank includes its own
episode), so the change keeps the size, per-dimension profile and timing of the mask's effect but not
its link to the scene. Run from the repository root on feat/e7-partial-mask (c1fb12e) in the
LIBERO-Plus environment with the e7 checkpoint arguments.
"""

import argparse
import json
import random
import sys
from pathlib import Path

import torch

sys.path.insert(0, "scripts")
import e7_future_ablation as e7  # noqa: E402
from wam_interp.evaluations.LIBERO import LiberoEvalConfig  # noqa: E402
from wam_interp.evaluations.LIBERO_PLUS import LiberoPlusEvalConfig, evaluate_libero_plus  # noqa: E402
from wam_interp.interventions.future_conditioning import FasterWAMFutureAblation, FutureAblationPolicy  # noqa: E402


def load_bank(paths, probe):
    """Per bank episode: whether the normal run solved it, and its changes, calls x steps x 7."""
    episodes = []
    for path in paths:
        for trial in json.loads(Path(path).read_text())["trials"]:
            calls = [call["future_ablation"]["delta"][probe] for call in trial["inference_metadata"]]
            episodes.append((trial["success"], torch.tensor(calls)))
    return episodes


class DeltaInjection(FutureAblationPolicy):
    def __init__(self, adapter, ablation, episodes, sequence=False, **kwargs):
        super().__init__(adapter, ablation, "normal", **kwargs)
        self.flat = torch.cat([deltas for _, deltas in episodes])
        self.solved = [deltas for solved, deltas in episodes if solved]
        self.sequence, self.calls, self.episode_seed, self.source = sequence, 0, None, None

    def begin_episode(self, observation):
        self.calls = 0

    def predict_action(self, request, recorder=None):
        if self.calls == 0:
            self.episode_seed = request.seed
            self.source = random.Random(f"{self.episode_seed}").randrange(len(self.solved))
        if self.sequence:
            deltas = self.solved[self.source]
            draw = {"bank_episode": self.source, "bank_call": self.calls % len(deltas)}
            delta = deltas[draw["bank_call"]]
        else:
            draw = {"bank_index": random.Random(f"{self.episode_seed}:{self.calls}").randrange(len(self.flat))}
            delta = self.flat[draw["bank_index"]]
        self.calls += 1
        result = self.run(request, "normal", recorder)
        result.actions = result.actions.clone()
        result.actions[:len(delta)] += delta.to(result.actions)
        result.metadata = {key: value for key, value in result.metadata.items() if key not in self.run_level_keys}
        result.metadata["future_ablation"] = {"condition": "delta_injection", **draw}
        return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", choices=("id", "ood"), required=True)
    parser.add_argument("--bank", type=Path, nargs="+", required=True, help="record_deltas.py result JSONs")
    parser.add_argument("--name", required=True, help="condition folder, e.g. inject_idbank")
    parser.add_argument("--probe", default="future1_mask_90")
    parser.add_argument("--sequence", action="store_true", help="replay one solved bank episode in call order")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset-stats", type=Path, required=True)
    args = parser.parse_args()
    args.task_config, args.levels, args.variants_per_cell, args.sample_seed = "libero_fasterwam_2cam224_1e-4", (2, 4), 10, 0

    episodes = load_bank(args.bank, args.probe)
    adapter, latent_frames = e7.load_adapter(args)
    ablation = FasterWAMFutureAblation(adapter.model, latent_frames)
    policy = DeltaInjection(adapter, ablation, episodes, sequence=args.sequence,
                            executed_steps=e7.PROTOCOL["replan_steps"])
    cells = e7.sample_variants(args) if args.arm == "ood" else None
    used = torch.cat(policy.solved) if args.sequence else policy.flat
    metadata = {"experiment": "E7 delta injection", "arm": args.arm, "condition": args.name, "probe": args.probe,
                "draw": "sequence" if args.sequence else "independent", "bank": [str(p) for p in args.bank],
                "bank_calls": len(used), "bank_mean_abs": float(used.abs().mean()), "ood_cells": cells}
    shared = {"seed": 42, "output_dir": args.output_dir / args.arm / args.name, "inference_options": {},
              "run_metadata": metadata, **e7.PROTOCOL}
    if args.arm == "id":
        report = e7.evaluate_original(policy, LiberoEvalConfig(task_suite_name="libero_10", **shared), 5)
    else:
        task_ids = tuple(sorted(i for ids in cells.values() for i in ids))
        report = evaluate_libero_plus(policy, LiberoPlusEvalConfig(task_suite_name="libero_10", task_ids=task_ids,
                                                                   num_trials=1, **shared))
    successes = sum(trial["success"] for trial in report["trials"])
    print(f"RESULT {args.arm} {args.name} {successes}/{len(report['trials'])} {report['output_path']}", flush=True)


if __name__ == "__main__":
    main()
