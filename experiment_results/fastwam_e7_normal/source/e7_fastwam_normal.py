"""Roadmap E7: Fast-WAM Optional IDM closed loop on the E7 episodes, with no intervention.

The episodes match the Faster-WAM pilot in e7_future_ablation.py: the ten original libero_10 tasks with
five initial states each for ID, the sampled LIBERO-Plus variants for OOD, both in the LIBERO-Plus
install, with the same protocol and seed. Every call takes the IDM path, which generates the future video
the content arms replace, so the failures here are the starting set for the OOD rescue.

    python scripts/e7_fastwam_normal.py --arm ood --output-dir results/e7_fastwam \\
        --checkpoint checkpoints/fastwam/libero_optional_idm_2cam224.pt \\
        --dataset-stats checkpoints/fastwam/libero_optional_idm_2cam224_dataset_stats.json
"""

import argparse
from pathlib import Path

from omegaconf import OmegaConf

import e7_future_ablation as e7
from wam_interp.evaluations.LIBERO import LiberoEvalConfig
from wam_interp.evaluations.LIBERO_PLUS import LiberoPlusEvalConfig, evaluate_libero_plus
from wam_interp.evaluations.provenance import file_hash
from wam_interp.models.fastwam.libero_plus_adapter import FastWAMLiberoPlusAdapter
from wam_interp.models.fastwam.loader import compose_fastwam_config, load_fastwam

TASK = "libero_optional_idm_2cam224_1e-4"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--arm", choices=("id", "ood"), required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset-stats", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--id-states", type=int, default=5, help="initial states per original task")
    parser.add_argument("--levels", type=lambda text: tuple(int(x) for x in text.split(",")), default=(2, 4))
    parser.add_argument("--variants-per-cell", type=int, default=10)
    parser.add_argument("--sample-seed", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--task-ids", type=int, nargs="+", help="OOD: only these sampled variants")
    args = parser.parse_args()

    cfg = compose_fastwam_config(TASK, benchmark="libero_plus")
    model = load_fastwam(cfg, args.checkpoint, device="cuda", dtype="bfloat16")
    adapter = FastWAMLiberoPlusAdapter(model, data_config=OmegaConf.to_container(cfg.data.train, resolve=True),
                                       dataset_stats=args.dataset_stats)
    cells = e7.sample_variants(args) if args.arm == "ood" else None
    metadata = {"experiment": "E7 Fast-WAM normal", "arm": args.arm, "condition": "idm", "fastwam_task": TASK,
                "checkpoint": str(args.checkpoint), "checkpoint_sha256": file_hash(args.checkpoint),
                "dataset_stats_sha256": file_hash(args.dataset_stats),
                "id_states": args.id_states if args.arm == "id" else None, "ood_cells": cells,
                "sample_seed": args.sample_seed if cells else None}
    shared = {"seed": args.seed, "output_dir": args.output_dir / args.arm / "normal",
              "inference_options": {"condition": "idm"}, "run_metadata": metadata, **e7.PROTOCOL}
    if args.arm == "id":
        report = e7.evaluate_original(adapter, LiberoEvalConfig(task_suite_name="libero_10", **shared), args.id_states)
    else:
        sampled = sorted(i for ids in cells.values() for i in ids)
        if args.task_ids and not set(args.task_ids) <= set(sampled):
            parser.error("--task-ids must come from the sampled variants")
        task_ids = tuple(args.task_ids or sampled)
        report = evaluate_libero_plus(adapter, LiberoPlusEvalConfig(task_suite_name="libero_10", task_ids=task_ids,
                                                                    num_trials=1, **shared))
    successes = sum(trial["success"] for trial in report["trials"])
    print(f"RESULT {args.arm} normal {successes}/{len(report['trials'])} {report['output_path']}", flush=True)


if __name__ == "__main__":
    main()
