"""Inspect audited E1 capabilities and prepare native baseline commands without a GPU."""

import argparse
import json
from pathlib import Path
import shlex

from wam_interp.experiments.model_capabilities import (
    CAPABILITIES, baseline_argv, capability_report, validate_local_assets,
)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    subs = parser.add_subparsers(dest="command", required=True)
    subs.add_parser("inventory")
    plan = subs.add_parser("baseline", help="Print a plan; does not load weights or execute it")
    plan.add_argument("--variant", choices=[c.variant for c in CAPABILITIES], required=True)
    plan.add_argument("--python", default="python")
    plan.add_argument("--checkpoint", required=True)
    plan.add_argument("--dataset-stats")
    plan.add_argument("--checkpoint-revision")
    plan.add_argument("--benchmark-root", required=True)
    plan.add_argument("--output", required=True, help="New evaluator output directory on the GPU host")
    plan.add_argument("--task-ids", type=int, nargs="+", default=[0])
    plan.add_argument("--episodes", type=int, default=1)
    plan.add_argument("--seed", type=int, default=0)
    plan.add_argument("--check-local-assets", action="store_true")
    plan.add_argument("--save-plan", type=Path)
    args = parser.parse_args(argv)
    if args.command == "inventory":
        result = capability_report()
    else:
        if args.check_local_assets:
            validate_local_assets(args.variant, checkpoint=args.checkpoint,
                                  dataset_stats=args.dataset_stats)
        command = baseline_argv(args.variant, python=args.python, checkpoint=args.checkpoint,
            benchmark_root=args.benchmark_root, output=args.output, dataset_stats=args.dataset_stats,
            task_ids=args.task_ids, episodes=args.episodes, seed=args.seed,
            checkpoint_revision=args.checkpoint_revision)
        result = {"status": "planned_not_executed", "variant": args.variant,
                  "argv": command, "linux_command": shlex.join(command),
                  "working_directory": "repository root on the GPU host",
                  "e1_intervention": False,
                  "note": "Verify imported LIBERO and its configured asset paths on the host first. "
                          "Use an isolated environment for OpenWAM (transformers 5). "
                          "A native episode does not validate a future hook or target-identity edit."}
        if args.save_plan:
            args.save_plan.parent.mkdir(parents=True, exist_ok=True)
            with args.save_plan.open("x", encoding="utf-8") as stream:
                json.dump(result, stream, indent=2, allow_nan=False)
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
