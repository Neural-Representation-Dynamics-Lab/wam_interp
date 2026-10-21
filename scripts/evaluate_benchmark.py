"""Single-process Linux/H100 smoke runs for LIBERO-Plus and RoboTwin.

Model orchestration belongs here; the evaluators consume already-loaded adapters.
"""

import argparse
from pathlib import Path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", choices=("libero_plus", "robotwin"), required=True)
    parser.add_argument("--backend", choices=("fastwam", "fasterwam", "lingbot_va"), required=True)
    parser.add_argument("--checkpoint", type=Path, required=True, help="PT file or LingBot checkpoint directory")
    parser.add_argument("--checkpoint-revision", help="Revision used to download this checkpoint")
    parser.add_argument("--dataset-stats", type=Path, help="Matching FastWAM/Faster-WAM statistics JSON")
    parser.add_argument("--benchmark-root", type=Path, required=True)
    parser.add_argument("--suite", default="libero_10")
    parser.add_argument("--task-id", type=int, default=0)
    parser.add_argument("--task-name", default="beat_block_hammer")
    parser.add_argument("--task-config", choices=("demo_clean", "demo_randomized"), default="demo_clean")
    parser.add_argument("--instruction-type", choices=("seen", "unseen"))
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--categories", nargs="*", default=[])
    parser.add_argument("--difficulty-levels", nargs="*", type=int, default=[])
    parser.add_argument("--all-tasks", action="store_true", help="LIBERO-Plus: select all variants before filtering")
    parser.add_argument("--max-seed-attempts", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=("bfloat16", "float32", "float16"), default="bfloat16")
    parser.add_argument("--condition")
    parser.add_argument("--output", type=Path, default=Path("results/benchmarks"))
    args = parser.parse_args(argv)
    args.checkpoint = args.checkpoint.expanduser().resolve()
    if not args.checkpoint.exists():
        parser.error(f"Missing checkpoint: {args.checkpoint}")
    if args.backend != "lingbot_va" and (args.dataset_stats is None or not args.dataset_stats.is_file()):
        parser.error("FastWAM/Faster-WAM require --dataset-stats pointing to a matching JSON file")

    import torch
    from omegaconf import OmegaConf
    from wam_interp.evaluations.provenance import file_hash, checkout_revision
    if torch.device(args.device).type == "cuda":
        if not torch.cuda.is_available():
            parser.error("CUDA is unavailable; use a Linux GPU server for real checkpoint validation")
        if args.dtype == "bfloat16":
            with torch.cuda.device(args.device):
                if not torch.cuda.is_bf16_supported():
                    parser.error("This GPU does not support bf16; no precision fallback was applied")

    metadata = {"checkpoint": str(args.checkpoint), "checkpoint_revision": args.checkpoint_revision,
                "dtype": args.dtype, "repository_revision": checkout_revision(Path(__file__).resolve().parents[1])}
    if args.checkpoint.is_file():
        metadata["checkpoint_sha256"] = file_hash(args.checkpoint)
    if args.dataset_stats is not None:
        metadata.update(dataset_stats=str(args.dataset_stats.resolve()), dataset_stats_sha256=file_hash(args.dataset_stats))
    plus = args.benchmark == "libero_plus"
    lingbot = args.backend == "lingbot_va"
    if lingbot:
        from wam_interp.models.lingbot_va.loader import load_lingbot_va
        model = load_lingbot_va(checkpoint_dir=args.checkpoint, config_name=args.benchmark,
                               device=args.device, dtype=args.dtype)
        metadata["resolved_model_config"] = dict(model.config)
        if plus:
            from wam_interp.models.lingbot_va.libero_plus_adapter import LingBotVALiberoPlusAdapter
            adapter = LingBotVALiberoPlusAdapter(model)
        else:
            from wam_interp.models.lingbot_va.robotwin_adapter import LingBotVARobotWinAdapter
            adapter = LingBotVARobotWinAdapter(model)
    else:
        if args.backend == "fastwam":
            from wam_interp.models.fastwam.loader import compose_fastwam_config, load_fastwam
            task = "libero_uncond_2cam224_1e-4" if plus else "robotwin_uncond_3cam_384_1e-4"
            cfg = compose_fastwam_config(task, benchmark=args.benchmark)
            model = load_fastwam(cfg, args.checkpoint, device=args.device, dtype=args.dtype)
            if plus:
                from wam_interp.models.fastwam.libero_plus_adapter import FastWAMLiberoPlusAdapter as Adapter
            else:
                from wam_interp.models.fastwam.robotwin_adapter import FastWAMRobotWinAdapter as Adapter
        else:
            from wam_interp.models.fasterwam.loader import compose_fasterwam_config, load_fasterwam
            task = "libero_fasterwam_2cam224_1e-4" if plus else "robotwin_fasterwam_3cam_384_1e-4"
            cfg = compose_fasterwam_config(task, benchmark=args.benchmark)
            model = load_fasterwam(cfg, args.checkpoint, device=args.device, dtype=args.dtype)
            if plus:
                from wam_interp.models.fasterwam.libero_plus_adapter import FasterWAMLiberoPlusAdapter as Adapter
            else:
                from wam_interp.models.fasterwam.robotwin_adapter import FasterWAMRobotWinAdapter as Adapter
        metadata["resolved_model_config"] = OmegaConf.to_container(cfg, resolve=True)
        adapter = Adapter(model, data_config=OmegaConf.to_container(cfg.data.train, resolve=True),
                          dataset_stats=args.dataset_stats)
    options = {} if args.condition is None else {"condition": args.condition}
    if plus:
        from wam_interp.evaluations.LIBERO_PLUS import evaluate_libero_plus, LiberoPlusEvalConfig
        result = evaluate_libero_plus(adapter, LiberoPlusEvalConfig(
            benchmark_root=args.benchmark_root.resolve(), task_suite_name=args.suite,
            task_ids=None if args.all_tasks else (args.task_id,), num_trials=args.episodes,
            categories=tuple(args.categories), difficulty_levels=tuple(args.difficulty_levels),
            action_horizon=16 if lingbot else 32, replan_steps=16 if lingbot else 10,
            num_steps_wait=5 if lingbot else 30, camera_resolution=128 if lingbot else 256,
            max_steps=796 if lingbot else None, inference_options=options,
            run_metadata=metadata, seed=args.seed, output_dir=args.output.resolve()))
    else:
        from wam_interp.evaluations.ROBOTWIN import evaluate_robotwin, RoboTwinEvalConfig
        result = evaluate_robotwin(adapter, RoboTwinEvalConfig(
            task_name=args.task_name, robotwin_root=args.benchmark_root.resolve(),
            task_config=args.task_config, num_episodes=args.episodes,
            instruction_type=args.instruction_type or ("seen" if lingbot else "unseen"),
            action_horizon=32, replan_steps=32 if lingbot else 28,
            inference_options=options, run_metadata=metadata, seed=args.seed,
            max_seed_attempts=args.max_seed_attempts, output_dir=args.output.resolve()))
    print({key: result[key] for key in ("benchmark", "episodes", "successes", "success_rate", "output_path")})


if __name__ == "__main__":
    main()
