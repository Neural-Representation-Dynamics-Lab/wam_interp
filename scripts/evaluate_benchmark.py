"""Single-process Linux GPU evaluation runs for LIBERO, LIBERO-Plus and RoboTwin.

Model orchestration belongs here; the evaluators consume already-loaded adapters.
"""

import argparse
from pathlib import Path


def robotwin_protocol(backend, seed):
    """Per-backend RoboTwin (replan_steps, start_seed) from each upstream eval; None means RoboTwin's 100000*(1+seed)."""
    replan = {"fastwam": 24, "fasterwam": 28, "lingbot_va": 32, "longwam": 24, "openwam": 32}[backend]
    return replan, (10000 * (1 + int(seed)) if backend == "lingbot_va" else None)


def libero_protocol(backend, suite):
    """Per-backend LIBERO/LIBERO-Plus episode settings; max_steps None uses the evaluator default (700 or 400)."""
    if backend == "lingbot_va":
        return {"action_horizon": 16, "replan_steps": 16, "num_steps_wait": 5, "camera_resolution": 128,
                "max_steps": 796}
    return {"action_horizon": 32, "replan_steps": 10, "num_steps_wait": 30, "camera_resolution": 256,
            "max_steps": 600 if backend == "openwam" and suite != "libero_10" else None}


FASTWAM_TASKS = {"libero": ("libero_uncond_2cam224_1e-4", "libero_optional_idm_2cam224_1e-4"),
                 "robotwin": ("robotwin_uncond_3cam_384_1e-4",)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", choices=("libero", "libero_plus", "robotwin"), required=True)
    parser.add_argument("--backend", choices=("fastwam", "fasterwam", "lingbot_va", "longwam", "openwam"), required=True)
    parser.add_argument("--fastwam-task", help="FastWAM: released task config (default: the uncond checkpoint's; "
                        "libero_optional_idm_2cam224_1e-4 for Optional IDM)")
    parser.add_argument("--model-config", type=Path,
                        help="Long-WAM: bundle config.yaml (default: config.yaml next to --checkpoint)")
    parser.add_argument("--openwam-wrist", choices=("real", "black"), default="real",
                        help="OpenWAM LIBERO: real wrist tile or black (see docs/openwam.md, PR #40)")
    parser.add_argument("--openwam-upstream-optimizations", action="store_true",
                        help="OpenWAM: DiT cache and torch.compile on, as in upstream's published numbers "
                             "(default off: exact forwards, hooks work)")
    parser.add_argument("--checkpoint", type=Path, required=True, help="PT file or LingBot checkpoint directory")
    parser.add_argument("--checkpoint-revision", help="Revision used to download this checkpoint")
    parser.add_argument("--dataset-stats", type=Path, help="Matching FastWAM/Faster-WAM statistics JSON")
    parser.add_argument("--benchmark-root", type=Path, required=True)
    parser.add_argument("--suite", default="libero_10")
    parser.add_argument("--task-id", type=int, default=0)
    parser.add_argument("--task-ids", type=int, nargs="+", help="LIBERO/LIBERO-Plus: several task ids (overrides --task-id)")
    parser.add_argument("--task-name", default="beat_block_hammer")
    parser.add_argument("--task-names", nargs="+",
                        help="RoboTwin: evaluate several tasks with one loaded model (overrides --task-name)")
    parser.add_argument("--task-config", choices=("demo_clean", "demo_randomized"), default="demo_clean")
    parser.add_argument("--instruction-type", choices=("seen", "unseen"))
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--categories", nargs="*", default=[])
    parser.add_argument("--difficulty-levels", nargs="*", type=int, default=[])
    parser.add_argument("--all-tasks", action="store_true",
                        help="LIBERO: every task in the suite; LIBERO-Plus: all variants before filtering")
    parser.add_argument("--max-seed-attempts", type=int, default=1000)
    parser.add_argument("--replan-steps", type=int,
                        help="RoboTwin: override the backend's published replan interval")
    parser.add_argument("--start-seed", type=int,
                        help="RoboTwin: override the backend's published first seed")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=("bfloat16", "float32", "float16"), default="bfloat16")
    parser.add_argument("--condition")
    parser.add_argument("--output", type=Path, default=Path("results/benchmarks"))
    args = parser.parse_args(argv)
    from wam_interp.evaluations.provenance import install_termination_handler
    install_termination_handler()  # Slurm SIGTERM -> SystemExit, so partial reports are saved
    args.checkpoint = args.checkpoint.expanduser().resolve()
    if not args.checkpoint.exists():
        parser.error(f"Missing checkpoint: {args.checkpoint}")
    if args.backend in ("fastwam", "fasterwam", "longwam") and (args.dataset_stats is None or not args.dataset_stats.is_file()):
        parser.error("FastWAM/Faster-WAM/Long-WAM require --dataset-stats pointing to a matching JSON file")

    import torch
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
    libero = args.benchmark in ("libero", "libero_plus")  # LIBERO checkpoints, configs and transforms
    family = "libero" if libero else "robotwin"
    if args.fastwam_task is not None and (args.backend != "fastwam" or args.fastwam_task not in FASTWAM_TASKS[family]):
        parser.error(f"--fastwam-task is for --backend fastwam; choose from {FASTWAM_TASKS[family]}")
    lingbot = args.backend == "lingbot_va"
    openwam = args.backend == "openwam"
    if args.backend == "longwam":
        from wam_interp.models.longwam.loader import load_longwam
        model_config = (args.model_config or args.checkpoint.parent / "config.yaml").expanduser().resolve()
        if not model_config.is_file():
            parser.error(f"Missing Long-WAM bundle config: {model_config}")
        model = load_longwam(model_config, args.checkpoint, device=args.device, dtype=args.dtype)
        metadata.update(model_config=str(model_config), model_config_sha256=file_hash(model_config),
                        resolved_model_config=model._wam_interp_metadata["resolved_model_config"])
        if plus:
            from wam_interp.models.longwam.libero_plus_adapter import LongWAMLiberoPlusAdapter as Adapter
        elif libero:
            from wam_interp.models.longwam.libero_adapter import LongWAMLiberoAdapter as Adapter
        else:
            from wam_interp.models.longwam.robotwin_adapter import LongWAMRobotWinAdapter as Adapter
        adapter = Adapter(model, dataset_stats=args.dataset_stats)
    elif openwam:
        from wam_interp.models.openwam.loader import load_openwam
        model = load_openwam(args.checkpoint, device=args.device, dtype=args.dtype,
                             upstream_optimizations=args.openwam_upstream_optimizations)
        metadata["openwam_upstream_optimizations"] = bool(args.openwam_upstream_optimizations)
        metadata["resolved_model_config"] = model._wam_interp_metadata["config"]
        if plus:
            from wam_interp.models.openwam.libero_plus_adapter import OpenWAMLiberoPlusAdapter
            adapter = OpenWAMLiberoPlusAdapter(model, wrist=args.openwam_wrist)
        elif libero:
            from wam_interp.models.openwam.libero_adapter import OpenWAMLiberoAdapter
            adapter = OpenWAMLiberoAdapter(model, wrist=args.openwam_wrist)
        else:
            from wam_interp.models.openwam.robotwin_adapter import OpenWAMRobotWinAdapter
            adapter = OpenWAMRobotWinAdapter(model)
    elif lingbot:
        from wam_interp.models.lingbot_va.loader import load_lingbot_va
        model = load_lingbot_va(checkpoint_dir=args.checkpoint, config_name=args.benchmark,
                               device=args.device, dtype=args.dtype)
        metadata["resolved_model_config"] = dict(model.config)
        if plus:
            from wam_interp.models.lingbot_va.libero_plus_adapter import LingBotVALiberoPlusAdapter
            adapter = LingBotVALiberoPlusAdapter(model)
        elif libero:
            from wam_interp.models.lingbot_va.libero_adapter import LingBotVALiberoAdapter
            adapter = LingBotVALiberoAdapter(model)
        else:
            from wam_interp.models.lingbot_va.robotwin_adapter import LingBotVARobotWinAdapter
            adapter = LingBotVARobotWinAdapter(model)
    else:
        if args.backend == "fastwam":
            from wam_interp.models.fastwam.loader import compose_fastwam_config, load_fastwam
            task = args.fastwam_task or FASTWAM_TASKS[family][0]
            metadata["fastwam_task"] = task
            cfg = compose_fastwam_config(task, benchmark=args.benchmark)
            model = load_fastwam(cfg, args.checkpoint, device=args.device, dtype=args.dtype)
            if plus:
                from wam_interp.models.fastwam.libero_plus_adapter import FastWAMLiberoPlusAdapter as Adapter
            elif libero:
                from wam_interp.models.fastwam.libero_adapter import FastWAMLiberoAdapter as Adapter
            else:
                from wam_interp.models.fastwam.robotwin_adapter import FastWAMRobotWinAdapter as Adapter
        else:
            from wam_interp.models.fasterwam.loader import compose_fasterwam_config, load_fasterwam
            task = "libero_fasterwam_2cam224_1e-4" if libero else "robotwin_fasterwam_3cam_384_1e-4"
            cfg = compose_fasterwam_config(task, benchmark=args.benchmark)
            model = load_fasterwam(cfg, args.checkpoint, device=args.device, dtype=args.dtype)
            if plus:
                from wam_interp.models.fasterwam.libero_plus_adapter import FasterWAMLiberoPlusAdapter as Adapter
            elif libero:
                from wam_interp.models.fasterwam.libero_adapter import FasterWAMLiberoAdapter as Adapter
            else:
                from wam_interp.models.fasterwam.robotwin_adapter import FasterWAMRobotWinAdapter as Adapter
        from omegaconf import OmegaConf  # FastWAM-family only; the LingBot-VA/OpenWAM extras do not ship it

        metadata["resolved_model_config"] = OmegaConf.to_container(cfg.model, resolve=True)
        adapter = Adapter(model, data_config=OmegaConf.to_container(cfg.data.train, resolve=True),
                          dataset_stats=args.dataset_stats)
    options = {} if args.condition is None else {"condition": args.condition}
    if args.benchmark == "libero":
        from wam_interp.evaluations.LIBERO import evaluate_libero, LiberoEvalConfig
        metadata["benchmark_root"] = str(args.benchmark_root.resolve())
        result = evaluate_libero(adapter, LiberoEvalConfig(
            task_suite_name=args.suite, task_ids=None if args.all_tasks else tuple(args.task_ids or [args.task_id]),
            num_trials=args.episodes, **libero_protocol(args.backend, args.suite), inference_options=options,
            run_metadata=metadata, seed=args.seed, output_dir=args.output.resolve()))
        print({"benchmark": "libero", "suite": args.suite, "episodes": result.episodes, "successes": result.successes,
               "success_rate": result.success_rate, "per_task": [task.successes for task in result.tasks],
               "output_path": result.output_path}, flush=True)
        return
    elif plus:
        from wam_interp.evaluations.LIBERO_PLUS import evaluate_libero_plus, LiberoPlusEvalConfig
        result = evaluate_libero_plus(adapter, LiberoPlusEvalConfig(
            benchmark_root=args.benchmark_root.resolve(), task_suite_name=args.suite,
            task_ids=None if args.all_tasks else tuple(args.task_ids or [args.task_id]), num_trials=args.episodes,
            categories=tuple(args.categories), difficulty_levels=tuple(args.difficulty_levels),
            **libero_protocol(args.backend, args.suite), inference_options=options,
            run_metadata=metadata, seed=args.seed, output_dir=args.output.resolve()))
    else:
        from wam_interp.evaluations.ROBOTWIN import evaluate_robotwin, RoboTwinEvalConfig
        replan, start_seed = robotwin_protocol(args.backend, args.seed)
        failed = []
        for task_name in args.task_names or [args.task_name]:
            try:
                result = evaluate_robotwin(adapter, RoboTwinEvalConfig(
                    task_name=task_name, robotwin_root=args.benchmark_root.resolve(),
                    task_config=args.task_config, num_episodes=args.episodes,
                    instruction_type=args.instruction_type or ("seen" if lingbot else "unseen"),
                    action_horizon=32, replan_steps=args.replan_steps or replan,
                    start_seed=args.start_seed if args.start_seed is not None else start_seed,
                    inference_options=options, run_metadata=metadata, seed=args.seed,
                    max_seed_attempts=args.max_seed_attempts, output_dir=args.output.resolve()))
            except Exception as exc:  # the per-task report (with the error) is already saved
                failed.append(task_name)
                print({"task_name": task_name, "error": f"{type(exc).__name__}: {exc}"}, flush=True)
                continue
            print({"task_name": task_name, **{key: result[key] for key in
                   ("benchmark", "episodes", "successes", "success_rate", "output_path")}}, flush=True)
        if failed:
            raise SystemExit(f"RoboTwin tasks with errors: {failed}")
        return
    print({key: result[key] for key in ("benchmark", "episodes", "successes", "success_rate", "output_path")})


if __name__ == "__main__":
    main()
