"""H100 handoff: download, infer, compare, or run one shared LIBERO trial.

Run port and upstream modes in separate processes to avoid GPU memory overlap.
"""

import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("fixture", "download", "port", "upstream", "compare", "libero"), required=True)
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/fasterwam_release/libero/step_021700.pt"))
    parser.add_argument("--dataset-stats", type=Path, default=Path("checkpoints/fasterwam_release/libero/dataset_stats.json"))
    parser.add_argument("--revision", help="Hugging Face release revision; download resolves main to a commit when omitted")
    parser.add_argument("--observation", type=Path, help="Prepared NPZ: image [3,224,448], normalized proprio [8]")
    parser.add_argument("--prompt", default="pick up the object")
    parser.add_argument("--condition", choices=("one_pass_future_cache", "joint_rollout"), default="one_pass_future_cache")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--upstream-root", type=Path)
    parser.add_argument("--output", type=Path, default=Path("results/fasterwam"))
    parser.add_argument("--rtol", type=float, default=1e-4)
    parser.add_argument("--atol", type=float, default=1e-5)
    parser.add_argument("--suite", default="libero_spatial")
    parser.add_argument("--task-id", type=int, default=0)
    args = parser.parse_args(argv)
    args.output.mkdir(parents=True, exist_ok=True)

    if args.mode == "fixture":
        import numpy as np
        generator = np.random.default_rng(args.seed)
        destination = args.observation or args.output / "observation.npz"
        destination.parent.mkdir(parents=True, exist_ok=True)
        np.savez(destination, image=generator.integers(0, 256, (3, 224, 448), dtype=np.uint8),
                 proprio=np.zeros(8, dtype=np.float32))
        print(f"Saved synthetic parity fixture to {destination}; this is not a benchmark observation")
        return

    if args.mode == "download":
        from huggingface_hub import HfApi, hf_hub_download
        revision = HfApi().model_info("hustvl/FasterWAM", revision=args.revision).sha
        for filename, destination in (("libero/step_021700.pt", args.checkpoint),
                                      ("libero/dataset_stats.json", args.dataset_stats)):
            downloaded = Path(hf_hub_download("hustvl/FasterWAM", filename, revision=revision,
                                              local_dir=args.checkpoint.parent.parent))
            if downloaded.resolve() != destination.resolve():
                raise ValueError("Download uses the official libero/ layout; keep checkpoint and stats paths in that layout")
        report = {"repository": "hustvl/FasterWAM", "revision": revision,
                  "checkpoint_sha256": sha256(args.checkpoint), "stats_sha256": sha256(args.dataset_stats)}
        (args.output / "download.json").write_text(json.dumps(report, indent=2))
        print(json.dumps(report, indent=2))
        return

    import torch
    if args.mode == "compare":
        port = torch.load(args.output / "port.pt", map_location="cpu", weights_only=True)
        upstream = torch.load(args.output / "upstream.pt", map_location="cpu", weights_only=True)
        if port["inputs"] != upstream["inputs"]:
            raise ValueError("Port and upstream runs have different inputs/settings/provenance")
        a, b = port["actions"], upstream["actions"]
        finite = bool(torch.isfinite(a).all() and torch.isfinite(b).all())
        same_shape = a.shape == b.shape
        passed = finite and same_shape and a.shape == (32, 7) and torch.allclose(a, b, rtol=args.rtol, atol=args.atol)
        report = {"passed": bool(passed), "finite": finite, "port_shape": list(a.shape),
                  "upstream_shape": list(b.shape), "rtol": args.rtol, "atol": args.atol,
                  "max_abs_diff": float((a-b).abs().max()) if same_shape else None,
                  "mean_abs_diff": float((a-b).abs().mean()) if same_shape else None}
        (args.output / "comparison.json").write_text(json.dumps(report, indent=2))
        print(json.dumps(report, indent=2))
        if not passed:
            raise SystemExit(1)
        return

    if args.mode in {"port", "upstream"} and (not args.observation or not args.observation.is_file()):
        parser.error("An existing --observation NPZ is required; use --mode fixture for numerical parity")
    if not args.checkpoint.is_file():
        parser.error(f"Checkpoint missing: {args.checkpoint}; run --mode download first")
    if str(args.device).startswith("cuda") and not torch.cuda.is_available():
        parser.error("CUDA unavailable")
    from omegaconf import OmegaConf, DictConfig
    from wam_interp.models.fasterwam.loader import compose_fasterwam_config, load_fasterwam, UPSTREAM_REVISION
    cfg = compose_fasterwam_config()
    dtype = getattr(torch, args.dtype)
    if args.mode == "upstream":
        if not args.upstream_root:
            parser.error("--upstream-root is required for upstream comparison")
        actual = subprocess.check_output(["git", "-C", str(args.upstream_root), "rev-parse", "HEAD"], text=True).strip()
        if actual != UPSTREAM_REVISION:
            raise ValueError(f"Upstream revision must be {UPSTREAM_REVISION}; found {actual}")
        sys.path.insert(0, str(args.upstream_root.resolve() / "src"))
        # Execute just the original inference factory, avoiding its unrelated
        # module-level trainer/dataset imports. Its body is unmodified.
        tree = ast.parse((args.upstream_root / "src/fasterwam/runtime.py").read_text(encoding="utf-8"))
        factory = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "create_fasterwam")
        namespace = {"__name__": "fasterwam._validation_factory", "__package__": "fasterwam",
                     "torch": torch, "DictConfig": DictConfig, "OmegaConf": OmegaConf}
        exec(compile(ast.Module(body=[factory], type_ignores=[]), "upstream/runtime.py", "exec"), namespace)
        native = namespace["create_fasterwam"](**{k: v for k, v in cfg.model.items() if not k.startswith("_")},
                                               device=args.device, model_dtype=dtype)
        native.load_checkpoint(str(args.checkpoint.resolve()))
        native.eval()
    else:
        native = load_fasterwam(cfg, args.checkpoint, device=args.device, dtype=args.dtype)

    if args.mode == "libero":
        from wam_interp.models.fasterwam.libero_adapter import FasterWAMLiberoAdapter
        from wam_interp.evaluations.LIBERO import evaluate_libero, LiberoEvalConfig
        adapter = FasterWAMLiberoAdapter(native, data_config=OmegaConf.to_container(cfg.data.train, resolve=True),
                                        dataset_stats=args.dataset_stats)
        evaluate_libero(adapter, LiberoEvalConfig(task_suite_name=args.suite, task_ids=(args.task_id,), num_trials=1,
            seed=args.seed, output_dir=args.output / "libero", camera_resolution=256,
            inference_options={"condition": args.condition, "num_inference_steps": args.steps},
            run_metadata={"checkpoint_sha256": sha256(args.checkpoint), "stats_sha256": sha256(args.dataset_stats),
                          "upstream_revision": UPSTREAM_REVISION}))
        return

    if not args.observation:
        parser.error("--observation is required for port/upstream inference")
    from wam_interp.data.npz import load_observation_npz
    from wam_interp.models.fastwam.adapter import FASTWAM_PROMPT_TEMPLATE
    from wam_interp.models.fasterwam.adapter import FasterWAMAdapter
    from wam_interp.types import InferenceRequest
    obs = load_observation_npz(args.observation, args.prompt)
    inputs = {"checkpoint_sha256": sha256(args.checkpoint), "observation_sha256": sha256(args.observation),
              "prompt": args.prompt, "condition": args.condition, "seed": args.seed, "steps": args.steps,
              "dtype": args.dtype, "device": args.device, "upstream_revision": UPSTREAM_REVISION,
              "config": OmegaConf.to_container(cfg, resolve=True), "torch_version": str(torch.__version__)}
    method = "infer_action_one_pass_future_cache" if args.condition == "one_pass_future_cache" else "infer_action"
    def infer():
        if args.mode == "port":
            return FasterWAMAdapter(native).predict_action(InferenceRequest(obs, condition=args.condition,
                num_video_frames=9, action_horizon=32, num_inference_steps=args.steps, seed=args.seed)).actions
        with torch.inference_mode():
            return getattr(native, method)(prompt=FASTWAM_PROMPT_TEMPLATE.format(task=args.prompt),
                input_image=obs.image.to(device=args.device, dtype=dtype), proprio=obs.proprio,
                num_video_frames=9, action_horizon=32, num_inference_steps=args.steps, seed=args.seed,
                rand_device="cpu", sigma_shift=None, negative_prompt="", text_cfg_scale=1.0, tiled=False)["action"]
    infer()  # Warm-up is recorded separately from steady inference timing.
    cuda = torch.device(args.device).type == "cuda"
    if cuda:
        torch.cuda.synchronize(args.device)
        torch.cuda.reset_peak_memory_stats(args.device)
    start = time.perf_counter()
    actions = infer()
    if cuda:
        torch.cuda.synchronize(args.device)
    elapsed = (time.perf_counter() - start) * 1000
    report = {"inputs": inputs, "shape": list(actions.shape), "dtype": str(actions.dtype),
              "finite": bool(torch.isfinite(actions).all()), "latency_ms": elapsed,
              "peak_memory_bytes": torch.cuda.max_memory_allocated(args.device) if cuda else None}
    torch.save({"actions": actions.detach().cpu(), "inputs": inputs}, args.output / f"{args.mode}.pt")
    (args.output / f"{args.mode}.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    if actions.shape != (32, 7) or not report["finite"]:
        raise SystemExit("Invalid action chunk")


if __name__ == "__main__":
    main()
