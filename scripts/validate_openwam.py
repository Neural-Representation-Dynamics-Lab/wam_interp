"""GPU parity check: our OpenWAM adapter vs upstream's client and engine on one saved observation.

Modes: fixture, download, port, upstream, compare (run on a GPU node).
"""

import argparse
import hashlib
import json
import resource
import subprocess
import sys
import time
from pathlib import Path

REPOS = {"libero": "OpenWAM/OpenWAM-Alpha-Sim-LIBERO", "robotwin": "OpenWAM/OpenWAM-Alpha-Sim-RoboTwin-Full"}
LIBERO_KEYS = ("agentview", "wrist", "eef_position", "eef_quaternion", "gripper_position")
ROBOTWIN_KEYS = ("head", "left_wrist", "right_wrist", "left_endpose", "right_endpose", "left_gripper", "right_gripper")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_observation(path):
    import numpy as np

    with np.load(path, allow_pickle=False) as data:
        values = {key: np.asarray(data[key]) for key in data.files}
    return values, str(values.pop("instruction"))


def attention_backends():
    """Fused kernels are picked from installed packages; both runs must match for bitwise parity."""
    import importlib.util
    import os

    packages = ("flash_attn_interface", "flash_attn", "sageattention", "xformers")
    return {"installed": {name: importlib.util.find_spec(name) is not None for name in packages},
            "env": {name: os.environ.get(name) for name in ("DIFFSYNTH_ATTENTION_IMPLEMENTATION", "WAM_ATTENTION_IMPL")}}


def resources():
    import torch

    unit = 1 if sys.platform == "darwin" else 1024  # ru_maxrss: bytes on macOS, KiB on Linux
    peak_rss_gb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * unit / 1024 ** 3
    vram = torch.cuda.max_memory_allocated() / 1024 ** 3 if torch.cuda.is_available() else None
    return {"peak_rss_gb": round(peak_rss_gb, 2), "peak_vram_allocated_gb": None if vram is None else round(vram, 2)}


def run_port(args, values, instruction):
    import numpy as np
    from wam_interp.models.openwam.loader import load_openwam
    from wam_interp.types import EnvironmentObservation, InferenceRequest

    start = time.perf_counter()
    model = load_openwam(args.checkpoint_dir, device=args.device, dtype=args.dtype)
    load_seconds = time.perf_counter() - start
    if args.benchmark == "libero":
        from wam_interp.models.openwam.libero_adapter import OpenWAMLiberoAdapter
        adapter = OpenWAMLiberoAdapter(model, wrist=args.wrist)
        observation = EnvironmentObservation(
            cameras={"agentview": values["agentview"], "wrist": values["wrist"]},
            state={key: values[key] for key in ("eef_position", "eef_quaternion", "gripper_position")},
            instruction=instruction)
    else:
        from wam_interp.models.openwam.robotwin_adapter import OpenWAMRobotWinAdapter
        adapter = OpenWAMRobotWinAdapter(model)
        observation = EnvironmentObservation(
            cameras={key: values[key] for key in ("head", "left_wrist", "right_wrist")},
            state={key: values[key] for key in ("left_endpose", "right_endpose", "left_gripper", "right_gripper")},
            instruction=instruction)
    prepared = adapter.prepare_observation(observation)
    request = InferenceRequest(prepared, **{**adapter.inference_defaults(), "seed": args.seed})
    timings = []
    for _ in range(args.repeats):
        tick = time.perf_counter()
        result = adapter.predict_action(request)
        timings.append(time.perf_counter() - tick)
    environment = "libero" if args.benchmark == "libero" else "robotwin"
    from wam_interp.models.openwam.adapter import image_to_pil
    return {"composite": np.asarray(image_to_pil(prepared.image)), "prompt": prepared.prompt,
            "raw_state": np.asarray(prepared.metadata["raw_state_eef10" if environment == "libero" else "raw_state_eef20"],
                                    np.float32),
            "raw_actions": adapter.denormalize(result.actions),
            "commands": adapter.postprocess_actions(result.actions, environment=environment),
            "load_seconds": load_seconds, "predict_seconds": timings,
            "provenance": {"backend": "wam_interp", "upstream_revision": model._wam_interp_metadata["upstream_revision"],
                           "deploy_overrides": model._wam_interp_metadata["deploy_overrides"],
                           "checkpoint_load": model._wam_interp_metadata["checkpoint_load"]}}


def run_upstream(args, values, instruction):
    import numpy as np
    from omegaconf import OmegaConf

    root = args.upstream_root.resolve()
    actual = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
    from wam_interp.models.openwam.loader import UPSTREAM_REVISION, research_config_overrides
    if actual != UPSTREAM_REVISION:
        raise SystemExit(f"Upstream checkout must be {UPSTREAM_REVISION}; found {actual}")
    sys.path.insert(0, str(root))
    from benchmarks.utils.client import build_payload, encode_numpy_b64, resize_for_lshape_slot
    from benchmarks.utils import action_conversion as ac
    from openwam.deploy.engine import JointInferenceEngine
    from openwam.deploy.model_loader import load_from_checkpoint_dir
    from openwam.deploy.obs_preprocess import ObsPreprocessor
    from openwam.deploy.policy import WAMPolicy
    from openwam.deploy.server import merge_deploy_cfg

    if args.benchmark == "libero":
        from benchmarks.libero.openwam2libero_interface import native_eef10_to_libero7d

        def rotate(image):  # OpenWAMLiberoPolicy._image with image_transform=rotate_180
            return np.asarray(image).astype(np.uint8, copy=False)[::-1, ::-1]
        wrist = encode_numpy_b64(resize_for_lshape_slot(rotate(values["wrist"]), "left_wrist_camera"))
        payload = build_payload(head=encode_numpy_b64(resize_for_lshape_slot(rotate(values["agentview"]), "head_camera")),
                                left_wrist=wrist if args.wrist == "real" else None, right_wrist=None,
                                prompt=instruction,
                                state=ac.libero_obs_to_eef10(values["eef_position"], values["eef_quaternion"],
                                                             values["gripper_position"]).tolist())
        convert = native_eef10_to_libero7d
    else:
        from benchmarks.robotwin.prompt_template import format_prompt_for_inference
        payload = build_payload(head=encode_numpy_b64(values["head"]), left_wrist=encode_numpy_b64(values["left_wrist"]),
                                right_wrist=encode_numpy_b64(values["right_wrist"]),
                                prompt=format_prompt_for_inference(instruction),
                                state=ac.robotwin_endpose_to_eef20d(values["left_endpose"], values["right_endpose"],
                                                                    values["left_gripper"], values["right_gripper"]).tolist())
        convert = ac.eef20d_to_ee16d
    start = time.perf_counter()
    training_cfg, architecture = load_from_checkpoint_dir(str(args.checkpoint_dir), device=args.device)
    load_seconds = time.perf_counter() - start
    deploy = OmegaConf.load(root / "configs/deploy.yaml") if args.upstream_defaults else \
        OmegaConf.create(research_config_overrides())
    cfg = merge_deploy_cfg(training_cfg, deploy)
    engine = JointInferenceEngine(cfg=cfg, architecture=architecture)
    obs = ObsPreprocessor.from_cfg(cfg, engine).preprocess(dict(payload))
    conditions = WAMPolicy._build_conditions(None, obs)
    conditions["seed"] = args.seed  # upstream's server omits it, so the engine falls back to 42
    timings = []
    for _ in range(args.repeats):
        tick = time.perf_counter()
        actions = np.asarray(engine.generate(dict(conditions))["actions"], np.float32)
        timings.append(time.perf_counter() - tick)
    return {"composite": np.asarray(obs["image"]), "prompt": obs["prompt"], "raw_state": obs["state"],
            "raw_actions": actions, "commands": np.stack([convert(step) for step in actions]),
            "load_seconds": load_seconds, "predict_seconds": timings,
            "provenance": {"backend": "upstream", "upstream_revision": actual,
                           "deploy": OmegaConf.to_container(deploy), "upstream_defaults": args.upstream_defaults}}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mode", choices=("fixture", "download", "port", "upstream", "compare"), required=True)
    parser.add_argument("--benchmark", choices=("libero", "robotwin"), default="libero")
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("checkpoints/openwam/OpenWAM-Alpha-Sim-LIBERO"))
    parser.add_argument("--repo-id", help="Hub repo for --mode download (default: the Alpha checkpoint of --benchmark)")
    parser.add_argument("--revision", help="Hub revision; resolved to a commit and recorded")
    parser.add_argument("--observation", type=Path)
    parser.add_argument("--instruction", default="put the black bowl on the plate")
    parser.add_argument("--wrist", choices=("real", "black"), default="real", help="LIBERO wrist tile")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--repeats", type=int, default=2, help="Predictions per run (timing; outputs must agree)")
    parser.add_argument("--dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--upstream-root", type=Path, help="Upstream OpenWAM checkout at the pinned revision")
    parser.add_argument("--upstream-defaults", action="store_true",
                        help="Upstream run uses configs/deploy.yaml (dit_cache + compile on) to measure their effect")
    parser.add_argument("--output", type=Path, default=Path("results/openwam_validation"))
    parser.add_argument("--atol", type=float, default=0.0, help="Max abs difference allowed on raw actions")
    args = parser.parse_args(argv)
    args.output.mkdir(parents=True, exist_ok=True)
    import numpy as np

    if args.mode == "fixture":
        rng = np.random.default_rng(args.seed)
        destination = args.observation or args.output / f"{args.benchmark}_observation.npz"
        quat = lambda: (lambda q: q / np.linalg.norm(q))(rng.normal(size=4))  # noqa: E731
        if args.benchmark == "libero":
            arrays = {"agentview": rng.integers(0, 256, (256, 256, 3), dtype=np.uint8),
                      "wrist": rng.integers(0, 256, (256, 256, 3), dtype=np.uint8),
                      "eef_position": np.array([-0.05, 0.03, 1.1]), "eef_quaternion": quat(),
                      "gripper_position": np.array([0.039, -0.039])}
        else:
            arrays = {"head": rng.integers(0, 256, (240, 320, 3), dtype=np.uint8),
                      "left_wrist": rng.integers(0, 256, (240, 320, 3), dtype=np.uint8),
                      "right_wrist": rng.integers(0, 256, (240, 320, 3), dtype=np.uint8),
                      "left_endpose": np.concatenate([[-0.3, -0.3, 0.9], quat()]),
                      "right_endpose": np.concatenate([[0.3, -0.3, 0.9], quat()]),
                      "left_gripper": np.array(1.0), "right_gripper": np.array(1.0)}
        np.savez(destination, instruction=np.array(args.instruction), **arrays)
        print(f"Saved synthetic {args.benchmark} observation to {destination} (numerical parity only)")
        return

    if args.mode == "download":
        from huggingface_hub import HfApi, snapshot_download
        repo = args.repo_id or REPOS[args.benchmark]
        revision = HfApi().model_info(repo, revision=args.revision).sha
        path = Path(snapshot_download(repo, revision=revision, local_dir=args.checkpoint_dir))
        weights = sorted(path.glob("checkpoint_step_*.safetensors"))
        report = {"repo_id": repo, "revision": revision, "local_dir": str(path),
                  "files": {p.name: sha256(p) for p in [*weights, path / "config.yaml", path / "normalization_stats.npy"]}}
        (args.output / f"download_{repo.split('/')[-1]}.json").write_text(json.dumps(report, indent=2))
        print(json.dumps(report, indent=2))
        return

    if args.mode == "compare":
        port = np.load(args.output / "port.npz", allow_pickle=False)
        upstream = np.load(args.output / "upstream.npz", allow_pickle=False)
        inputs_equal = {key: bool(np.array_equal(port[key], upstream[key])) for key in ("composite", "raw_state")}
        inputs_equal["prompt"] = str(port["prompt"]) == str(upstream["prompt"])
        report = {"inputs_equal": inputs_equal}
        for key in ("raw_actions", "commands"):
            a, b = port[key], upstream[key]
            report[key] = {"shape": [list(a.shape), list(b.shape)], "bitwise_equal": bool(np.array_equal(a, b)),
                           "max_abs_diff": float(np.abs(a - b).max()) if a.shape == b.shape else None}
        report["passed"] = all(inputs_equal.values()) and report["raw_actions"]["max_abs_diff"] is not None \
            and report["raw_actions"]["max_abs_diff"] <= args.atol
        for side in ("port", "upstream"):
            report[side] = json.loads((args.output / f"{side}.json").read_text())
        (args.output / "comparison.json").write_text(json.dumps(report, indent=2))
        print(json.dumps(report, indent=2))
        raise SystemExit(0 if report["passed"] else 1)

    import torch
    if not args.observation or not args.observation.is_file():
        parser.error("--observation NPZ is required; create one with --mode fixture")
    if str(args.device).startswith("cuda") and not torch.cuda.is_available():
        parser.error("CUDA unavailable")
    if args.mode == "upstream" and not args.upstream_root:
        parser.error("--upstream-root is required")
    values, instruction = load_observation(args.observation)
    keys = LIBERO_KEYS if args.benchmark == "libero" else ROBOTWIN_KEYS
    if set(keys) - set(values):
        parser.error(f"Observation lacks {sorted(set(keys) - set(values))} for {args.benchmark}")
    outcome = (run_port if args.mode == "port" else run_upstream)(args, values, instruction)
    runs = outcome["raw_actions"], outcome["commands"]
    if not all(np.isfinite(run).all() for run in runs):
        raise SystemExit("Nonfinite actions")
    np.savez(args.output / f"{args.mode}.npz", composite=outcome["composite"], prompt=np.array(outcome["prompt"]),
             raw_state=outcome["raw_state"], raw_actions=outcome["raw_actions"], commands=outcome["commands"])
    report = {"observation": str(args.observation.resolve()), "observation_sha256": sha256(args.observation),
              "checkpoint_dir": str(args.checkpoint_dir.resolve()), "benchmark": args.benchmark, "wrist": args.wrist,
              "seed": args.seed, "dtype": args.dtype, "device": torch.cuda.get_device_name() if torch.cuda.is_available()
              else args.device, "load_seconds": outcome["load_seconds"], "predict_seconds": outcome["predict_seconds"],
              "attention_backends": attention_backends(), **resources(), **outcome["provenance"]}
    (args.output / f"{args.mode}.json").write_text(json.dumps(report, indent=2, default=str))
    print(json.dumps(report, indent=2, default=str))


if __name__ == "__main__":
    main()
