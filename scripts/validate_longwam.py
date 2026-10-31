"""GPU parity check: upstream Long-WAM runtime vs this repo's adapter on the same episode.

Compares model inputs and actions at every inference step (run on a GPU node).
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

SCRIPT = Path(__file__).resolve()
REPOSITORY = SCRIPT.parents[1]
DEFAULT_ANCHORS = {"libero": (0, 10, 60), "robotwin": (0, 24, 72)}
DEFAULT_SEED = {"libero": 1, "robotwin": 0}  # upstream configs/eval/{libero,robotwin2}.yaml
LIBERO_KEYS = ("agentview_image", "robot0_eye_in_hand_image", "robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos")
ROBOTWIN_KEYS = ("head_camera", "left_camera", "right_camera", "joint_vector")



def _sha256(path):
    import hashlib

    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _download(args, spec):
    from huggingface_hub import hf_hub_download
    sys.path.insert(0, str(REPOSITORY.parent))
    from wam_interp.models.longwam.loader import BASE_ASSETS

    bundle_dir = args.cache_dir / "bundles" / args.bundle
    for name in ("config.yaml", "dataset_stats.json", "model.pt"):
        path = Path(hf_hub_download(spec["repo_id"], name, revision=spec["revision"], local_dir=bundle_dir))
        if name == "model.pt" and args.skip_model_sha256:
            continue
        actual = _sha256(path)
        if actual != spec["sha256"][name]:
            raise SystemExit(f"{name}: SHA-256 {actual} != pinned {spec['sha256'][name]}")
    models = args.cache_dir / "models"
    for repo_id, asset in BASE_ASSETS.items():
        for name in asset["files"]:
            hf_hub_download(repo_id, name, revision=asset["revision"], local_dir=models / repo_id)
    return bundle_dir, models


def _synthetic_observations(benchmark, count, seed):
    import numpy as np

    rng = np.random.default_rng(seed)

    def frames(height, width, phase):
        yy, xx = np.mgrid[0:height, 0:width]
        out = []
        for t in range(count):
            base = 127 + 80 * np.sin((xx + 3 * t + phase) / 17.0)[..., None] * np.cos((yy - 2 * t) / 23.0)[..., None]
            noise = rng.normal(0, 12, (height, width, 3))
            out.append(np.clip(base + noise + np.array([phase, -phase, 0.5 * phase]), 0, 255).astype(np.uint8))
        return np.stack(out)

    steps = np.arange(count)[:, None]
    if benchmark == "libero":
        angle = np.pi - 0.2 + 0.005 * steps  # rotation about x; unit quaternion (x, y, z, w)
        return {"agentview_image": frames(256, 256, 0), "robot0_eye_in_hand_image": frames(256, 256, 40),
                "robot0_eef_pos": np.concatenate([0.05 + 0.002 * steps, -0.1 + 0 * steps, 1.0 - 0.003 * steps], 1),
                "robot0_eef_quat": np.concatenate([np.sin(angle / 2), 0 * steps, 0 * steps, np.cos(angle / 2)], 1),
                "robot0_gripper_qpos": np.concatenate([0.03 - 0.0002 * steps, -0.03 + 0.0002 * steps], 1),
                "instruction": np.array("put both the alphabet soup and the tomato sauce in the basket")}
    joints = np.linspace(-0.5, 0.5, 14)[None] + 0.01 * np.sin(steps / 5.0 + np.arange(14)[None])
    return {"head_camera": frames(240, 320, 0), "left_camera": frames(240, 320, 30), "right_camera": frames(240, 320, 60),
            "joint_vector": joints, "instruction": np.array("Lift the pot by its handles with both arms.")}


def _compare(upstream, ours, atol):
    import torch

    failed = False
    rows = []
    for theirs, mine in zip(upstream["calls"], ours["calls"]):
        row = {"anchor": theirs["anchor"]}
        for key in ("input_image", "obs_window", "proprio", "context", "context_mask"):
            a, b = theirs[key], mine[key]
            same = (a is None and b is None) or (a is not None and b is not None and a.shape == b.shape
                                                  and a.dtype == b.dtype and torch.equal(a, b))
            row[key] = "equal" if same else "DIFFERENT"
            failed |= not same
        for key in ("seed", "num_inference_steps", "action_horizon", "sigma_shift", "rand_device"):
            if theirs[key] != mine[key]:
                row[key] = f"DIFFERENT {theirs[key]} vs {mine[key]}"
                failed = True
        for key in ("action", "commands"):
            diff = float((torch.as_tensor(theirs[key]).float() - torch.as_tensor(mine[key]).float()).abs().max())
            row[f"max_abs_diff_{key}"] = diff
            failed |= diff > atol
        rows.append(row)
    if len(upstream["calls"]) != len(ours["calls"]):
        failed = True
    baseline = {}
    for name, result in (("upstream", upstream), ("ours", ours)):
        first, repeat = result["calls"][-2], result["calls"][-1]
        baseline[name] = float((first["action"].float() - repeat["action"].float()).abs().max())
    return rows, baseline, failed


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--role", choices=("compare", "upstream", "ours"), default="compare", help=argparse.SUPPRESS)
    parser.add_argument("--bundle", required=True, choices=("libero_idm", "libero_cod", "robotwin2_idm", "robotwin2_cod"))
    parser.add_argument("--upstream-root", type=Path, required=True, help="Long-WAM directory of an NVlabs/LongLive checkout")
    parser.add_argument("--upstream-python", default=sys.executable,
                        help="Python of upstream's own environment (pip install -e '.[infra]' in Long-WAM)")
    parser.add_argument("--ours-python", default=sys.executable)
    parser.add_argument("--cache-dir", type=Path, required=True, help="Bundle + Wan asset cache (~26 GB)")
    parser.add_argument("--work-dir", type=Path, required=True, help="Observations, captures and the report")
    parser.add_argument("--observations", type=Path, help="NPZ of raw simulator observations")
    parser.add_argument("--anchors", type=int, nargs="+", help="Control steps at which to infer")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--num-inference-steps", type=int, help="Default: bundle eval_num_inference_steps")
    parser.add_argument("--action-horizon", type=int, help="Default: bundle data.train.action_chunk")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--atol", type=float, default=0.0, help="Max allowed |diff| for normalized and decoded actions")
    parser.add_argument("--skip-model-sha256", action="store_true")
    args = parser.parse_args(argv)
    args.work_dir = args.work_dir.expanduser().resolve()
    args.cache_dir = args.cache_dir.expanduser().resolve()
    args.upstream_root = args.upstream_root.expanduser().resolve()
    if args.role == "upstream":
        return _run_upstream(args)
    if args.role == "ours":
        return _run_ours(args)

    sys.path.insert(0, str(REPOSITORY.parent))
    from wam_interp.models.longwam.loader import RELEASED_BUNDLES, UPSTREAM_REVISION

    spec = RELEASED_BUNDLES[args.bundle]
    revision = subprocess.run(["git", "-C", str(args.upstream_root), "rev-parse", "HEAD"], capture_output=True,
                              text=True, check=True).stdout.strip()
    if revision != UPSTREAM_REVISION:
        raise SystemExit(f"Upstream checkout is at {revision}, expected {UPSTREAM_REVISION}")
    args.work_dir.mkdir(parents=True, exist_ok=True)
    bundle_dir, models = _download(args, spec)
    anchors = tuple(args.anchors or DEFAULT_ANCHORS[spec["benchmark"]])
    import numpy as np
    if args.observations is None:
        observations = _synthetic_observations(spec["benchmark"], max(anchors) + 1, seed=0)
    else:
        with np.load(args.observations, allow_pickle=False) as data:
            observations = {key: data[key] for key in data.files}
    keys = LIBERO_KEYS if spec["benchmark"] == "libero" else ROBOTWIN_KEYS
    missing = [key for key in (*keys, "instruction") if key not in observations]
    if missing or len(observations[keys[0]]) <= max(anchors):
        raise SystemExit(f"Observations need {missing or keys} with more than {max(anchors)} steps")
    np.savez(args.work_dir / "observations.npz", **observations)

    import yaml
    run = yaml.safe_load((bundle_dir / "config.yaml").read_text())
    settings = {"bundle": args.bundle, "benchmark": spec["benchmark"], "condition": spec["condition"],
                "bundle_dir": str(bundle_dir), "anchors": list(anchors),
                "seed": DEFAULT_SEED[spec["benchmark"]] if args.seed is None else args.seed,
                "num_inference_steps": args.num_inference_steps or int(run["eval_num_inference_steps"]),
                "action_horizon": args.action_horizon or int(run["data"]["train"]["action_chunk"]),
                "device": args.device}
    (args.work_dir / "settings.json").write_text(json.dumps(settings, indent=2))
    env = {**os.environ, "DIFFSYNTH_MODEL_BASE_PATH": str(models), "DIFFSYNTH_DOWNLOAD_SOURCE": "huggingface",
           "DIFFSYNTH_SKIP_DOWNLOAD": "true", "TOKENIZERS_PARALLELISM": "false"}
    common = ["--bundle", args.bundle, "--upstream-root", str(args.upstream_root), "--cache-dir", str(args.cache_dir),
              "--work-dir", str(args.work_dir)]
    upstream_env = {key: value for key, value in env.items() if key != "PYTHONPATH"}
    subprocess.run([args.upstream_python, "-I", str(SCRIPT), "--role", "upstream", *common], env=upstream_env, check=True)
    ours_env = {**env, "PYTHONPATH": os.pathsep.join(filter(None, [str(REPOSITORY.parent), env.get("PYTHONPATH")]))}
    subprocess.run([args.ours_python, str(SCRIPT), "--role", "ours", *common], env=ours_env, check=True)

    import torch
    upstream = torch.load(args.work_dir / "upstream.pt", weights_only=False)
    ours = torch.load(args.work_dir / "ours.pt", weights_only=False)
    rows, baseline, failed = _compare(upstream, ours, args.atol)
    report = {"settings": settings, "upstream_revision": UPSTREAM_REVISION, "rows": rows,
              "run_to_run_max_abs_diff_action": baseline, "atol": args.atol, "passed": not failed,
              "environments": {"upstream": upstream["environment"], "ours": ours["environment"]}}
    (args.work_dir / "report.json").write_text(json.dumps(report, indent=2))
    for row in rows:
        print(row)
    print("run-to-run max|diff| (same process):", baseline)
    print("PARITY", "PASSED" if not failed else "FAILED", "->", args.work_dir / "report.json")
    return 1 if failed else 0



def _load_inputs(args):
    import numpy as np

    settings = json.loads((args.work_dir / "settings.json").read_text())
    with np.load(args.work_dir / "observations.npz", allow_pickle=False) as data:
        observations = {key: data[key] for key in data.files}
    return settings, observations


def _raw_step(observations, benchmark, step):
    if benchmark == "libero":
        return {key: observations[key][step] for key in LIBERO_KEYS}
    return {"joint_action": {"vector": observations["joint_vector"][step]},
            "observation": {name: {"rgb": observations[name][step]} for name in ("head_camera", "left_camera", "right_camera")}}


def _capture_infer_joint_ar(base_class, calls):
    original = base_class.infer_joint_ar

    def wrapped(self, *args, **kwargs):
        if args:
            raise RuntimeError("Expected keyword-only infer_joint_ar calls")
        output = original(self, **kwargs)

        def cpu(name):
            value = kwargs.get(name)
            return None if value is None else value.detach().cpu().clone()

        calls.append({name: cpu(name) for name in ("input_image", "obs_window", "proprio", "context", "context_mask")}
                     | {name: kwargs.get(name) for name in ("seed", "num_inference_steps", "action_horizon",
                                                            "sigma_shift", "rand_device")}
                     | {"action": output["action"].detach().cpu().clone()})
        return output

    base_class.infer_joint_ar = wrapped


def _environment():
    import importlib.metadata as metadata
    import torch

    versions = {}
    for name in ("torch", "torchvision", "numpy", "pillow", "transformers", "hydra-core", "omegaconf"):
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    versions["device"] = torch.cuda.get_device_name() if torch.cuda.is_available() else "cpu"
    return versions



def _unbatch(batch):
    """longwam.runtime.lerobot._observation for uint8 images, without importing lerobot."""
    import numpy as np
    import torch

    raw = {}
    for key, value in batch.items():
        if not key.startswith("observation."):
            continue
        value = torch.as_tensor(value).detach().cpu()[0]
        if key.startswith("observation.images."):
            value = value.permute(1, 2, 0)
        raw[key] = np.ascontiguousarray(value.numpy())
    return raw


def _run_upstream(args):
    sys.path.insert(0, str(args.upstream_root / "src"))
    import torch
    import longwam
    from longwam.models.wan22.longwam_base import LongWAMBase
    from longwam.runtime.policy import initialize_policy

    if not Path(longwam.__file__).resolve().is_relative_to(args.upstream_root):
        raise SystemExit(f"Imported longwam from {longwam.__file__}, not {args.upstream_root}")
    settings, observations = _load_inputs(args)
    benchmark, instruction = settings["benchmark"], str(observations["instruction"])
    bundle = Path(settings["bundle_dir"])
    # The released config omits processor.train_transforms, which upstream's runtime requires.
    from omegaconf import OmegaConf
    run_config = OmegaConf.load(bundle / "config.yaml")
    if "train_transforms" not in run_config.data.train.processor:
        print("note: adding data.train.processor.train_transforms=null for upstream's processor constructor")
        run_config.data.train.processor.train_transforms = None
    model_config = args.work_dir / "upstream_config.yaml"
    OmegaConf.save(run_config, model_config)
    runtime = {
        "benchmark": "libero" if benchmark == "libero" else "robotwin2",
        "checkpoint": str(bundle / "model.pt"), "model_config": str(model_config),
        "stats": str(bundle / "dataset_stats.json"), "output_dir": str(args.work_dir / "upstream_output"),
        "device": settings["device"], "seed": settings["seed"], "gpu_id": 0,
        "num_inference_steps": settings["num_inference_steps"], "action_horizon": settings["action_horizon"],
        "replan_steps": settings["action_horizon"], "execute_steps": settings["action_horizon"],
        "inference_mode": settings["condition"], "execution": "sync", "trigger_stride": None,
        "streaming_vae": False, "hardware": "reference", "optimization": {}, "text_cache": None,
        "policy_options": {"sigma_shift": None, "text_cfg_scale": 1.0, "negative_prompt": "", "rand_device": "cpu",
                           "tiled": False},
    }
    if benchmark == "libero":
        from longwam.benchmarks.libero.adapter import policy_batch
        runtime.update(suite="libero_10", task_id=0, episodes=1)
        runtime["policy_options"].update(env_num=1, num_steps_wait=30, binarize_gripper=True, use_action_ensembler=False)
        cameras = {"image": "image", "wrist_image": "image2"}
    else:
        from longwam.benchmarks.robotwin.adapter import policy_batch
        runtime.update(simulator_root=str(args.work_dir), task="parity", setting="demo_clean", episodes=1)
        cameras = {"cam_high": "head_camera", "cam_left_wrist": "left_camera", "cam_right_wrist": "right_camera"}
    calls = []
    _capture_infer_joint_ar(LongWAMBase, calls)
    handle, _ = initialize_policy(runtime)
    handle("reset", {"episode": 1, "instruction": instruction})
    commands, anchors = [], settings["anchors"]
    for step in range(max(anchors) + 1):
        raw = _unbatch(policy_batch(_raw_step(observations, benchmark, step), instruction))
        observation = {"state": raw["observation.state"],
                       "images": {key: raw[f"observation.images.{feature}"] for key, feature in cameras.items()}}
        handle("observe", {"episode": 1, "step": step, "observation": observation})
        if step in anchors:
            commands.append(handle("infer", {"episode": 1, "anchor": step})["actions"])
    commands.append(handle("infer", {"episode": 1, "anchor": max(anchors)})["actions"])  # run-to-run baseline
    for call, anchor, command in zip(calls, [*anchors, max(anchors)], commands):
        call.update(anchor=anchor, commands=torch.as_tensor(command))
    torch.save({"calls": calls, "environment": _environment()}, args.work_dir / "upstream.pt")
    return 0



def _run_ours(args):
    import torch
    from wam_interp.evaluations.LIBERO.environment import observation_from_libero
    from wam_interp.evaluations.ROBOTWIN.environment import observation_from_robotwin
    from wam_interp.models.longwam.loader import load_longwam
    from wam_interp.models.longwam.model_code.models.wan22.longwam_base import LongWAMBase
    from wam_interp.types import InferenceRequest

    settings, observations = _load_inputs(args)
    benchmark, instruction = settings["benchmark"], str(observations["instruction"])
    bundle = Path(settings["bundle_dir"])
    model = load_longwam(bundle / "config.yaml", bundle / "model.pt", device=settings["device"])
    if benchmark == "libero":
        from wam_interp.models.longwam.libero_adapter import LongWAMLiberoAdapter
        adapter = LongWAMLiberoAdapter(model, dataset_stats=bundle / "dataset_stats.json", binarize_gripper=True)
        frontend = lambda raw: observation_from_libero(raw, instruction=instruction)
        environment = "libero"
    else:
        from wam_interp.models.longwam.robotwin_adapter import LongWAMRobotWinAdapter
        adapter = LongWAMRobotWinAdapter(model, dataset_stats=bundle / "dataset_stats.json")
        frontend = lambda raw: observation_from_robotwin(raw, instruction)
        environment = "robotwin"
    defaults = adapter.inference_defaults()
    if defaults != {"condition": settings["condition"], "num_inference_steps": settings["num_inference_steps"]}:
        print(f"note: adapter defaults {defaults} differ from the requested settings; using the requested settings")
    calls = []
    _capture_infer_joint_ar(LongWAMBase, calls)
    commands, anchors = [], settings["anchors"]

    def infer(observation):
        request = InferenceRequest(adapter.prepare_observation(observation), condition=settings["condition"],
                                   action_horizon=settings["action_horizon"],
                                   num_inference_steps=settings["num_inference_steps"], seed=settings["seed"])
        result = adapter.predict_action(request)
        commands.append(torch.as_tensor(adapter.postprocess_actions(result.actions, environment=environment)))

    for step in range(max(anchors) + 1):
        observation = frontend(_raw_step(observations, benchmark, step))
        if step == 0:
            adapter.begin_episode(observation)
        else:
            adapter.observe_step(observation, action=None)
        if step in anchors:
            infer(observation)
    infer(observation)  # run-to-run baseline at the last anchor
    for call, anchor, command in zip(calls, [*anchors, max(anchors)], commands):
        call.update(anchor=anchor, commands=command)
    torch.save({"calls": calls, "environment": _environment(), "metadata": model._wam_interp_metadata},
               args.work_dir / "ours.pt")
    return 0


if __name__ == "__main__":
    sys.exit(main())
