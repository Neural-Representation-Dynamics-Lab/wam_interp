"""Concrete FastWAM Optional IDM / LIBERO pilot at a reproducible observation.

Restoration recreates the simulator and controller and replays deterministic
warm-up and an optional recorded native prefix. It never pretends that restoring
MuJoCo qpos alone restores an episode.
"""
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import random
import subprocess
from time import perf_counter

import numpy as np

from .e1 import context_id
from .fastwam_session import FastWAMChunkSession, _array
from .futures import predict_with_future


def digest_array(value):
    a = np.ascontiguousarray(_array(value))
    return hashlib.sha256(str((a.shape, str(a.dtype))).encode() + a.tobytes()).hexdigest()


def digest_file(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def rng_snapshot():
    import torch
    return {"python": random.getstate(), "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None}


def rng_restore(state):
    import torch
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if state["cuda"] is not None:
        torch.cuda.set_rng_state_all(state["cuda"])


def save_frames(model, latents, directory):
    import torch
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    with torch.inference_mode():
        frames = model._decode_latents(torch.as_tensor(latents, device=model.device,
                                                      dtype=model.torch_dtype))
    for i, frame in enumerate(frames):
        frame.save(directory / f"{i:03d}.png")
    return len(frames)


class LiberoPilot:
    def __init__(self, config, adapter, *, task, initial_state, create_env, reset_env, step_env):
        self.config, self.adapter, self.task = config, adapter, task
        self.initial_state = initial_state
        self.create_env, self.reset_env, self.step_env = create_env, reset_env, step_env
        self.env = None
        self.last_rollout = None
        from .replay_prefix import load_replay_prefix
        self.replay_data, self.replay_identity = load_replay_prefix(config)
        try:
            self.reset()
            self.reference = self.observation
            self.request = self.make_request()
            self.saved_rng = rng_snapshot()
        except Exception:
            self.close()
            raise
        self.session = FastWAMChunkSession(adapter, self.request,
            snapshot_fn=lambda: self.saved_rng, restore_fn=self.restore,
            rollout_fn=self.rollout, execute_steps=config["execute_steps"])

    def make_request(self):
        from wam_interp.types import InferenceRequest
        options = dict(self.adapter.inference_defaults())
        options.update(condition="idm", seed=self.config["action_seed"],
                       num_inference_steps=self.config["inference_steps"],
                       compile_action_infer=False)
        return InferenceRequest(self.adapter.prepare_observation(self.observation),
                                action_horizon=32, **options)

    def reset(self):
        self.close()
        random.seed(self.config["environment_seed"])
        np.random.seed(self.config["environment_seed"])
        self.env = self.create_env(self.task, self.config["camera_resolution"], self.config["environment_seed"])
        meta = {"task_id": self.config["task_id"], "initial_state_index": self.config["initial_state_index"]}
        self.observation = self.reset_env(self.env, self.initial_state,
            instruction=self.task.language, metadata=meta)
        for _ in range(self.config["warmup_steps"]):
            from wam_interp.evaluations.LIBERO.environment import LIBERO_DUMMY_ACTION
            step = self.step_env(self.env, LIBERO_DUMMY_ACTION, instruction=self.task.language, metadata=meta)
            if step.done or step.success:
                raise RuntimeError("Episode terminated during warm-up")
            self.observation = step.observation

        from .replay_prefix import replay_prefix
        replay_prefix(self)

    def restore(self, state):
        self.reset()
        # Reject restoration drift before any policy branch can execute.
        for group in ("cameras", "state"):
            for name, value in getattr(self.reference, group).items():
                actual = getattr(self.observation, group)[name]
                if not np.array_equal(value, actual):
                    raise RuntimeError(f"Initial-state replay differs in {group}/{name}")
        rng_restore(state)

    def rollout(self, commands):
        observations = [self.observation]
        executed = []
        reason = "chunk_complete"
        success = False
        for command in commands:
            step = self.step_env(self.env, command, instruction=self.task.language, metadata={})
            executed.append(command.copy())
            observations.append(step.observation)
            if step.done or step.success:
                success = step.success
                reason = "success" if step.success else "environment_done"
                break
        positions = np.stack([o.state["eef_position"] for o in observations])
        self.last_rollout = {"eef_position": positions,
            "eef_quaternion": np.stack([o.state["eef_quaternion"] for o in observations]),
            "agentview": np.stack([o.cameras["agentview"] for o in observations]),
            "wrist": np.stack([o.cameras["wrist"] for o in observations]),
            "control_steps": np.arange(len(observations))}
        return {"executed_commands": np.asarray(executed),
            "behavior": float(positions[-1, self.config["behavior_axis"]] - positions[0, self.config["behavior_axis"]]),
            "metadata": {"termination": reason, "success": success,
                         "behavior_units": "meters", "behavior_frame": "simulator_world"}}

    def verify_replay(self):
        commands = np.zeros((2, 7))
        commands[:, -1] = -1
        self.restore(self.saved_rng)
        self.rollout(commands)
        first = self.last_rollout
        self.restore(self.saved_rng)
        self.rollout(commands)
        for key in first:
            if not np.array_equal(first[key], self.last_rollout[key]):
                raise RuntimeError(f"Simulator replay is not identical: {key}")
        self.restore(self.saved_rng)

    def infer(self, seed=None, donor=None, prompt=None):
        self.restore(self.saved_rng)
        request = self.request if seed is None else replace(self.request, seed=seed)
        if prompt is not None:
            # Donor acquisition only: never mutate the recipient request held
            # by the execution session.
            request = replace(request, observation=replace(request.observation, prompt=prompt))
        import torch
        if torch.cuda.is_initialized():
            torch.cuda.synchronize()
        start = perf_counter()
        result, captured = predict_with_future(self.adapter, request, donor)
        if torch.cuda.is_initialized():
            torch.cuda.synchronize()
        return result, captured, perf_counter() - start

    def close(self):
        if self.env is not None:
            self.env.close()
            self.env = None


def validate_config(config):
    required = {"backend", "precision", "behavior_axis", "execute_steps", "warmup_steps",
        "inference_steps", "candidate_seeds", "identity_atol", "environment_seed", "action_seed",
        "task_id", "initial_state_index", "camera_resolution", "suite", "checkpoint", "statistics",
        "device", "repository_root", "property", "frame_steps"}
    if not required <= config.keys():
        raise ValueError(f"Missing config fields: {sorted(required - config.keys())}")
    for key in ("behavior_axis", "execute_steps", "warmup_steps", "inference_steps",
                "environment_seed", "action_seed", "task_id", "initial_state_index", "camera_resolution"):
        if type(config[key]) is not int or config[key] < 0:
            raise ValueError(f"{key} must be a nonnegative integer")
    if config["camera_resolution"] == 0 or not isinstance(config["property"], str) or not config["property"].strip():
        raise ValueError("Camera resolution and property must be specified")
    if config["suite"] not in {"libero_10", "libero_90", "libero_spatial", "libero_object", "libero_goal"}:
        raise ValueError("Unknown LIBERO suite")
    if any(type(seed) is not int or seed < 0 for seed in config["candidate_seeds"]):
        raise ValueError("Candidate seeds must be nonnegative integers")
    steps = config["frame_steps"]
    if not steps or steps[0] != 0 or any(type(s) is not int for s in steps) or any(b <= a for a, b in zip(steps, steps[1:])):
        raise ValueError("Frame mapping must start at zero and strictly increase")
    if config["backend"] != "fastwam_optional_idm" or config["precision"] != "bfloat16":
        raise ValueError("Initial pilot supports FastWAM Optional IDM in bf16 only")
    if config["behavior_axis"] not in (0, 1, 2) or not 1 <= config["execute_steps"] <= 32:
        raise ValueError("Invalid behavior_axis or execute_steps")
    if config["warmup_steps"] < 0 or config["inference_steps"] < 1:
        raise ValueError("Invalid warmup or inference steps")
    if len(set(config["candidate_seeds"])) != len(config["candidate_seeds"]) or len(config["candidate_seeds"]) < 2:
        raise ValueError("At least two distinct candidate seeds required")
    prompts = config.get("donor_prompts")
    if prompts is not None:
        import re
        if not isinstance(prompts, dict) or len(prompts) < 2 or any(
            not isinstance(key, str) or not re.fullmatch(r"[a-z][a-z0-9_]*", key) or not isinstance(value, str) or not value.strip()
            for key, value in prompts.items()
        ):
            raise ValueError("donor_prompts requires at least two safe identifiers and nonempty instructions")
    if not np.isfinite(config["identity_atol"]) or config["identity_atol"] < 0:
        raise ValueError("identity_atol must be nonnegative")


def load_pilot(config):
    validate_config(config)
    for field in ("checkpoint", "statistics"):
        if not Path(config[field]).is_file():
            raise FileNotFoundError(config[field])
    from omegaconf import OmegaConf
    from wam_interp.models.fastwam.loader import compose_fastwam_config, load_fastwam
    from wam_interp.models.fastwam.libero_adapter import FastWAMLiberoAdapter
    from wam_interp.evaluations.LIBERO.runner import _get_suite
    from wam_interp.evaluations.LIBERO.environment import (
        create_libero_environment, initial_states_for_task, reset_to_initial_state, step_libero)
    # Use the official Wan assets: the optional converted-weight mirror can
    # disappear independently of the released model. Native loading supports
    # the original .pth weights and validates their architecture fingerprints.
    cfg = compose_fastwam_config("libero_optional_idm_2cam224_1e-4", config["repository_root"],
                                overrides=["model.redirect_common_files=false"])
    model = load_fastwam(cfg, config["checkpoint"], device=config["device"], dtype=config["precision"])
    adapter = FastWAMLiberoAdapter(model, data_config=cfg.data.train, dataset_stats=config["statistics"])
    suite = _get_suite(config["suite"])
    task = suite.get_task(config["task_id"])
    state = initial_states_for_task(suite, config["task_id"])[config["initial_state_index"]]
    pilot = LiberoPilot(config, adapter, task=task, initial_state=state,
        create_env=create_libero_environment, reset_env=reset_to_initial_state, step_env=step_libero)
    pilot.resolved_model_config = OmegaConf.to_container(cfg.model, resolve=True)
    return pilot


def provenance(pilot):
    c = pilot.config
    root = Path(c["repository_root"])
    from wam_interp.evaluations.provenance import package_versions
    source = hashlib.sha256()
    # Include uncommitted source so local work can be transported without a push.
    for folder in ("src", "configs", "scripts"):
        for path in sorted((root / folder).rglob("*")):
            if path.is_file() and path.suffix in (".py", ".json", ".yaml"):
                source.update(path.relative_to(root).as_posix().encode())
                source.update(path.read_bytes())
    from dataclasses import fields
    request_options = {f.name: getattr(pilot.request, f.name) for f in fields(pilot.request) if f.name != "observation"}
    request_options = json.loads(json.dumps(request_options))
    assets = {}
    for name, location in getattr(pilot.adapter.model, "model_paths", {}).items():
        if location is None or str(location).startswith("SKIPPED"):
            assets[name] = location
            continue
        path = Path(location)
        if path.is_file():
            assets[name] = digest_file(path)
        elif path.is_dir():
            assets[name] = {p.relative_to(path).as_posix(): digest_file(p) for p in sorted(path.rglob("*"))
                            if p.is_file() and ".cache" not in p.parts}
        else:
            raise FileNotFoundError(f"Cannot identify loaded base asset: {location}")
    return {"repo_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
        "source_sha256": source.hexdigest(), "backend": c["backend"],
        "checkpoint": digest_file(c["checkpoint"]), "statistics": digest_file(c["statistics"]),
        "resolved_config": pilot.resolved_model_config,
        "observation_sha256": digest_array(pilot.request.observation.image),
        "proprio_sha256": digest_array(pilot.request.observation.proprio),
        "state_sha256": digest_array(pilot.initial_state), "prompt": pilot.task.language,
        "action_seed": c["action_seed"], "precision": c["precision"],
        "request_options": request_options,
        "base_assets": assets,
        "behavior_measure": f"eef_world_axis_{c['behavior_axis']}_displacement_m",
        "state_id": f"{c['suite']}/{c['task_id']}/{c['initial_state_index']}",
        "protocol": {**{k: c[k] for k in ("suite", "task_id", "initial_state_index", "warmup_steps",
            "environment_seed", "execute_steps", "inference_steps", "camera_resolution", "identity_atol")},
            "donor_generation": {"kind": "counterfactual_instruction" if c.get("donor_prompts") else "same_instruction_seed_variation",
                                 "prompts": c.get("donor_prompts"), "seeds": c["candidate_seeds"]}},
        "versions": package_versions(),
        **({"replay_prefix": pilot.replay_identity} if getattr(pilot, 'replay_identity', None) else {})}
