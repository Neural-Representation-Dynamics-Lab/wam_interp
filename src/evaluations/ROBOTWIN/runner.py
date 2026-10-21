"""Model-neutral RoboTwin evaluation with bounded expert seed screening."""

from time import perf_counter
import numpy as np

from wam_interp.types import InferenceRequest, InferenceResult
from .._libero_rollout import _json_safe
from ..provenance import package_versions, model_provenance, save_report, synchronize_adapter
from .config import RoboTwinEvalConfig, ROBOTWIN_REVISION
from .environment import open_environment


def evaluate_robotwin(model, config: RoboTwinEvalConfig):
    config.validate()
    for method in ("prepare_observation", "predict_action", "postprocess_actions", "inference_defaults", "environment_action_type"):
        if not callable(getattr(model, method, None)):
            raise TypeError(f"RoboTwin evaluation adapter is missing {method}")
    action_type = model.environment_action_type("robotwin")
    if action_type not in {"qpos", "ee"}:
        raise ValueError(f"Unsupported RoboTwin action type {action_type!r}")
    expected_dim = 14 if action_type == "qpos" else 16
    if getattr(model, "requires_full_action_chunk", False) and config.replan_steps < config.action_horizon:
        raise ValueError("This adapter requires full action chunks; replan_steps must cover action_horizon")
    begin_episode = getattr(model, "begin_episode", None)
    observe_step = getattr(model, "observe_step", None)
    start = perf_counter()
    report = {"benchmark": "robotwin", "benchmark_revision": ROBOTWIN_REVISION,
              "config": _json_safe(config), "model": model_provenance(model),
              "package_versions": package_versions(), "action_type": action_type,
              "accepted_seeds": [], "rejected_seeds": [], "trials": [], "completed": False}
    seed = config.start_seed if config.start_seed is not None else 100000 * (1 + config.seed)
    attempts = 0
    try:
        with open_environment(config) as frontend:
            while len(report["trials"]) < config.num_episodes and attempts < config.max_seed_attempts:
                attempts += 1
                episode = len(report["trials"])
                accepted, info, reason = frontend.screen_seed(seed, episode)
                if not accepted:
                    report["rejected_seeds"].append({"seed": seed, "stage": "expert", "reason": reason})
                    seed += 1
                    continue
                try:
                    observation = frontend.begin(seed, episode, info, config.instruction_type)
                except Exception as exc:
                    report["rejected_seeds"].append({"seed": seed, "stage": "rollout_setup",
                                                      "reason": f"{type(exc).__name__}: {exc}"})
                    frontend.close()
                    seed += 1
                    continue
                report["accepted_seeds"].append(seed)
                trial = {"seed": seed, "episode": episode, "instruction": observation.instruction,
                         "success": False, "termination_reason": "max_steps", "policy_steps": 0,
                         "inference_calls": 0, "inference_metadata": [], "inference_latency_ms": []}
                report["trials"].append(trial)
                try:
                    if callable(begin_episode):
                        begin_episode(observation)
                    limit = min(config.max_steps or frontend.step_limit, frontend.step_limit)
                    while trial["policy_steps"] < limit and frontend.action_count < frontend.step_limit:
                        options = dict(model.inference_defaults())
                        options.update(config.inference_options)
                        options.setdefault("seed", seed + trial["inference_calls"])
                        request = InferenceRequest(observation=model.prepare_observation(observation),
                                                   action_horizon=config.action_horizon, **options)
                        synchronize_adapter(model)
                        tick = perf_counter()
                        prediction = model.predict_action(request)
                        if not isinstance(prediction, InferenceResult):
                            raise TypeError("predict_action must return InferenceResult")
                        decoded = model.postprocess_actions(prediction.actions, environment="robotwin")
                        synchronize_adapter(model)
                        if hasattr(decoded, "detach"):
                            decoded = decoded.detach().float().cpu().numpy()
                        decoded = np.asarray(decoded)
                        if decoded.ndim != 2 or decoded.shape[1] != expected_dim or not len(decoded):
                            raise ValueError(f"Expected {action_type} commands [T,{expected_dim}], got {decoded.shape}")
                        if not np.isfinite(decoded).all():
                            raise ValueError("Model returned nonfinite RoboTwin commands")
                        trial["inference_latency_ms"].append((perf_counter() - tick) * 1000.0)
                        trial["inference_metadata"].append(_json_safe(prediction.metadata))
                        trial["inference_calls"] += 1
                        for action in decoded[:min(config.replan_steps, limit - trial["policy_steps"])]:
                            observation, success = frontend.step(action, action_type)
                            trial["policy_steps"] += 1
                            if callable(observe_step):
                                observe_step(observation, action)
                            if success:
                                trial["success"] = True
                                trial["termination_reason"] = "success"
                                break
                            if frontend.action_count >= frontend.step_limit:
                                break
                        if trial["success"]:
                            break
                except Exception as exc:
                    trial["termination_reason"] = "error"
                    trial["error"] = {"type": type(exc).__name__, "message": str(exc)}
                    raise
                finally:
                    frontend.close()
                seed += 1
            if len(report["trials"]) != config.num_episodes:
                raise RuntimeError(f"Seed search exhausted after {attempts} attempts")
            report["completed"] = True
    except Exception as exc:
        report["error"] = {"type": type(exc).__name__, "message": str(exc)}
        raise
    finally:
        report["attempts"] = attempts
        report["duration_seconds"] = perf_counter() - start
        report["episodes"] = len(report["trials"])
        report["successes"] = sum(int(trial["success"]) for trial in report["trials"])
        report["success_rate"] = report["successes"] / report["episodes"] if report["episodes"] else None
        save_report(report, config.output_dir, "robotwin")
    return report
