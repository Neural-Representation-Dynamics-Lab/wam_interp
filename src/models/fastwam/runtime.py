"""Hydra factories for the copied FastWAM model implementations."""

from typing import Any

import torch
from omegaconf import OmegaConf

from .model_code.fastwam import FastWAM
from .model_code.fastwam_idm import FastWAMIDM
from .model_code.fastwam_joint import FastWAMJoint
from .model_code.fastwam_optional_idm import FastWAMOptionalIDM


def _as_dict(value: Any, name: str, default: dict | None = None) -> dict:
    if value is None:
        return {} if default is None else default
    if OmegaConf.is_config(value):
        value = OmegaConf.to_container(value, resolve=True)
    if not isinstance(value, dict):
        raise TypeError(f"{name} must be a mapping, got {type(value).__name__}")
    return value


def _create_model(
    model_class,
    *,
    model_id: str,
    tokenizer_model_id: str,
    video_dit_config: Any,
    tokenizer_max_len: int = 512,
    load_text_encoder: bool = True,
    proprio_dim: int | None = None,
    action_dit_config: Any = None,
    action_dit_pretrained_path: str | None = None,
    skip_dit_load_from_pretrain: bool = False,
    video_scheduler: Any = None,
    action_scheduler: Any = None,
    loss: Any = None,
    mot_checkpoint_mixed_attn: bool = False,
    compile_training_denoise: bool = False,
    redirect_common_files: bool = True,
    model_dtype: torch.dtype = torch.bfloat16,
    device: str = "cuda",
    action_idm_prob: float | None = None,
    video_cond_noise_prob: float = 0.5,
):
    video_config = _as_dict(video_dit_config, "video_dit_config")
    action_config = _as_dict(action_dit_config, "action_dit_config")
    video_schedule = _as_dict(video_scheduler, "video_scheduler")
    action_schedule = _as_dict(action_scheduler, "action_scheduler")
    loss_config = _as_dict(loss, "loss")

    required_scheduler_keys = {"train_shift", "infer_shift", "num_train_timesteps"}
    missing = required_scheduler_keys - action_schedule.keys()
    if missing:
        raise ValueError(f"action_scheduler is missing: {sorted(missing)}")

    kwargs = {
        "model_id": model_id,
        "tokenizer_model_id": tokenizer_model_id,
        "video_dit_config": video_config,
        "tokenizer_max_len": int(tokenizer_max_len),
        "load_text_encoder": bool(load_text_encoder),
        "proprio_dim": None if proprio_dim is None else int(proprio_dim),
        "action_dit_config": action_config,
        "action_dit_pretrained_path": action_dit_pretrained_path,
        "skip_dit_load_from_pretrain": bool(skip_dit_load_from_pretrain),
        "mot_checkpoint_mixed_attn": bool(mot_checkpoint_mixed_attn),
        "compile_training_denoise": bool(compile_training_denoise),
        "redirect_common_files": bool(redirect_common_files),
        "device": device,
        "torch_dtype": model_dtype,
        "video_train_shift": float(video_schedule.get("train_shift", 5.0)),
        "video_infer_shift": float(video_schedule.get("infer_shift", 5.0)),
        "video_num_train_timesteps": int(video_schedule.get("num_train_timesteps", 1000)),
        "action_train_shift": float(action_schedule["train_shift"]),
        "action_infer_shift": float(action_schedule["infer_shift"]),
        "action_num_train_timesteps": int(action_schedule["num_train_timesteps"]),
        "loss_lambda_video": float(loss_config.get("lambda_video", 1.0)),
        "loss_lambda_action": float(loss_config.get("lambda_action", 1.0)),
    }
    if model_class in (FastWAMIDM, FastWAMOptionalIDM):
        kwargs["video_cond_noise_prob"] = float(video_cond_noise_prob)
    if model_class is FastWAMOptionalIDM:
        if action_idm_prob is None:
            raise ValueError("action_idm_prob is required for FastWAMOptionalIDM")
        kwargs["action_idm_prob"] = float(action_idm_prob)

    return model_class.from_wan22_pretrained(**kwargs)


def create_fastwam(**kwargs):
    return _create_model(FastWAM, **kwargs)


def create_fastwam_idm(**kwargs):
    return _create_model(FastWAMIDM, **kwargs)


def create_fastwam_optional_idm(**kwargs):
    return _create_model(FastWAMOptionalIDM, **kwargs)


def create_fastwam_joint(**kwargs):
    return _create_model(FastWAMJoint, **kwargs)
