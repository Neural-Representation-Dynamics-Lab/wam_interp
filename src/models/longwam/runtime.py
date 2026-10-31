"""Inference-only port of upstream ``runtime/factory.py:create_longwam``."""


def _plain(value, name, *, default=None):
    from omegaconf import DictConfig, OmegaConf

    if isinstance(value, DictConfig):
        value = OmegaConf.to_container(value, resolve=True)
    if value is None:
        value = default
    if value is not None and not isinstance(value, dict):
        raise ValueError(f"`{name}` must resolve to a dict, got {type(value)}")
    return value


def create_longwam(
    model_id: str,
    tokenizer_model_id: str,
    video_dit_config,
    tokenizer_max_len: int = 512,
    load_text_encoder: bool = True,
    proprio_dim: int | None = None,
    action_dit_config=None,
    action_dit_pretrained_path: str | None = None,
    skip_dit_load_from_pretrain: bool = True,
    video_scheduler=None,
    action_scheduler=None,
    loss=None,
    mot_checkpoint_mixed_attn: bool = True,
    redirect_common_files: bool = True,
    model_dtype=None,
    device: str = "cuda",
    longlive_video_weights: str | None = None,
    longwam_num_frame_per_block: int = 1,
    longwam_local_attn_size: int = -1,
    longwam_sink_size: int = 0,
    longwam_past_obs_size: int = 0,
    longwam_num_imagine_frames: int = 0,
    longwam_imagine_sigma: float = 0.9,
    longwam_imagine_infer_steps: int = 4,
    longwam_joint_denoise: bool = False,
    skip_video_dit_load_from_pretrain: bool | None = None,
    skip_action_dit_load_from_pretrain: bool | None = None,
):
    import torch
    from .model_code.models.wan22.long_wam import LongWAM

    if longlive_video_weights:
        raise ValueError("longlive_video_weights is a training initializer; inference loads the full checkpoint")
    video_dit_config = _plain(video_dit_config, "video_dit_config")
    if video_dit_config is None:
        raise ValueError("`video_dit_config` must resolve to a dict, got None")
    action_dit_config = _plain(action_dit_config, "action_dit_config", default={})
    video_scheduler = _plain(video_scheduler, "video_scheduler", default={})
    action_scheduler = _plain(action_scheduler, "action_scheduler")
    if action_scheduler is None:
        raise ValueError("`action_scheduler` is required for LongWAM.")
    missing = {"train_shift", "infer_shift", "num_train_timesteps"} - set(action_scheduler)
    if missing:
        raise ValueError(f"`action_scheduler` missing required keys: {sorted(missing)}.")
    loss = _plain(loss, "loss", default={})

    legacy_skip = bool(skip_dit_load_from_pretrain)
    skip_video = legacy_skip if skip_video_dit_load_from_pretrain is None else bool(skip_video_dit_load_from_pretrain)
    skip_action = legacy_skip if skip_action_dit_load_from_pretrain is None else bool(skip_action_dit_load_from_pretrain)
    return LongWAM.from_wan22_pretrained(
        device=device,
        torch_dtype=torch.bfloat16 if model_dtype is None else model_dtype,
        model_id=model_id,
        tokenizer_model_id=tokenizer_model_id,
        tokenizer_max_len=int(tokenizer_max_len),
        load_text_encoder=bool(load_text_encoder),
        proprio_dim=None if proprio_dim is None else int(proprio_dim),
        redirect_common_files=bool(redirect_common_files),
        video_dit_config=video_dit_config,
        action_dit_config=action_dit_config,
        action_dit_pretrained_path=action_dit_pretrained_path,
        skip_dit_load_from_pretrain=legacy_skip,
        skip_video_dit_load_from_pretrain=skip_video,
        skip_action_dit_load_from_pretrain=skip_action,
        mot_checkpoint_mixed_attn=bool(mot_checkpoint_mixed_attn),
        video_train_shift=float(video_scheduler.get("train_shift", 5.0)),
        video_infer_shift=float(video_scheduler.get("infer_shift", 5.0)),
        video_num_train_timesteps=int(video_scheduler.get("num_train_timesteps", 1000)),
        action_train_shift=float(action_scheduler["train_shift"]),
        action_infer_shift=float(action_scheduler["infer_shift"]),
        action_num_train_timesteps=int(action_scheduler["num_train_timesteps"]),
        loss_lambda_video=float(loss.get("lambda_video", 1.0)),
        loss_lambda_action=float(loss.get("lambda_action", 1.0)),
        longwam_num_frame_per_block=int(longwam_num_frame_per_block),
        longwam_local_attn_size=int(longwam_local_attn_size),
        longwam_sink_size=int(longwam_sink_size),
        longwam_past_obs_size=int(longwam_past_obs_size),
        longwam_num_imagine_frames=int(longwam_num_imagine_frames),
        longwam_imagine_sigma=float(longwam_imagine_sigma),
        longwam_imagine_infer_steps=int(longwam_imagine_infer_steps),
        longwam_joint_denoise=bool(longwam_joint_denoise),
    )
