"""Inference-only Hydra factory for the native Faster-WAM implementation."""


def create_fasterwam(*, model_dtype, device="cuda", **config):
    from omegaconf import OmegaConf
    from .model_code.fasterwam import FasterWAM

    def plain(value):
        return OmegaConf.to_container(value, resolve=True) if OmegaConf.is_config(value) else value

    config = {key: plain(value) for key, value in config.items()}
    video_schedule = config.pop("video_scheduler")
    action_schedule = config.pop("action_scheduler")
    loss = config.pop("loss", {})
    for prefix, schedule in (("video", video_schedule), ("action", action_schedule)):
        config[f"{prefix}_train_shift"] = float(schedule["train_shift"])
        config[f"{prefix}_infer_shift"] = float(schedule["infer_shift"])
        config[f"{prefix}_num_train_timesteps"] = int(schedule["num_train_timesteps"])
    config["loss_lambda_video"] = float(loss.get("lambda_video", 1.0))
    config["loss_lambda_action"] = float(loss.get("lambda_action", 1.0))
    return FasterWAM.from_wan22_pretrained(device=device, torch_dtype=model_dtype, **config)
