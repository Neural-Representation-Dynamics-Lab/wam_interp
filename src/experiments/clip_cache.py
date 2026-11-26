"""Experimental first-chunk, fixed-recipient RGB donors for the LingBot and OpenWAM KV sites."""

from contextlib import contextmanager
from unittest.mock import patch


def _replace_future(current, future):
    import torch

    if current.ndim != 5 or future.shape[1:] != current[:, :, 1:].shape[1:]:
        raise ValueError("Encoded future does not fit native video slots")
    if future.shape[0] not in (1, current.shape[0]) or not torch.isfinite(future).all():
        raise ValueError("Invalid clean future batch")
    return torch.cat(
        (current[:, :, :1], future.to(current).expand_as(current[:, :, 1:])), dim=2
    )


@contextmanager
def lingbot_clip(api, req, frames):
    """Replace clean future input only for the donor's terminal predicted video write."""
    import torch
    from wam_interp.models.lingbot_va.model_code.loading import WanVAEStreamingWrapper

    runtime = api.owner
    cfg = runtime.config
    if (
        runtime.frame_st_id != 0
        or cfg.video_exec_step != -1
        or len(frames) != 1 + 4 * (cfg.frame_chunk_size - 1)
        or not torch.equal(frames[0], req.observation.image)
    ):
        raise ValueError(
            "Expected a matching first-chunk anchor and full clean-video schedule"
        )
    # The real episode's causal encoder state must survive synthetic clip encoding.
    with (
        torch.inference_mode(),
        patch.object(runtime, "streaming_vae", WanVAEStreamingWrapper(runtime.vae)),
    ):
        runtime.encode_frames(frames[:1])
        future = torch.cat(
            [
                runtime.encode_frames(frames[i : i + 4])
                for i in range(1, len(frames), 4)
            ],
            2,
        )
    calls = []

    def modify(module, args, kwargs):
        if kwargs.get("action_mode") is False and kwargs.get("update_cache") == 1:
            (payload,) = args
            if payload["timesteps"].count_nonzero():
                raise ValueError("Expected terminal clean video cache write")
            args = (
                dict(
                    payload,
                    noisy_latents=_replace_future(payload["noisy_latents"], future),
                ),
            )
            calls.append(True)
        return args, kwargs

    handle = runtime.transformer.register_forward_pre_hook(modify, with_kwargs=True)
    try:
        yield
        if len(calls) != 1:
            raise ValueError("Expected exactly one terminal video cache write")
    finally:
        handle.remove()


@contextmanager
def openwam_clip(api, req, frames):
    """Capture one layer at a noninitial joint step with native action state and sequence mask."""
    import torch
    from wam_interp.models.openwam.model_code.model.video_backbone.wan import encode

    arch = api.adapter.architecture
    vb = arch.video_backbone
    if (
        not torch.equal(frames[0], req.observation.image)
        or len(frames) != req.num_video_frames
        or api.stage == 0
    ):
        raise ValueError(
            "Expected a matching recipient anchor, native frame count and noninitial stage"
        )
    with torch.inference_mode():
        video = torch.stack(frames, 1)[None].to(device=vb.device, dtype=vb.dtype)
        clean = encode.encode_video(video, vae=vb.vae, encoder=vb.video_encoder)[
            :, :, 1:
        ]
    forward = arch.forward
    initial_noise = None
    calls = 0
    applied = []

    def replace(*args, **kwargs):
        nonlocal calls, initial_noise
        latents = kwargs["latents"]
        sigma = (
            float(kwargs["timestep"].flatten()[0])
            / arch.video_scheduler.num_train_timesteps
        )
        if calls == 0:
            if abs(sigma - 1.0) > 0.002:
                raise ValueError("Expected native Gaussian at first joint evaluation")
            initial_noise = latents[:, :, 1:].detach().clone()
        if calls == api.stage:
            if not 0 <= sigma < 1:
                raise ValueError("Selected joint stage has no clean signal")
            kwargs["latents"] = _replace_future(
                latents, (1 - sigma) * clean.to(latents) + sigma * initial_noise
            )
            applied.append(sigma)
        calls += 1
        return forward(*args, **kwargs)

    with patch.object(arch, "forward", replace):
        yield
    if len(applied) != 1:
        raise ValueError("Requested joint stage was not reached")
