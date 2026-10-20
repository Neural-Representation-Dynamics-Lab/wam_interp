"""LingBot-VA inference adapter; the model keeps a per-episode KV cache."""

from typing import Any

from wam_interp.models.registry import register_model_adapter
from wam_interp.tracing.hooks import capture_module_outputs
from wam_interp.types import InferenceResult

CONDITIONS = ("video",)


@register_model_adapter("lingbot_va")
class LingBotVAAdapter:
    name = "lingbot_va"

    def __init__(self, model: Any):
        self.model = model

    @property
    def action_horizon(self) -> int:
        return self.model.config.frame_chunk_size * self.model.config.action_per_frame

    def inference_defaults(self):
        return {"condition": "video",
                "num_inference_steps": int(self.model.config.num_inference_steps)}

    def reset(self, prompt: str) -> None:
        self.model.reset(prompt)

    def update(self, frames: list[Any], executed_chunk: Any) -> None:
        """Write the frames observed while executing the last chunk into the cache."""
        self.model.compute_kv_cache(frames, executed_chunk)

    def predict_action(self, request, recorder=None):
        import torch

        if self.model.prompt_embeds is None:
            raise RuntimeError("Call reset(prompt) before the first LingBot-VA prediction")
        if request.condition not in CONDITIONS:
            raise ValueError(f"LingBot-VA conditions are {CONDITIONS}; got {request.condition!r}")
        if request.action_horizon != self.action_horizon:
            raise ValueError(f"LingBot-VA predicts {self.action_horizon} actions per chunk; "
                             f"got action_horizon={request.action_horizon}")
        if request.trace_modules and recorder is None:
            from wam_interp.tracing.recorder import TraceRecorder
            recorder = TraceRecorder()

        cfg = self.model.config
        generator = None
        if request.seed is not None:
            generator = torch.Generator(device=self.model.device).manual_seed(int(request.seed))
        frame_st_id = self.model.frame_st_id
        init_frames = [request.observation.image] if frame_st_id == 0 else None

        with capture_module_outputs(self.model.transformer, request.trace_modules, recorder):
            actions, _latents = self.model.infer_chunk(
                init_frames, generator=generator,
                num_inference_steps=int(request.num_inference_steps))
        metadata = {
            "backend": self.name,
            "condition": request.condition,
            "frame_st_id": frame_st_id,
            "action_horizon": self.action_horizon,
            "frame_chunk_size": cfg.frame_chunk_size,
            "action_per_frame": cfg.action_per_frame,
            "video_inference_steps": int(request.num_inference_steps),
            "action_inference_steps": cfg.action_num_inference_steps,
            "video_exec_step": cfg.video_exec_step,
            "guidance_scale": cfg.guidance_scale,
            "seed": request.seed,
            "observation": request.observation.metadata,
        }
        return InferenceResult(
            actions=actions,
            metadata=metadata,
            trace=[] if recorder is None else recorder.events,
        )
