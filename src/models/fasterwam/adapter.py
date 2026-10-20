"""Explicit translation of shared requests to Faster-WAM's two native paths."""

from wam_interp.models.registry import register_model_adapter
from wam_interp.models.fastwam.adapter import FASTWAM_PROMPT_TEMPLATE
from wam_interp.tracing.hooks import capture_module_outputs
from wam_interp.types import InferenceResult


@register_model_adapter("fasterwam")
class FasterWAMAdapter:
    name = "fasterwam"

    def __init__(self, model):
        self.model = model

    def predict_action(self, request, recorder=None):
        methods = {
            "one_pass_future_cache": "infer_action_one_pass_future_cache",
            "joint_rollout": "infer_action",
        }
        if request.condition not in methods:
            raise ValueError(f"Faster-WAM condition must be one of {sorted(methods)}; got {request.condition!r}")
        if request.compile_action_infer:
            raise ValueError("Faster-WAM does not support compile_action_infer in this backend")
        if request.num_video_frames < 1 or (request.num_video_frames - 1) % 4:
            raise ValueError("num_video_frames must be positive and satisfy (frames - 1) % 4 == 0")
        if request.action_horizon < 1 or request.num_inference_steps < 1:
            raise ValueError("action_horizon and num_inference_steps must be positive")
        if request.trace_modules and recorder is None:
            from wam_interp.tracing.recorder import TraceRecorder
            recorder = TraceRecorder()
        import torch

        image = request.observation.image.to(device=self.model.device, dtype=self.model.torch_dtype)
        method = getattr(self.model, methods[request.condition])
        with capture_module_outputs(self.model, request.trace_modules, recorder):
            with torch.inference_mode():
                output = method(
                    prompt=FASTWAM_PROMPT_TEMPLATE.format(task=request.observation.prompt),
                    input_image=image, proprio=request.observation.proprio,
                    action_horizon=request.action_horizon, num_video_frames=request.num_video_frames,
                    num_inference_steps=request.num_inference_steps, seed=request.seed,
                    rand_device=request.rand_device, sigma_shift=request.sigma_shift,
                    negative_prompt=request.negative_prompt, text_cfg_scale=request.text_cfg_scale,
                    tiled=request.tiled,
                )
        actions = output["action"]
        if actions.ndim != 2 or actions.shape[0] != request.action_horizon or actions.shape[1] != self.model.action_expert.action_dim:
            raise ValueError(f"Invalid Faster-WAM action shape: {tuple(actions.shape)}")
        if not torch.isfinite(actions).all():
            raise ValueError("Faster-WAM returned nonfinite actions")
        metadata = {
            **getattr(self.model, "_wam_interp_metadata", {}),
            "backend": self.name, "model_class": type(self.model).__name__,
            "condition": request.condition, "inference_method": methods[request.condition],
            "seed": request.seed, "rand_device": request.rand_device,
            "action_horizon": request.action_horizon, "num_video_frames": request.num_video_frames,
            "num_inference_steps": request.num_inference_steps, "sigma_shift": request.sigma_shift,
            "compiled": False, "action_shape": list(actions.shape),
            "action_dtype": str(actions.dtype), "action_device": str(actions.device),
            "model_dtype": str(self.model.torch_dtype), "model_device": str(self.model.device),
            "observation": request.observation.metadata, "model_output_keys": sorted(output),
        }
        return InferenceResult(actions=actions, metadata=metadata,
                               trace=[] if recorder is None else recorder.events)
