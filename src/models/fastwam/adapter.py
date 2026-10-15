"""FastWAM inference adapter; does not duplicate FastWAM denoising code."""

from inspect import signature
from typing import Any

from wam_interp.models.registry import register_model_adapter
from wam_interp.types import InferenceResult
from wam_interp.tracing.hooks import capture_module_outputs

FASTWAM_PROMPT_TEMPLATE = "A video recorded from a robot's point of view executing the following instruction: {task}"


@register_model_adapter("fastwam")
class FastWAMAdapter:
    name = "fastwam"

    def __init__(self, model: Any):
        self.model = model

    def predict_action(self, request, recorder=None):
        if request.trace_modules and recorder is None:
            from wam_interp.tracing.recorder import TraceRecorder
            recorder = TraceRecorder()
        method = self.model.infer_action
        parameters = signature(method).parameters
        if request.compile_action_infer and (recorder or request.trace_modules):
            raise ValueError("Disable compilation when capturing Python-level traces")
        kwargs = {
            "prompt": FASTWAM_PROMPT_TEMPLATE.format(task=request.observation.prompt),
            "input_image": request.observation.image,
            "action_horizon": request.action_horizon,
            "proprio": request.observation.proprio,
            "negative_prompt": request.negative_prompt,
            "text_cfg_scale": request.text_cfg_scale,
            "num_inference_steps": request.num_inference_steps,
            "sigma_shift": request.sigma_shift,
            "seed": request.seed,
            "rand_device": request.rand_device,
            "tiled": request.tiled,
            "compile_action_infer": request.compile_action_infer,
        }
        if "num_video_frames" in parameters:
            kwargs["num_video_frames"] = request.num_video_frames
        if "action_infer_mode" in parameters:
            if request.condition not in {"idm", "first_frame"}:
                raise ValueError("FastWAM Optional IDM conditions are 'idm' and 'first_frame'")
            if request.condition == "idm" and (request.num_video_frames - 1) % 4:
                raise ValueError("FastWAM IDM requires num_video_frames satisfying (frames - 1) % 4 == 0")
            kwargs["action_infer_mode"] = request.condition
        elif "num_video_frames" in parameters:
            # Fixed IDM and Joint models always use their native future-conditioned path.
            # Optional IDM exposes action_infer_mode above and is handled separately.
            native_condition = "joint" if type(self.model).__name__ == "FastWAMJoint" else "idm"
            if request.condition != native_condition:
                raise ValueError(
                    f"{type(self.model).__name__} uses its native '{native_condition}' condition; "
                    f"got {request.condition!r}."
                )
            if (request.num_video_frames - 1) % 4:
                raise ValueError(
                    "FastWAM video-conditioned inference requires "
                    "num_video_frames satisfying (frames - 1) % 4 == 0"
                )
        elif request.condition != "first_frame":
            raise ValueError(
                f"{type(self.model).__name__} supports only its native 'first_frame' condition; "
                f"got {request.condition!r}."
            )
        kwargs = {key: value for key, value in kwargs.items() if key in parameters}

        with capture_module_outputs(self.model, request.trace_modules, recorder):
            import torch
            with torch.inference_mode():
                output = method(**kwargs)
        metadata = {
            "backend": self.name,
            "model_class": type(self.model).__name__,
            "condition": request.condition,
            "action_horizon": request.action_horizon,
            "num_video_frames": request.num_video_frames,
            "num_inference_steps": request.num_inference_steps,
            "seed": request.seed,
            "compiled": request.compile_action_infer,
            "observation": request.observation.metadata,
            "model_output_keys": sorted(output.keys()),
        }
        return InferenceResult(
            actions=output["action"],
            metadata=metadata,
            trace=[] if recorder is None else recorder.events,
        )

