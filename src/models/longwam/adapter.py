"""Long-WAM inference adapter: shared requests -> ``LongWAMBase.infer_joint_ar`` (IDM or CodeDenoise, fixed by the checkpoint)."""

from typing import Any

from wam_interp.models.registry import register_model_adapter
from wam_interp.tracing.hooks import capture_module_outputs
from wam_interp.types import InferenceResult

LONGWAM_PROMPT_TEMPLATE = "A video recorded from a robot's point of view executing the following instruction: {task}"
CONDITIONS = ("idm", "codenoise")


@register_model_adapter("longwam")
class LongWAMAdapter:
    """Stateless; ``observation.image`` is the current frame ``[1,3,H,W]`` or the history window ``[1,3,T,H,W]`` ending at it."""

    name = "longwam"

    def __init__(self, model: Any):
        self.model = model
        self._text_contexts: dict[str, Any] = {}

    @property
    def native_condition(self) -> str:
        return "codenoise" if bool(getattr(self.model, "joint_denoise", False)) else "idm"

    def inference_defaults(self) -> dict[str, Any]:
        """Checkpoint regime plus the bundle's ``eval_num_inference_steps`` when recorded."""
        defaults: dict[str, Any] = {"condition": self.native_condition}
        steps = (getattr(self.model, "_wam_interp_metadata", None) or {}).get("evaluation", {}).get("num_inference_steps")
        if steps is not None:
            defaults["num_inference_steps"] = int(steps)
        return defaults

    def text_context(self, prompt: str):
        """Encode each templated prompt once (upstream ``TextConditioning`` without a disk cache)."""
        if prompt not in self._text_contexts:
            if getattr(self.model, "text_encoder", None) is None:
                raise RuntimeError("Long-WAM needs its T5 text encoder; load with load_text_encoder=True")
            import torch
            with torch.no_grad():
                self._text_contexts[prompt] = self.model.encode_prompt(prompt)
        return self._text_contexts[prompt]

    def _split_observation_image(self, image):
        """Return (current [1,3,H,W], window [1,3,T,H,W] or None) for this checkpoint."""
        clean_frames = int(self.model.num_clean_frames)
        if image.ndim == 3:
            image = image.unsqueeze(0)
        if image.ndim == 5:
            if clean_frames == 1:
                raise ValueError("This Long-WAM checkpoint has no history; pass the current frame [1,3,H,W]")
            expected = (clean_frames - 1) * int(self.model.vae.temporal_downsample_factor) + 1
            if image.shape[0] != 1 or image.shape[1] != 3 or image.shape[2] != expected:
                raise ValueError(f"Observation window must be [1,3,{expected},H,W], got {tuple(image.shape)}")
            return image[:, :, -1], image
        if image.ndim != 4 or image.shape[:2] != (1, 3):
            raise ValueError(f"Observation image must be [1,3,H,W] or [1,3,T,H,W], got {tuple(image.shape)}")
        if clean_frames != 1:
            raise ValueError(
                f"This checkpoint conditions on {clean_frames} clean history latents; pass the sampled "
                "[1,3,T,H,W] observation window or use a benchmark adapter with begin_episode/observe_step"
            )
        return image, None

    def predict_action(self, request, recorder=None):
        input_image, obs_window = self._split_observation_image(request.observation.image)
        return self._infer(request, input_image=input_image, obs_window=obs_window, recorder=recorder)

    def _infer(self, request, *, input_image, obs_window, recorder=None, history=None):
        if request.condition not in CONDITIONS:
            raise ValueError(f"Long-WAM conditions are {CONDITIONS}; got {request.condition!r}")
        if request.condition != self.native_condition:
            raise ValueError(
                f"This checkpoint was trained for {self.native_condition!r} denoising; requested "
                f"{request.condition!r}. Load the matching IDM/COD bundle instead of switching regimes."
            )
        if request.compile_action_infer:
            raise ValueError("Long-WAM reference inference has no compiled path in this backend")
        if request.action_horizon < 1 or request.num_inference_steps < 1:
            raise ValueError("action_horizon and num_inference_steps must be positive")
        if request.text_cfg_scale != 1.0 or request.negative_prompt:
            raise ValueError("Long-WAM infer_joint_ar implements no text guidance; use text_cfg_scale=1 and no negative_prompt")
        if int(getattr(self.model, "num_imagine_frames", 0)) <= 0:
            raise ValueError("The Long-WAM IDM/COD adapter requires a checkpoint with imagined future latents")
        if request.trace_modules and recorder is None:
            from wam_interp.tracing.recorder import TraceRecorder
            recorder = TraceRecorder()
        import torch

        prompt = LONGWAM_PROMPT_TEMPLATE.format(task=request.observation.prompt)
        context, context_mask = self.text_context(prompt)
        with capture_module_outputs(self.model, request.trace_modules, recorder):
            with torch.no_grad():
                output = self.model.infer_joint_ar(
                    prompt=None, context=context, context_mask=context_mask,
                    input_image=input_image, obs_window=obs_window,
                    proprio=request.observation.proprio, action_horizon=request.action_horizon,
                    num_inference_steps=request.num_inference_steps, sigma_shift=request.sigma_shift,
                    seed=request.seed, rand_device=request.rand_device, tiled=request.tiled,
                    negative_prompt=request.negative_prompt, text_cfg_scale=request.text_cfg_scale,
                )
        actions = output["action"]
        action_dim = int(self.model.action_expert.action_dim)
        if tuple(actions.shape) != (request.action_horizon, action_dim):
            raise ValueError(f"Invalid Long-WAM action shape {tuple(actions.shape)}")
        if not torch.isfinite(actions).all():
            raise ValueError("Long-WAM returned nonfinite actions")
        joint = bool(self.model.joint_denoise)
        metadata = {
            "backend": self.name, "model_class": type(self.model).__name__,
            "condition": request.condition, "inference_method": "infer_joint_ar",
            "action_horizon": request.action_horizon,
            "trained_action_chunk": (getattr(self.model, "_wam_interp_metadata", None) or {}).get("data_config", {}).get("action_chunk"),
            "action_inference_steps": request.num_inference_steps,
            "video_inference_steps": request.num_inference_steps if joint else int(self.model.imagine_infer_steps),
            "num_clean_frames": int(self.model.num_clean_frames),
            "num_imagine_frames": int(self.model.num_imagine_frames),
            "imagine_sigma": float(self.model.imagine_timestep) / float(self.model.train_video_scheduler.num_train_timesteps),
            "sigma_shift": request.sigma_shift, "seed": request.seed, "rand_device": request.rand_device,
            "tiled": request.tiled, "prompt": prompt, "text_context_shape": list(context.shape),
            "input_image_shape": list(input_image.shape),
            "observation_window_shape": None if obs_window is None else list(obs_window.shape),
            "history": history, "action_shape": list(actions.shape), "action_dtype": str(actions.dtype),
            "model_dtype": str(self.model.torch_dtype), "model_device": str(self.model.device),
            "observation": request.observation.metadata,
        }
        return InferenceResult(actions=actions, metadata=metadata,
                               trace=[] if recorder is None else recorder.events)
