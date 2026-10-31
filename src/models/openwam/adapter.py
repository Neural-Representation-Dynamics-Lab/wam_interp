"""OpenWAM inference adapter: shared requests -> upstream ``JointInferenceEngine.generate``.

Inputs and outputs stay in model space; ``denormalize`` maps actions back.
"""

from contextlib import contextmanager, nullcontext

from wam_interp.models.registry import register_model_adapter
from wam_interp.tracing.hooks import capture_module_outputs
from wam_interp.types import InferenceResult

MASK_MODES = ("mutual", "action_sees_video", "video_sees_action", "isolated")
CONDITIONS = ("joint",) + MASK_MODES


class _ModelSpaceNormalizer:
    """Pass-through normalize/unnormalize; other attributes (_dst_index, _unify_dim) delegate."""

    def __init__(self, inner):
        self._inner = inner

    def normalize(self, x):
        import numpy as np
        return np.asarray(x, dtype=np.float32)

    def unnormalize(self, x):
        return x

    def __getattr__(self, name):
        return getattr(self._inner, name)


def image_to_pil(image):
    """CHW (or HWC) image in [-1,1] (float) or uint8 -> RGB PIL image on the uint8 grid upstream uses."""
    import numpy as np
    from PIL import Image

    value = image.detach().float().cpu().numpy() if hasattr(image, "detach") else np.asarray(image)
    if value.ndim != 3:
        raise ValueError(f"OpenWAM image must be 3-D, got shape {value.shape}")
    if value.shape[0] == 3 and value.shape[-1] != 3:
        value = np.moveaxis(value, 0, -1)
    if value.dtype != np.uint8:
        if value.min() < -1.001 or value.max() > 1.001:
            raise ValueError("Float OpenWAM images must be scaled to [-1, 1]")
        value = np.clip(np.rint((value.astype(np.float64) + 1.0) * 127.5), 0, 255).astype(np.uint8)
    return Image.fromarray(np.ascontiguousarray(value), mode="RGB")


@register_model_adapter("openwam")
class OpenWAMAdapter:
    """``condition_hook`` edits engine inputs before ``generate``; ``attention_mask_hook`` edits the joint attention mask each denoising step."""

    name = "openwam"

    def __init__(self, model):
        self.model = model
        self.condition_hook = None
        self.attention_mask_hook = None
        self.last_output = None

    @property
    def architecture(self):
        return self.model.architecture

    @property
    def inference_settings(self):
        inference = self.model.cfg.inference
        return {"action_horizon": int(inference.num_frames) - 1, "num_video_frames": int(inference.video_num_frames),
                "num_inference_steps": int(inference.denoise_steps), "height": int(inference.height),
                "width": int(inference.width)}

    def inference_defaults(self):
        settings = self.inference_settings
        # Upstream's policy server never sends a seed, so every chunk uses the engine default 42.
        return {"condition": "joint", "num_video_frames": settings["num_video_frames"],
                "num_inference_steps": settings["num_inference_steps"], "seed": 42}

    @property
    def training_mask_mode(self):
        driver = getattr(self.architecture, "mot_driver", None)
        return getattr(driver, "attention_mask_mode", None)

    @contextmanager
    def _attention_override(self, condition):
        if condition == "joint" and self.attention_mask_hook is None:
            yield
            return
        from .model_code.model.architectures.dual_system.joint_self_attn import DualSystemSelfAttnArchitecture

        if not isinstance(self.architecture, DualSystemSelfAttnArchitecture):
            raise ValueError("Attention-mask overrides need a dual-system joint self-attention checkpoint, "
                             f"got {type(self.architecture).__name__}")
        driver = self.architecture.mot_driver
        original_mode = driver.attention_mask_mode
        if condition != "joint":
            driver.attention_mask_mode = condition
        hook = self.attention_mask_hook
        if hook is not None:
            build = driver._build_attention_mask

            def hooked(s_video, s_action, video_tokens_per_frame, *, device):
                mask = build(s_video, s_action, video_tokens_per_frame, device=device)
                layout = {"s_video": s_video, "s_action": s_action, "video_tokens_per_frame": video_tokens_per_frame,
                          "mode": driver.attention_mask_mode}
                edited = hook(mask.clone(), layout)
                if tuple(edited.shape) != tuple(mask.shape) or edited.dtype != mask.dtype:
                    raise ValueError(f"attention_mask_hook must return a {mask.dtype} mask of shape {tuple(mask.shape)}")
                return edited.to(device=mask.device)

            driver._build_attention_mask = hooked
        try:
            yield
        finally:
            driver.attention_mask_mode = original_mode
            if hook is not None:
                del driver._build_attention_mask

    @contextmanager
    def _model_space_normalizer(self):
        architecture = self.architecture
        original = architecture.normalizer
        if original is not None:
            architecture.normalizer = _ModelSpaceNormalizer(original)
        try:
            yield
        finally:
            architecture.normalizer = original

    def _validate(self, request):
        settings = self.inference_settings
        if self.model._wam_interp_metadata.get("upstream_optimizations") and (
                request.trace_modules or self.condition_hook or self.attention_mask_hook
                or request.condition != "joint"):
            raise ValueError("This model was loaded with upstream_optimizations (DiT cache, compile), which skip "
                             "forwards and bypass hooks; load without it to trace or intervene")
        if request.condition not in CONDITIONS:
            raise ValueError(f"OpenWAM condition must be one of {CONDITIONS}; got {request.condition!r}")
        if request.compile_action_infer:
            raise ValueError("OpenWAM runs eager; compile_action_infer is not supported")
        if request.action_horizon != settings["action_horizon"]:
            raise ValueError(f"OpenWAM predicts {settings['action_horizon']}-step chunks; got "
                             f"action_horizon={request.action_horizon}")
        if request.num_video_frames != settings["num_video_frames"]:
            raise ValueError(f"This checkpoint was trained with {settings['num_video_frames']} video frames; got "
                             f"num_video_frames={request.num_video_frames} (use condition_hook to override deliberately)")
        if request.num_inference_steps < 1:
            raise ValueError("num_inference_steps must be positive")
        if request.rand_device != "cpu":
            raise ValueError("Upstream draws video noise on CPU and action noise on the model device; "
                             "rand_device must be 'cpu'")
        if request.sigma_shift is not None:
            raise ValueError("Use condition_hook with 'shift' / 'shift_video' to change the flow shifts")
        if request.negative_prompt or request.text_cfg_scale != 1.0:
            raise ValueError("Wan-backbone OpenWAM checkpoints run without text CFG")
        if self.model.architecture.uses_proprioception and request.observation.proprio is None:
            raise ValueError("This checkpoint is proprio-conditioned; observation.proprio is required")

    def predict_action(self, request, recorder=None):
        import numpy as np

        self._validate(request)
        if request.trace_modules and recorder is None:
            from wam_interp.tracing.recorder import TraceRecorder
            recorder = TraceRecorder()
        observation = request.observation
        conditions = {"first_frame_image": [image_to_pil(observation.image)], "prompt": observation.prompt,
                      "denoise_steps": int(request.num_inference_steps), "tiled": bool(request.tiled)}
        if request.seed is not None:
            conditions["seed"] = int(request.seed)
        if observation.proprio is not None:
            proprio = observation.proprio
            proprio = proprio.detach().float().cpu().numpy() if hasattr(proprio, "detach") else proprio
            conditions["proprio"] = np.asarray(proprio, dtype=np.float32).reshape(-1)
        if self.condition_hook is not None:
            conditions = self.condition_hook(dict(conditions))
        tracing = capture_module_outputs(self.architecture, request.trace_modules, recorder) \
            if request.trace_modules else nullcontext()
        with self._model_space_normalizer(), self._attention_override(request.condition), tracing:
            output = self.model.engine.generate(conditions)
        self.last_output = output
        actions = np.asarray(output["actions"], dtype=np.float32)
        expected = (int(conditions.get("num_frames", request.action_horizon + 1)) - 1, self.architecture.action_dim)
        if actions.shape != expected:
            raise ValueError(f"Invalid OpenWAM action shape {actions.shape}; expected {expected}")
        if not np.isfinite(actions).all():
            raise ValueError("OpenWAM returned nonfinite actions")
        metadata = {
            "backend": self.name, "architecture": type(self.architecture).__name__,
            "checkpoint": self.model._wam_interp_metadata.get("checkpoint"),
            "upstream_revision": self.model._wam_interp_metadata.get("upstream_revision"),
            "condition": request.condition, "training_attention_mask_mode": self.training_mask_mode,
            "attention_mask_hook": self.attention_mask_hook is not None,
            "condition_hook": self.condition_hook is not None,
            "seed": conditions.get("seed", 42), "action_horizon": actions.shape[0],
            "num_video_frames": int(conditions.get("video_num_frames", request.num_video_frames)),
            "num_inference_steps": int(conditions["denoise_steps"]), "action_shape": list(actions.shape),
            "action_space": "model (normalized)", "model_dtype": str(self.model.dtype),
            "model_device": str(self.model.device), "dit_cache": False, "compiled": False,
            "observation": observation.metadata,
        }
        return InferenceResult(actions=actions, metadata=metadata,
                               trace=[] if recorder is None else recorder.events)

    def denormalize(self, actions):
        """Model-space actions -> raw checkpoint action space (e.g. EEF10 / EEF20), as upstream returns them."""
        import numpy as np

        actions = actions.detach().float().cpu().numpy() if hasattr(actions, "detach") else actions
        actions = np.asarray(actions, dtype=np.float32)
        normalizer = self.architecture.normalizer
        raw = np.asarray(actions if normalizer is None else normalizer.unnormalize(actions), dtype=np.float32)
        # Upstream WAMPolicy.predict_action: legality projection for two-point command dims.
        dims = getattr(self.architecture, "binary_command_dims", ()) or ()
        if dims:
            raw = raw.copy()
            for dim in dims:
                raw[..., dim] = np.where(raw[..., dim] > 0.5, 1.0, -1.0)
        return raw

    def normalize_state(self, state):
        """Raw proprio -> model space (unified for Alpha checkpoints), as upstream's engine does."""
        import numpy as np

        state = np.asarray(state, dtype=np.float32)
        normalizer = self.architecture.normalizer
        return state if normalizer is None else np.asarray(normalizer.normalize(state), dtype=np.float32)
