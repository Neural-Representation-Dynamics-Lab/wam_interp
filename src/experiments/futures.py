"""Opt-in, eager FastWAM IDM boundary instrumentation (no weights on import)."""

from contextlib import contextmanager


BOUNDARY = "fastwam.idm.clean_video_before_action_prefill"


def predict_with_future(adapter, request, donor=None):
    """Run the existing public adapter and return (InferenceResult, CPU latents).

    Decode captured latents with the model's own VAE later, outside the policy
    timing measurement. Float conversion for NPZ storage must be explicit.
    """
    if request.condition != "idm" or request.compile_action_infer:
        raise ValueError("Future extraction requires condition='idm' and compilation disabled")
    with capture_future(adapter.model, donor) as captured:
        result = adapter.predict_action(request)
    result.metadata["research_future_boundary"] = BOUNDARY
    result.metadata["research_future_replaced"] = donor is not None
    return result, captured


def apply_future_hook(model, latents):
    """Called once after video denoising, before action K/V construction.

    Latent time index zero is the observed anchor, not a future video frame.
    Hooks receive a clone so in-place edits cannot destroy the recipient.
    """
    hook = getattr(model, "_research_future_hook", None)
    if hook is None:
        return latents
    import torch

    replacement = hook(latents.detach().clone())
    if replacement is None:
        return latents
    if not isinstance(replacement, torch.Tensor) or replacement.shape != latents.shape:
        raise ValueError("Future replacement must retain the latent tensor shape")
    if replacement.dtype != latents.dtype or replacement.device != latents.device:
        raise ValueError("Future replacement must retain dtype and device")
    if not torch.isfinite(replacement).all():
        raise ValueError("Future replacement must be finite")
    if not torch.equal(replacement[:, :, :1], latents[:, :, :1]):
        raise ValueError("Future replacement changed the observed anchor")
    return replacement


@contextmanager
def capture_future(model, donor=None):
    """Capture CPU latents and optionally replace future time slices only.

    Sequential use only. The model must expose the explicit native boundary.
    Donor semantic/context validation belongs to the paired experiment runner.
    """
    if getattr(model, "research_future_boundary", None) != BOUNDARY:
        raise ValueError("This model has no supported E1 future boundary")
    if getattr(model, "_research_future_hook", None) is not None:
        raise RuntimeError("A future hook is already active")
    captured = {}
    sentinel = object()
    previous = model.__dict__.get("_research_future_hook", sentinel)

    def hook(latents):
        if captured:
            raise RuntimeError("Expected exactly one future boundary per inference")
        captured["original"] = latents.detach().cpu().clone()
        if donor is not None:
            import torch
            replacement = torch.as_tensor(donor, device=latents.device, dtype=latents.dtype)
            if replacement.shape != latents.shape:
                raise ValueError("Donor and recipient latent shapes differ")
            latents[:, :, 1:] = replacement[:, :, 1:]
        captured["used"] = latents.detach().cpu().clone()
        return latents

    model._research_future_hook = hook
    try:
        yield captured
        if not captured:
            raise RuntimeError("Future boundary was not reached; use eager IDM inference")
    finally:
        if previous is sentinel:
            delattr(model, "_research_future_hook")
        else:
            model._research_future_hook = previous
