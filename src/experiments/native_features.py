"""Scoped future-content interventions on the four native action-conditioning paths."""
from contextlib import contextmanager


class FeatureCapture:
    def __init__(self, layer=None, donors=None, stage=None):
        self.layer = layer
        self.stage = stage
        self.donors = donors
        self.captured = {}
        self.seen = 0

    def visit(self, parts):
        import torch
        occurrence = self.seen
        self.seen += 1
        if self.stage is not None and occurrence != self.stage:
            return None
        if any(x.numel() == 0 or not torch.isfinite(x).all() for x in parts):
            raise ValueError("Invalid native future features")
        self.captured[occurrence] = tuple(x.detach().to("cpu", copy=True) for x in parts)
        if self.donors is None:
            return None
        values = self.donors[occurrence]
        if len(values) != len(parts):
            raise ValueError("Donor feature count differs")
        for value, target in zip(values, parts):
            if value.shape != target.shape or value.dtype != target.dtype or not torch.isfinite(value).all():
                raise ValueError("Invalid donor layout, precision or values")
        return tuple(x.to(target.device) for x, target in zip(values, parts))


@contextmanager
def native_features(owner, *, layer=None, stage=None, donors=None):
    """Install for one eager predict_action call. The caller pairs context and schedules."""
    if getattr(owner, "_e1_features", None) is not None:
        raise RuntimeError("An E1 hook is already active")
    capture = FeatureCapture(layer, donors, stage)
    owner._e1_features = capture
    try:
        yield capture
        if not capture.captured:
            raise RuntimeError("Requested future stage/layer was never reached")
        if donors is not None and donors.keys() != capture.captured.keys():
            raise ValueError("Donor and recipient stage counts differ")
    finally:
        del owner._e1_features


def edit_future_kv(hook, key, value, first):
    """Token dimension 1, with the observed prefix protected, including CFG batch rows."""
    import torch
    if not 0 <= first < key.shape[1] or value.shape != key.shape:
        raise ValueError("Invalid action-facing K/V layout")
    patched = hook.visit((key[:, first:], value[:, first:]))
    if patched is None:
        return None
    return tuple(torch.cat((old[:, :first], new), dim=1) for old, new in zip((key,value), patched))


def lingbot_predicted_kv(runtime, hook):
    cache = runtime._cache_attentions()[hook.layer].attn_caches[runtime.cache_name]
    ids = cache["id"]
    slots = (cache["mask"] & cache["is_pred"] & (ids == ids[cache["mask"]].max())).nonzero().flatten()
    cfg = runtime.config
    pt, ph, pw = cfg.patch_size
    if pt != 1:
        raise ValueError("Unsupported temporal patch size")
    per_frame = (runtime.latent_height // ph) * (runtime.latent_width // pw)
    if slots.numel() != per_frame * cfg.frame_chunk_size:
        raise ValueError("Predicted slots do not match the current video chunk")
    slots = slots[per_frame:] if runtime.frame_st_id == 0 else slots
    patched = hook.visit(tuple(cache[k][:, slots] for k in ("k","v")))
    if patched is not None:
        for key, value in zip(("k","v"), patched):
            cache[key][:, slots] = value
