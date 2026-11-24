"""Inference-time ablations of Faster-WAM's future conditioning (Roadmap E7).

The action expert attends to a fused K/V cache over [current frame, future latent frames] at the
condition layers. A mask sets the action rows of those layers' attention masks to False for the
chosen key columns, so those tokens get no attention weight; zeroing their K would still leave them a
logit of 0 and a share of the weight.
``resample`` keeps the current frame and swaps in the future K/V built from other future noise.
"""

from dataclasses import replace
from typing import Any

import torch

MASKS = ("future_mask", "current_mask", "future1_mask", "future2_mask")
CONDITIONS = ("normal", *MASKS, "resample")


class FasterWAMFutureAblation:
    """Patches one Faster-WAM model in place; ``condition`` applies to every later prediction."""

    def __init__(self, model: Any, latent_frames: int):
        if latent_frames != 3:
            raise ValueError(f"Masks assume one current and two future latent frames, got {latent_frames} frames")
        self.mot = model.mot
        self.latent_frames = latent_frames
        self.condition = "normal"
        self.donor_future = None
        self.last_cache = None
        self._forward_action = self.mot.forward_action_with_video_cache
        self._prefill = self.mot.prefill_video_cache
        self.mot.forward_action_with_video_cache = self.forward_action
        self.mot.prefill_video_cache = self.prefill

    def masked_span(self, video_seq_len: int) -> tuple[int, int] | None:
        frame = video_seq_len // self.latent_frames
        spans = {"current_mask": (0, frame), "future_mask": (frame, video_seq_len),
                 "future1_mask": (frame, 2 * frame), "future2_mask": (2 * frame, video_seq_len)}
        return spans.get(self.condition)

    def forward_action(self, **kwargs):
        span = self.masked_span(kwargs["video_seq_len"])
        if span is not None:
            video_seq_len = kwargs["video_seq_len"]
            total = video_seq_len + kwargs["action_tokens"].shape[1]
            masks = self.mot._resolve_attention_masks(kwargs["attention_mask"], expected_seq_len=total)
            for layer in self.mot.condition_layer_set:
                masks[layer] = masks[layer].clone()
                masks[layer][video_seq_len:, span[0]:span[1]] = False
            kwargs["attention_mask"] = masks
        return self._forward_action(**kwargs)

    def current_len(self, cache):
        return next(layer for layer in cache if layer is not None)["k"].shape[1] // self.latent_frames

    def future(self, cache):
        """Future positions of each cached layer (None where the layer holds no K/V)."""
        first = self.current_len(cache)
        return [None if layer is None else {"k": layer["k"][:, first:], "v": layer["v"][:, first:]}
                for layer in cache]

    def prefill(self, *args, **kwargs):
        cache = self._prefill(*args, **kwargs)
        self.last_cache = cache
        if self.condition != "resample":
            return cache
        if self.donor_future is None:
            raise RuntimeError("resample needs donor_future from an earlier pass")
        first = self.current_len(cache)
        return [layer if layer is None else
                {**layer, "k": torch.cat([layer["k"][:, :first], donor["k"].to(layer["k"])], dim=1),
                 "v": torch.cat([layer["v"][:, :first], donor["v"].to(layer["v"])], dim=1)}
                for layer, donor in zip(cache, self.donor_future)]


class FutureAblationPolicy:
    """Evaluation adapter that runs one condition and can measure how far probe conditions move the chunk.

    Probes rerun the same request under other conditions and record the mean absolute difference from
    the executed chunk, over the whole chunk and over the steps executed before the next replan. A
    ``normal`` probe measures run-to-run noise. Model-level metadata is dropped from each call's record
    because the evaluator already stores it once per run.
    """

    def __init__(self, adapter: Any, ablation: FasterWAMFutureAblation, condition: str = "normal",
                 probes: tuple[str, ...] = (), executed_steps: int = 10, resample_offset: int = 10_000):
        for name in (condition, *probes):
            if name not in CONDITIONS:
                raise ValueError(f"Unknown condition {name!r}; choose from {CONDITIONS}")
        self.adapter, self.ablation = adapter, ablation
        self.condition, self.probes = condition, tuple(probes)
        self.executed_steps, self.resample_offset = executed_steps, resample_offset
        self.run_level_keys = set(getattr(adapter.model, "_wam_interp_metadata", {}))

    def __getattr__(self, name):
        if name == "adapter":  # not set yet, e.g. while copying
            raise AttributeError(name)
        return getattr(self.adapter, name)

    def run(self, request, condition, recorder=None):
        """Predict under ``condition``; ``resample`` first builds the donor future with seed + resample_offset."""
        if condition == "resample":
            self.ablation.condition = "normal"
            self.adapter.predict_action(replace(request, seed=request.seed + self.resample_offset))
            self.ablation.donor_future = self.ablation.future(self.ablation.last_cache)
        self.ablation.condition = condition
        return self.adapter.predict_action(request, recorder)

    def predict_action(self, request, recorder=None):
        result = self.run(request, self.condition, recorder)
        record = {"condition": self.condition}
        if self.probes:
            diffs = {}
            for probe in self.probes:
                delta = (self.run(request, probe).actions - result.actions).abs().float()
                diffs[probe] = {"chunk": float(delta.mean()), "executed": float(delta[:self.executed_steps].mean())}
            record["action_abs_diff"] = diffs
        result.metadata = {key: value for key, value in result.metadata.items() if key not in self.run_level_keys}
        result.metadata["future_ablation"] = record
        return result
