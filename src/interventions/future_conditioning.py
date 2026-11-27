"""Inference-time ablations of Faster-WAM's future conditioning (Roadmap E7).

The action expert attends to a fused K/V cache over [current frame, future latent frames] at the
condition layers. A mask sets the action rows of those layers' attention masks to False for the
chosen key columns, so those tokens get no attention weight; zeroing their K would still leave them a
logit of 0 and a share of the weight. A partial mask, ``<mask>_<percent>``, removes that share of one
frame's positions. The subset is fixed, nested across percentages and the same within frames 0 and 1,
so ``current_mask_50`` and ``future1_mask_50`` remove matching positions.
``resample`` keeps the current frame and swaps in the future K/V built from other future noise.
"""

from dataclasses import replace
from typing import Any

import torch

FULL_MASKS = {"future_mask": (1, 3), "current_mask": (0, 1), "future1_mask": (1, 2), "future2_mask": (2, 3)}
PARTIAL_PERCENTS = (25, 50, 75, 90, 95)
PARTIAL_MASKS = tuple(f"{name}_{p}" for name in ("current_mask", "future1_mask") for p in PARTIAL_PERCENTS)
MASKS = (*FULL_MASKS, *PARTIAL_MASKS)
CONDITIONS = ("normal", *MASKS, "resample")


class FasterWAMFutureAblation:
    """Patches one Faster-WAM model in place; ``condition`` applies to every later prediction."""

    def __init__(self, model: Any, latent_frames: int, subset_seed: int = 0):
        if latent_frames != 3:
            raise ValueError(f"Masks assume one current and two future latent frames, got {latent_frames} frames")
        self.mot = model.mot
        self.latent_frames = latent_frames
        self.subset_seed = subset_seed
        self.condition = "normal"
        self.donor_future = None
        self.last_cache = None
        self._forward_action = self.mot.forward_action_with_video_cache
        self._prefill = self.mot.prefill_video_cache
        self.mot.forward_action_with_video_cache = self.forward_action
        self.mot.prefill_video_cache = self.prefill

    def columns(self, condition: str, video_seq_len: int):
        """Key columns ``condition`` removes: a slice for whole frames, sorted indices for a partial mask."""
        frame = video_seq_len // self.latent_frames
        if condition in FULL_MASKS:
            first, last = FULL_MASKS[condition]
            return slice(first * frame, last * frame)
        if condition not in PARTIAL_MASKS:
            return None
        name, _, percent = condition.rpartition("_")
        count = int(int(percent) * frame / 100 + 0.5)
        order = torch.randperm(frame, generator=torch.Generator().manual_seed(self.subset_seed))
        return FULL_MASKS[name][0] * frame + order[:count].sort().values

    def forward_action(self, **kwargs):
        video_seq_len = kwargs["video_seq_len"]
        columns = self.columns(self.condition, video_seq_len)
        if columns is not None:
            total = video_seq_len + kwargs["action_tokens"].shape[1]
            masks = self.mot._resolve_attention_masks(kwargs["attention_mask"], expected_seq_len=total)
            for layer in self.mot.condition_layer_set:
                masks[layer] = masks[layer].clone()
                index = columns if isinstance(columns, slice) else columns.to(masks[layer].device)
                masks[layer][video_seq_len:, index] = False
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
