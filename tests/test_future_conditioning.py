"""CPU checks for the Faster-WAM future-conditioning ablations (Roadmap E7)."""

import copy
from dataclasses import dataclass
from types import SimpleNamespace

import pytest
import torch

from wam_interp.interventions.future_conditioning import (
    PARTIAL_PERCENTS, FasterWAMFutureAblation, FutureAblationPolicy,
)
from wam_interp.types import InferenceResult

VIDEO_LEN, ACTION_LEN = 6, 2  # 3 latent frames of 2 tokens


class TinyMoT:
    """Cache at layers 0 and 2: current tokens hold their position, future tokens hold the noise seed."""

    num_layers = 4
    condition_layer_set = frozenset({0, 2})

    def _resolve_attention_masks(self, attention_mask, expected_seq_len):
        return [attention_mask] * self.num_layers if isinstance(attention_mask, torch.Tensor) else list(attention_mask)

    def prefill_video_cache(self, *, seed):
        current = torch.arange(2.0).view(1, 2, 1).expand(1, 2, 3)
        kv = torch.cat([current, torch.full((1, 4, 3), float(seed))], dim=1)
        return [{"k": kv, "v": kv.clone()} if layer in self.condition_layer_set else None
                for layer in range(self.num_layers)]

    def forward_action_with_video_cache(self, *, action_tokens, video_kv_cache, attention_mask, video_seq_len):
        self.seen = {"masks": attention_mask, "cache": video_kv_cache}
        masks = self._resolve_attention_masks(attention_mask, video_seq_len + action_tokens.shape[1])
        return sum(video_kv_cache[layer]["v"][0, masks[layer][video_seq_len, :video_seq_len]].mean()
                   for layer in sorted(self.condition_layer_set))


class TinyAdapter:
    def __init__(self):
        self.model = SimpleNamespace(mot=TinyMoT(), _wam_interp_metadata={"checkpoint": "ckpt", "resolved_config": {}})
        self.base_mask = torch.ones(VIDEO_LEN + ACTION_LEN, VIDEO_LEN + ACTION_LEN, dtype=torch.bool)

    def predict_action(self, request, recorder=None):
        mot = self.model.mot
        cache = mot.prefill_video_cache(seed=request.seed)
        out = mot.forward_action_with_video_cache(action_tokens=torch.zeros(1, ACTION_LEN, 1), video_kv_cache=cache,
                                                  attention_mask=[self.base_mask] * mot.num_layers,
                                                  video_seq_len=VIDEO_LEN)
        return InferenceResult(actions=torch.full((32, 7), float(out)),
                               metadata={"checkpoint": "ckpt", "resolved_config": {}, "seed": request.seed})


@dataclass
class Request:
    seed: int


def make_policy(condition="normal", probes=()):
    adapter = TinyAdapter()
    ablation = FasterWAMFutureAblation(adapter.model, latent_frames=3)
    return adapter, ablation, FutureAblationPolicy(adapter, ablation, condition, probes=probes, resample_offset=100)


@pytest.mark.parametrize("condition, columns", [("future_mask", (2, 6)), ("current_mask", (0, 2)),
                                                ("future1_mask", (2, 4)), ("future2_mask", (4, 6))])
def test_masks_drop_keys_only_from_action_rows_of_condition_layers(condition, columns):
    adapter, _, policy = make_policy(condition)
    policy.predict_action(Request(seed=7))
    masks = adapter.model.mot.seen["masks"]
    for layer, mask in enumerate(masks):
        if layer not in adapter.model.mot.condition_layer_set:
            assert mask is adapter.base_mask
            continue
        assert not mask[VIDEO_LEN:, columns[0]:columns[1]].any()
        assert mask[:VIDEO_LEN].all() and mask[VIDEO_LEN:, VIDEO_LEN:].all()
        kept = [c for c in range(VIDEO_LEN) if not columns[0] <= c < columns[1]]
        assert mask[VIDEO_LEN:, kept].all()
    assert adapter.base_mask.all()


def test_partial_mask_drops_only_its_columns_from_action_rows():
    adapter, ablation, policy = make_policy("future1_mask_50")
    policy.predict_action(Request(seed=7))
    dropped = ablation.columns("future1_mask_50", VIDEO_LEN).tolist()
    assert len(dropped) == 1 and 2 <= dropped[0] < 4
    for layer in adapter.model.mot.condition_layer_set:
        mask = adapter.model.mot.seen["masks"][layer]
        kept = [c for c in range(VIDEO_LEN + ACTION_LEN) if c not in dropped]
        assert not mask[VIDEO_LEN:, dropped].any() and mask[VIDEO_LEN:, kept].all() and mask[:VIDEO_LEN].all()


def test_partial_masks_are_nested_and_match_across_frames():
    _, ablation, _ = make_policy()
    frame = 98
    columns = {(name, p): ablation.columns(f"{name}_{p}", 3 * frame).tolist()
               for name in ("current_mask", "future1_mask") for p in PARTIAL_PERCENTS}
    for p in PARTIAL_PERCENTS:
        assert len(columns["future1_mask", p]) == int(p * frame / 100 + 0.5)
        assert [c + frame for c in columns["current_mask", p]] == columns["future1_mask", p]
        assert all(frame <= c < 2 * frame for c in columns["future1_mask", p])
    for smaller, larger in zip(PARTIAL_PERCENTS, PARTIAL_PERCENTS[1:]):
        assert set(columns["future1_mask", smaller]) < set(columns["future1_mask", larger])


def test_resample_keeps_current_positions_and_swaps_in_donor_future():
    adapter, _, policy = make_policy()
    policy.run(Request(seed=7), "resample")
    for layer in adapter.model.mot.seen["cache"]:
        if layer is not None:
            assert torch.equal(layer["k"][0, :2, 0], torch.tensor([0.0, 1.0]))
            assert (layer["v"][0, 2:] == 107).all()


def test_probes_measure_action_change_and_drop_run_level_metadata():
    _, _, policy = make_policy(probes=("normal", "future_mask"))
    result = policy.predict_action(Request(seed=7))
    record = result.metadata["future_ablation"]
    assert record["condition"] == "normal"
    assert record["action_abs_diff"]["normal"] == {"chunk": 0.0, "executed": 0.0}
    assert record["action_abs_diff"]["future_mask"]["chunk"] > 0
    assert result.metadata["seed"] == 7 and "checkpoint" not in result.metadata
    assert "resolved_config" not in result.metadata


def test_unknown_condition_is_rejected():
    adapter = TinyAdapter()
    with pytest.raises(ValueError, match="Unknown condition"):
        FutureAblationPolicy(adapter, FasterWAMFutureAblation(adapter.model, latent_frames=3), "zero_future")


def test_policy_can_be_deep_copied():
    _, _, policy = make_policy("future_mask")
    assert copy.deepcopy(policy).condition == "future_mask"
