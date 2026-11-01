"""LingBot-VA: predictions between two cache updates start from the same state.

A tiny real WanTransformer3DModel on CPU with a fake streaming VAE. The cache holds two chunks,
so after a few chunks every prediction evicts real K/V; that is the case a plain "clear the
predicted K/V" fix gets wrong.
"""

import copy
from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("diffusers")

from wam_interp.models.lingbot_va.adapter import LingBotVAAdapter  # noqa: E402
from wam_interp.models.lingbot_va.model_code.model import WanTransformer3DModel  # noqa: E402
from wam_interp.models.lingbot_va.model_code.scheduler import FlowMatchScheduler  # noqa: E402
from wam_interp.models.lingbot_va.runtime import LingBotVARuntime  # noqa: E402
from wam_interp.types import InferenceRequest, PreparedObservation  # noqa: E402

CHUNK = 2 * 4  # frame_chunk_size * action_per_frame


class FakeStreamingVAE:
    """Output depends on what was encoded before, like the real temporal cache."""

    def __init__(self):
        generator = torch.Generator().manual_seed(1)
        self.weight = torch.randn(96, 3, generator=generator)
        self.clear_cache()

    def clear_cache(self):
        self.feat_cache = [None]

    def encode_chunk(self, x):  # [cams, 3, T, H, W] -> [cams, 96, T // 4 or 1, H / 16, W / 16]
        x = x[:, :, ::4]
        pooled = torch.nn.functional.avg_pool3d(x, (1, 16, 16))
        out = torch.einsum("oc,ncthw->nothw", self.weight, pooled)
        if self.feat_cache[0] is not None:
            out = out + 0.5 * self.feat_cache[0].mean()
        self.feat_cache[0] = x[:, :, -1:].clone()
        return out


def tiny_runtime():
    torch.manual_seed(0)
    config = SimpleNamespace(
        frame_chunk_size=2, action_per_frame=4, patch_size=[1, 2, 2], action_dim=30,
        guidance_scale=2.0, action_guidance_scale=1.0, num_inference_steps=3,
        action_num_inference_steps=3, video_exec_step=-1, snr_shift=5.0, action_snr_shift=1.0,
        attn_window=4, env_type="libero", obs_cam_keys=["a", "b"], height=32, width=32,
        norm_stat={"q01": [-1.0] * 30, "q99": [1.0] * 30},
        used_action_channel_ids=list(range(7)), inverse_used_action_channel_ids=list(range(7)) + [7] * 23)
    rt = LingBotVARuntime.__new__(LingBotVARuntime)
    rt.config, rt.device, rt.dtype = config, torch.device("cpu"), torch.float32
    rt.scheduler = FlowMatchScheduler(shift=config.snr_shift, sigma_min=0.0, extra_one_step=True)
    rt.action_scheduler = FlowMatchScheduler(shift=config.action_snr_shift, sigma_min=0.0, extra_one_step=True)
    rt.scheduler.set_timesteps(1000, training=True)
    rt.action_scheduler.set_timesteps(1000, training=True)
    rt.vae = SimpleNamespace(config=SimpleNamespace(latents_mean=[0.0] * 48, latents_std=[1.0] * 48))
    rt.streaming_vae, rt.streaming_vae_half = FakeStreamingVAE(), None
    rt.transformer = WanTransformer3DModel(
        patch_size=[1, 2, 2], num_attention_heads=2, attention_head_dim=12, in_channels=48,
        out_channels=48, action_dim=30, text_dim=16, freq_dim=16, ffn_dim=32, num_layers=2,
        attn_mode="torch").eval().requires_grad_(False)
    rt.actions_q01 = torch.tensor(config.norm_stat["q01"]).reshape(-1, 1, 1)
    rt.actions_q99 = torch.tensor(config.norm_stat["q99"]).reshape(-1, 1, 1)
    rt.action_mask = torch.zeros([config.action_dim]).bool()
    rt.action_mask[config.used_action_channel_ids] = True
    rt.frame_st_id, rt.init_latent = 0, None
    rt.prompt_embeds = rt.negative_prompt_embeds = None
    rt._t5_embeds = lambda prompt: torch.full((1, 8, 16), float(len(prompt)) / 10)
    return rt


def frame(seed):
    return torch.rand(2, 3, 32, 32, generator=torch.Generator().manual_seed(seed)) * 2 - 1


def request(seed, observation_seed=0):
    return InferenceRequest(PreparedObservation(frame(observation_seed), None, "task"), condition="video",
                            action_horizon=CHUNK, num_inference_steps=3, seed=seed)


def advance(adapter, chunks, start=0):
    """Run chunks the way the evaluators do: predict, execute, write the real frames."""
    for k in range(start, start + chunks):
        actions = adapter.predict_action(request(seed=k, observation_seed=k)).actions
        executed = adapter.model.postprocess_action(actions)
        adapter.update([frame(100 * k + i) for i in range(CHUNK)], executed)


@pytest.mark.parametrize("chunks_before", [0, 1, 4])  # first chunk, cache filling, cache full (evicting)
def test_repeat_prediction_is_exact(chunks_before):
    adapter = LingBotVAAdapter(tiny_runtime())
    adapter.reset("task")
    advance(adapter, chunks_before)
    if chunks_before == 4:
        mask = adapter.model._cache_attentions()[0].attn_caches["pos"]["mask"]
        assert bool(mask.all()), "the test needs a full cache, so predictions evict real K/V"
    first = adapter.predict_action(request(seed=7)).actions
    other = adapter.predict_action(request(seed=8)).actions
    again = adapter.predict_action(request(seed=7)).actions
    assert not torch.equal(first, other)
    assert torch.equal(first, again)


def test_repeats_leave_the_episode_unchanged():
    """Extra predictions before an update do not change anything that follows."""
    base = LingBotVAAdapter(tiny_runtime())
    base.reset("task")
    advance(base, 4)
    probed = LingBotVAAdapter(copy.deepcopy(base.model))
    for seed in (11, 12, 13):
        probed.predict_action(request(seed=seed))
    for adapter in (base, probed):
        actions = adapter.predict_action(request(seed=5)).actions
        adapter.update([frame(900 + i) for i in range(CHUNK)], adapter.model.postprocess_action(actions))
    after = [adapter.predict_action(request(seed=6)).actions for adapter in (base, probed)]
    assert torch.equal(after[0], after[1])


def test_lone_predictions_match_the_runtime_without_checkpoints():
    """Checkpointing changes no output when each state is predicted once (the evaluation path)."""
    adapter = LingBotVAAdapter(tiny_runtime())
    plain = copy.deepcopy(adapter.model)
    adapter.reset("task")
    plain.reset("task")
    for k in range(5):
        ours = adapter.predict_action(request(seed=k, observation_seed=k)).actions
        generator = torch.Generator().manual_seed(k)
        theirs, _ = plain.infer_chunk([frame(k)] if plain.frame_st_id == 0 else None,
                                      generator=generator, num_inference_steps=3)
        assert torch.equal(ours, theirs)
        frames = [frame(100 * k + i) for i in range(CHUNK)]
        adapter.update(frames, adapter.model.postprocess_action(ours))
        plain.compute_kv_cache(frames, plain.postprocess_action(theirs))


def test_without_restore_the_cache_leaks_between_predictions():
    """The original behaviour, kept as a check that the tests above can see the difference."""
    rt = tiny_runtime()
    rt.reset("task")
    adapter = LingBotVAAdapter(rt)
    advance(adapter, 4)
    first, _ = rt.infer_chunk(None, generator=torch.Generator().manual_seed(7), num_inference_steps=3)
    again, _ = rt.infer_chunk(None, generator=torch.Generator().manual_seed(7), num_inference_steps=3)
    assert not torch.equal(first, again)


def test_libero_adapter_allows_repeats_and_keeps_first_chunk_trim():
    from wam_interp.models.lingbot_va.libero_adapter import LingBotVALiberoAdapter
    from wam_interp.types import EnvironmentObservation

    rng = np.random.default_rng(0)
    observation = EnvironmentObservation(
        cameras={"agentview": rng.integers(0, 255, (64, 64, 3), dtype=np.uint8),
                 "wrist": rng.integers(0, 255, (64, 64, 3), dtype=np.uint8)},
        state={}, instruction="task", metadata={})
    adapter = LingBotVALiberoAdapter(tiny_runtime())
    adapter.begin_episode(observation)
    prepared = adapter.prepare_observation(observation)
    req = InferenceRequest(prepared, condition="video", action_horizon=CHUNK, num_inference_steps=3, seed=3)
    commands = [adapter.postprocess_actions(adapter.predict_action(req).actions) for _ in range(2)]
    assert commands[0].shape == commands[1].shape == (CHUNK - 4, 7)  # first chunk drops one frame
    np.testing.assert_array_equal(commands[0], commands[1])
