# LingBot-VA backend

Backend for [LingBot-VA](https://github.com/Robbyant/lingbot-va) ("Causal World Modeling for Robot Control", arXiv 2601.21998). The released weights are the shared-backbone version: one Wan2.2-5B-sized transformer (about 5.1B parameters) serves both video and action tokens, with separate action input/output layers. The MoT version described in the paper is not released.

## Assets

Only a LIBERO-Long (`libero_10`) post-trained checkpoint is released for LIBERO:

```bash
hf download robbyant/lingbot-va-posttrain-libero-long \
  --revision 0e89d1e753019988aba484e8da2dc0810e264d9f \
  --include "transformer/*" "vae/*" "text_encoder/*" "tokenizer/*" \
  --local-dir checkpoints/lingbot_va/libero_long
```

The directory is about 24 GB (transformer 10 GB, umT5-XXL text encoder 11 GB, Wan2.2 VAE 3 GB) and is loaded as is. Install with `pip install -e ".[lingbot_va,libero]"` (or `.[all]`). flash-attn is not needed; inference uses PyTorch SDPA.

## Inference path

Per chunk, the model denoises 4 latent video frames (20 steps, text CFG 5), writes their K/V into a per-layer cache flagged as predicted, then denoises 16 actions (4 per latent frame, 50 steps) attending to that cache. After the chunk is executed, the predicted entries are dropped and the observed frames plus the executed actions are written in their place. The cache lives in `transformer.blocks[i].attn1.attn_caches["pos"]`.

`runtime.py` ports upstream `wan_va_server.VA_Server` (commit 7c6ffa9) to a single process, without the websocket server, FSDP, or per-chunk debug dumps. Against `VA_Server` built in-process without torch.distributed, with the same inputs and global RNG state, it gave identical actions over three chunks and two cache updates.

The model is stateful, so only the LIBERO adapter path is supported; the single-observation CLI in `main.py` is not.

## LIBERO protocol

The LIBERO adapter uses the evaluator's optional `begin_episode` and `observe_step` hooks: it resets the cache with the instruction after warm-up and collects every observed frame for the next cache update. It needs the whole chunk executed before replanning. To follow upstream's client:

```python
LiberoEvalConfig(
    task_suite_name="libero_10",
    num_steps_wait=5,
    max_steps=796,          # upstream checks timestep < 800 before each chunk; the last one ends at 801
    camera_resolution=128,
    replan_steps=16,
    action_horizon=16,
    inference_options={"condition": "video", "num_inference_steps": 20},
)
```

Cameras are agentview then wrist, flipped vertically only. Actions come out as LIBERO commands after quantile denormalization; the first chunk skips its first frame, so it has 12 actions. One known difference: our warm-up sends a gripper command of -1, upstream sends 0.

Loading prints that `patch_embedding.weight/bias` were unused; upstream prints the same.
