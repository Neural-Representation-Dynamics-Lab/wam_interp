# OpenWAM backend

Backend for [OpenWAM](https://github.com/OpenWAM-Official/OpenWAM) (arXiv 2609.07398). `model_code/` vendors the upstream inference code at commit `3950b6d9`; changes are limited to import paths and trimmed package `__init__`s.

## Assets

```bash
hf download OpenWAM/OpenWAM-Alpha-Sim-LIBERO --revision 110b07e4d5581c81520a77419cbb0e9371df231b \
  --local-dir checkpoints/openwam/libero
```

For RoboTwin use `OpenWAM/OpenWAM-Alpha-Sim-RoboTwin-Full` at `04b96af53eeeb111c64631f822efcbb82f4b186e`. Checkpoint directories are self-contained. OpenWAM needs transformers 5, so install it in its own environment: `pip install -e ".[openwam,libero]"`.

## Inference path

Video latents and a 32-step action chunk are denoised jointly (10 steps) through a shared self-attention. By default the model runs exactly, without upstream's DiT cache or `torch.compile`, so hooks always fire. `load_openwam(..., upstream_optimizations=True)` uses upstream's deploy settings and refuses hooks.

```python
from wam_interp.models.openwam.loader import load_openwam
from wam_interp.models.openwam.libero_adapter import OpenWAMLiberoAdapter

model = load_openwam("checkpoints/openwam/libero", device="cuda")
adapter = OpenWAMLiberoAdapter(model)
```

`condition` selects the cross-modal attention mask (`mutual`, `action_sees_video`, `video_sees_action`, `isolated`). `attention_mask_hook` and `condition_hook` edit the mask or the engine inputs. Trace sub-modules such as `video_backbone.dit.blocks.3.self_attn.q`; whole blocks are bypassed.

LIBERO uses `action_horizon=32, replan_steps=10, max_steps=600` (700 for `libero_10`); RoboTwin executes full chunks as `ee` actions.

`scripts/validate_openwam.py` compares actions with upstream's engine on the same observation.
