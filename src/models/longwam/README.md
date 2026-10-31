# Long-WAM backend

Backend for [Long-WAM](https://github.com/NVlabs/LongLive/tree/main/Long-WAM) (arXiv 2610.10528). `model_code/` vendors the upstream inference code at commit `242635807b4e`; changes are limited to import paths and dropped unused helpers.

## Assets

Released bundles (`config.yaml`, `model.pt`, `dataset_stats.json`) are pinned in `loader.RELEASED_BUNDLES`:

```bash
hf download Efficient-Large-Model/Long-WAM-LIBERO-IDM --revision eca2710140fc0d03a2b0259db1d49d5ca60d5122 \
  --local-dir checkpoints/longwam/libero_idm
```

The Wan VAE, T5 and tokenizer files are downloaded into `DIFFSYNTH_MODEL_BASE_PATH` on first load (revisions in `loader.BASE_ASSETS`). Install with `pip install -e ".[longwam,libero]"`.

## Inference path

Each call encodes a 48-step observation history into 4 video latents, imagines 2 future latents, and decodes a 32-step action chunk from them (IDM bundles; COD bundles denoise video and actions jointly). The benchmark adapters keep the history through `begin_episode` / `observe_step`. Video K/V for interventions comes from `model.mot.prefill_video_cache`.

```python
from wam_interp.models.longwam.loader import load_longwam
from wam_interp.models.longwam.libero_adapter import LongWAMLiberoAdapter

model = load_longwam("checkpoints/longwam/libero_idm/config.yaml", "checkpoints/longwam/libero_idm/model.pt")
adapter = LongWAMLiberoAdapter(model, dataset_stats="checkpoints/longwam/libero_idm/dataset_stats.json")
```

LIBERO uses `action_horizon=32, replan_steps=10`; RoboTwin replans every 24 steps.

`scripts/validate_longwam.py` runs one episode through upstream's runtime and this adapter and compares the actions.
