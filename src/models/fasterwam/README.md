# Faster-WAM backend

This backend vendors the inference dependency closure from
[hustvl/FasterWAM](https://github.com/hustvl/FasterWAM) at commit
`83667817df0d4f823f39d90700e61ea2f432ac45`. The original Apache-2.0 license
is retained in `model_code/LICENSE`. Changes to the copied model implementation
are limited to logging import paths and whitespace cleanup; the logging helper uses Python's standard
logger. Original inherited class methods remain available; upstream training
entrypoints, dataset code, and simulator code are not included.

Faster-WAM retains future context during inference. Its SparseActionDiT uses
video-conditioned layers `[0,4,8,12,16,20,24,28]` and lightweight action-only
layers elsewhere. SparseMoT fuses video KV representations using
`interval_weighted_sum`. These are checkpoint architecture choices, not
adapter approximations.

## Loading

Install editable from the repository root with `pip install -e ".[fasterwam]"`.
Use `.[fasterwam,libero]` when also running the shared evaluator. Config files
are resolved from this checkout, matching the existing FastWAM workflow.

```python
from wam_interp.models import load_model_adapter
from wam_interp.data.npz import load_observation_npz
from wam_interp.types import InferenceRequest

adapter = load_model_adapter(
    "fasterwam",
    task="libero_fasterwam_2cam224_1e-4",
    checkpoint="checkpoints/fasterwam_release/libero/step_021700.pt",
    device="cuda",
    dtype="bfloat16",
)
observation = load_observation_npz("observation.npz", "pick up the object")
result = adapter.predict_action(InferenceRequest(
    observation,
    condition="one_pass_future_cache",
    action_horizon=32,
    num_video_frames=9,
    num_inference_steps=10,
    seed=0,
))
```

`repository_root` and `overrides=["model.action_scheduler.infer_shift=..."]`
are optional loader arguments. Missing checkpoints fail before construction.
The inference config enables prompt encoding, skips training-only DiT
initialization, and relies on the released checkpoint for the video/action
weights. Wan VAE, T5, and tokenizer assets still download through the native
helper. Set `DIFFSYNTH_MODEL_BASE_PATH` to choose their cache directory and
`DIFFSYNTH_DOWNLOAD_SOURCE=modelscope`; the converted Wan T5/VAE files are
not public on the Hugging Face Hub.

The NPZ input is the same prepared format as FastWAM: RGB CHW/HWC uint8 or
float `[-1,1]` and a normalized one-dimensional proprio vector. The LIBERO
checkpoint expects a `[3,224,448]` two-camera montage and 8 state values.
Outputs are normalized `[32,7]` float32 CPU actions by the native API, even
when model computation uses bf16. Use a sufficiently large CUDA device for
the full model and activations; the local 4 GB laptop GPU is not the target.

## Conditions and tracing

- `one_pass_future_cache` calls `infer_action_one_pass_future_cache`, the
  released configuration's default. It builds the future KV context once.
- `joint_rollout` calls `infer_action`, which jointly denoises video/actions.

FastWAM's `first_frame` and `idm` conditions are not aliases for these paths.
Frame count must be positive with `(frames-1) % 4 == 0`. Compilation requests
are rejected because this port has no supported compiled inference path.
Native cache inference ignores negative prompt and text guidance arguments,
matching upstream behavior.

`trace_modules` and module-output interventions use the existing scoped hooks.
Functional attention, post-RoPE cache tensors, and intermediate denoising
states are not automatically visible at every module boundary.

## LIBERO

`FasterWAMLiberoAdapter` composes the existing FastWAM-family transforms:
rotate both views by 180 degrees, center crop/resize, concatenate agentview
then wrist horizontally, normalize RGB, convert quaternion to axis-angle,
normalize the 8-D state, and decode actions with the released stats. The
gripper conversion is `1 - 2*gripper` followed by optional binarization.
These formulas match Faster-WAM's upstream LIBERO evaluator.

```python
from omegaconf import OmegaConf
from wam_interp.models.fasterwam.loader import compose_fasterwam_config, load_fasterwam
from wam_interp.models.fasterwam.libero_adapter import FasterWAMLiberoAdapter
from wam_interp.evaluations.LIBERO import LiberoEvalConfig, evaluate_libero

cfg = compose_fasterwam_config()
model = load_fasterwam(cfg, "checkpoints/fasterwam_release/libero/step_021700.pt")
adapter = FasterWAMLiberoAdapter(
    model,
    data_config=OmegaConf.to_container(cfg.data.train, resolve=True),
    dataset_stats="checkpoints/fasterwam_release/libero/dataset_stats.json",
)
result = evaluate_libero(adapter, LiberoEvalConfig(
    num_trials=1,
    task_ids=(0,),
    inference_options={"condition": "one_pass_future_cache"},
))
```

No shared evaluator changes or new episode hooks are required. This backend is
stateless between requests. See `docs/fasterwam_validation.md` for the H100
handoff, numerical comparison, and one-trial evaluation procedure.
