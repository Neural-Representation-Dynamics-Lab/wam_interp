# FastWAM backend

This directory contains the FastWAM model implementation used by `wam_interp`, its adapter and loader, and a small set of Hydra factories. The backend lets shared code ask for actions through one model-neutral interface while keeping FastWAM's inputs, model construction, checkpoints, and inference details here.

FastWAM is built on Wan 2.2's video model and adds an action expert. The video side turns the current camera input into latent video tokens. The action side starts from a noisy action chunk and denoises it. Both use the language instruction (and, when supplied, proprioception) as context. The model's attention/cache machinery lets the action stream use video representations. The model variants differ in what video information the action stream can use and whether video is generated at inference time.

## Config names

The `task` passed to `load_model_adapter("fastwam", task=...)` selects a Hydra task config. That config selects both a dataset/input setup and a model variant. It does not, by itself, select a benchmark evaluator or start an environment.

For example, `libero_uncond_2cam224_1e-4` can be read as:

- `libero`: use the LIBERO data configuration.
- `uncond`: use the standard FastWAM model config. In this naming scheme it means action inference without first generating future video; it does **not** mean that the policy ignores the current image, language instruction, or proprioception.
- `2cam`: use the two-camera LIBERO input setup. The two images are combined horizontally by the configured data processor.
- `224`: resize each camera image to 224 by 224 pixels; the model receives the configured two-camera image layout.
- `1e-4`: the training learning rate recorded in that task config. It is part of the checkpoint's experiment identity, not an inference-time setting that changes weights.

The same pieces appear in names such as `robotwin_uncond_3cam_384_1e-4`. Read the actual config before assuming details: camera arrangement, action dimensions, preprocessing, and state representation are configuration choices, not universal meanings of the suffixes.

## Model variants and inference paths

| Config variant | What the local implementation does | Adapter condition |
| --- | --- | --- |
| `uncond` / `FastWAM` | Encodes the current frame once at video timestep zero, caches its video representations, then iteratively denoises the action chunk using that cache. It does not denoise future video at inference. | `first_frame` |
| `idm` / `FastWAMIDM` | First denoises future video latents while restoring the observed first frame at each step. It then freezes the resulting video context and denoises actions using its cached representations. | `idm` |
| `joint` / `FastWAMJoint` | Denoises future video and action latents together over the same inference steps. The joint attention mask determines which tokens can exchange information. | `joint` |
| `optional_idm` / `FastWAMOptionalIDM` | Uses the Optional IDM model, whose inference method delegates to either the standard first-frame path or the IDM future-video path. | `first_frame` or `idm` |

The adapter's `condition` argument selects a path only when the loaded model supports that choice. It does not swap the model architecture selected by the task config. For example, to use both paths with one checkpoint, load an `optional_idm` task and pass the desired condition in each `InferenceRequest`. Fixed `uncond`, `idm`, and `joint` configs accept only their native conditions. The repository's adapter checks this and raises a clear error for unsupported combinations.

The local FastWAM attention mask is also worth knowing about: in the standard `FastWAM` action path, action tokens can attend to other action tokens and the current-frame video tokens. They do not attend to future video tokens. The video stream uses its configured causal mask. This is what allows the standard action path to omit test-time future-video denoising without asking action prediction to consume future-frame tokens.

## What happens during one action prediction

For the standard `uncond` / first-frame path, the local implementation:

1. Receives a resized image, instruction text, optional proprioception, and an action horizon.
2. Encodes the image with the configured video VAE and prepares its video tokens at timestep zero.
3. Encodes the instruction with the configured text encoder and appends proprioception context when provided.
4. Runs the video tokens through the model once to build per-layer key/value caches.
5. Starts with a random action tensor shaped `[batch, horizon, action_dim]` and denoises it for the requested number of inference steps, attending to the cached video representations.
6. Returns the action chunk in the model's normalized representation.

So “one pass” refers to the video-context preparation, not the entire prediction: action denoising still has multiple steps. For `idm`, future video is generated first and then used as frozen context for action denoising. For `joint`, the video and action latents are both updated at each step.

## Where things live

- `adapter.py` translates the shared `InferenceRequest` into FastWAM's `infer_action` arguments and packages the result. It does not implement denoising.
- `loader.py` composes `configs/fastwam/sim_libero.yaml` with the requested task, instantiates its configured model through Hydra, loads the provided checkpoint, and returns the model to the adapter.
- `runtime.py` contains the Hydra `_target_` factories for the four model variants.
- `model_code/fastwam.py`, `fastwam_idm.py`, `fastwam_joint.py`, and `fastwam_optional_idm.py` contain model construction and inference implementations. Supporting Wan layers, schedulers, dataset code, and model-download helpers are also under `model_code/`.
- `configs/fastwam/task/` names task/data/model combinations. `configs/fastwam/model/` describes architecture and pretrained model IDs. `configs/fastwam/data/` describes dataset shapes and preprocessing.

For adapter inference, the loader composes `sim_libero.yaml` and then instantiates the resulting model. That config sets `skip_dit_load_from_pretrain=true` and `action_dit_pretrained_path=null`, so inference does not require a separate ActionDiT initialization file. The Wan 2.2 and text tokenizer/encoder assets are still needed from the configured model cache. After construction, the loader applies the released checkpoint, which supplies the trained model weights.

The individual configs under `configs/fastwam/model/` are also used in training-oriented compositions; they set `skip_dit_load_from_pretrain=false` and point to `checkpoints/ActionDiT_linear_interp_Wan22_alphascale_1024hdim.pt`. That initialization file is not included in this repository. The upstream FastWAM project describes how to generate it in its [Model Preparation instructions](https://github.com/yuantianyuan01/FastWAM#model-preparation). The released model checkpoint and dataset-statistics files are listed at [yuanty/fastwam](https://huggingface.co/yuanty/fastwam). A Colab example is in [`docs/fastwam_colab.md`](../../../docs/fastwam_colab.md).

## Current boundary

The adapter can load a supported FastWAM variant and predict one normalized action chunk from an already prepared observation. This repository does not yet create a LIBERO or RoboTwin environment, run episodes, denormalize actions using dataset statistics, map gripper outputs, or calculate benchmark success rates. The task names describe configs; their presence is not evidence that the corresponding evaluation pipeline exists.

For upstream details and the paper's reported experiments, see the [FastWAM repository](https://github.com/yuantianyuan01/FastWAM). This README describes the implementation and configuration currently included in this checkout; where local behavior matters, the code and configs here are authoritative.
