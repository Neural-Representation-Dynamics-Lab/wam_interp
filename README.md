# WAM Interpretability

A modular research scaffold for testing whether and how imagined futures causally influence WAM action policies. The core question is faithfulness: when a WAM predicts or represents a future, does the action policy use that content as an internal reasoning trace, and which parts of it matter?

The project is model-neutral at its interfaces. FastWAM is the first concrete adapter because this checkout includes FastWAM and an optional-IDM model that exposes both IDM and first-frame action inference modes. This is an initial testbed, not evidence that results generalize to all WAMs. We must add independent model adapters before making cross-family claims.

## Research questions

1. Are actions causally dependent on imagined future content, or does future processing help through extra computation, learned registers, or another mechanism?
2. At which denoising steps, layers, and tokens does the action pathway read future representations?
3. Which future content matters: gripper pose, contact, object motion, or coarse scene layout?
4. What happens when the future is truncated, corrupted, swapped between matched scenes, or replaced with a content-destroyed control?
5. Can a validated mechanism support adaptive imagination, using it when the current observation is out of distribution?

Treat imagined video as a candidate reasoning trace, not as faithful reasoning by definition. Compare models and inference modes with matched prompts, observations, seeds, normalization, checkpoints, and evaluation conditions. Keep in-distribution performance, OOD robustness, latency, and intervention effects as separate outcomes.

## First implementation target

The FastWAM Optional IDM config supports inference modes **idm** (denoise future video, then condition action prediction on it) and **first_frame** (use the Fast-WAM action-only path). This is useful for a controlled initial contrast. The FastWAM source has optimized action inference and IDM teacher-forcing paths; the adapter preserves their existing implementations rather than reimplementing model math.

The initial runner consumes a preprocessed NPZ observation: image in CHW or HWC layout and uint8 [0,255] or float [-1,1], plus already normalized proprioception. It returns normalized model actions. Environment-specific transforms, action denormalization, gripper mapping, and robot stepping belong in separate adapters and are not silently applied by this runner.

## Layout

- **src/wam_interp/main.py**: orchestration; compose config, prepare input, load model, run inference, save outputs.
- **src/wam_interp/data/**: model-ready observations and input loading.
- **src/wam_interp/models/**: model-neutral interface/registry and FastWAM adapter/loader.
- **src/wam_interp/tracing/**: passive module-output capture and trace records.
- **src/wam_interp/interventions/**: scoped module-output intervention primitives.
- **src/wam_interp/experiments/**: paired-condition execution helpers.
- **docs/**: research framing, architecture boundaries, reproducibility protocol.

FastWAM itself remains under the parent repository’s src/fastwam package. The scaffold is installed separately and imports FastWAM lazily.

## Setup

From the repository root, install the scaffold in editable mode with the FastWAM extra:

~~~bash
python -m pip install -e 'wam-interp[fastwam]'
~~~

The parent repository’s dependencies and model assets are still required to load FastWAM. The extra declares the adapter-facing runtime dependencies; it does not install simulator assets or download a checkpoint.

## Input format and first run

Create an NPZ with keys **image** and **proprio**. Image must be camera-preprocessed, in CHW or HWC layout; uint8 is scaled to [-1,1], while floating image data must already lie in [-1,1]. Proprio must be a 1-D, already normalized state vector compatible with the selected checkpoint. Supply the same image montage and processor normalization used by the evaluator.

~~~bash
python -m wam_interp.main \
  --task libero_optional_idm_2cam224_1e-4 \
  --checkpoint /path/to/checkpoint.pt \
  --observation /path/to/observation.npz \
  --prompt "pick up the object" \
  --condition first_frame \
  --seed 0 \
  --output results/first_frame.json
~~~

Use **--condition idm** for optional future denoising. IDM requires a valid num_video_frames value (default 33). Set **--trace-module** to a full PyTorch module path to capture its input/output. Module hooks are passive and do not expose every internal attention/cache boundary; see docs/architecture.md.

This first CLI path does offline single-observation inference. It does not launch LIBERO or claim environment success. Checkpoint/config compatibility, model downloads, CUDA, and the provided preprocessed inputs must be valid.

## References

- [From World Models to World Action Models: A Concise Tutorial for Robotics](https://arxiv.org/abs/2607.00836)
- [Fast-WAM: Do World Action Models Need Test-time Future Imagination?](https://arxiv.org/abs/2603.16666)
- [Faster-WAM: Efficient Inference-Time Future Conditioning for Robust World Action Models](https://arxiv.org/abs/2608.04404)
- [NVIDIA: Pretrained to Imagine, Fine-Tuned to Act](https://developer.nvidia.com/blog/pretrained-to-imagine-fine-tuned-to-act-the-rise-of-world-action-models/)