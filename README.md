# WAM Interpretability

This project provides a research scaffold for studying how world-action models (WAMs) use predicted futures when producing actions. It focuses on three questions: what information a future representation contains, whether the action pathway uses that information, and whether doing so improves task performance or robustness. These are separate claims and require separate evidence. Predicted video is treated as a candidate internal representation, not assumed to be a faithful explanation.

FastWAM Optional IDM is the first supported backend. Its `first_frame` and `idm` inference modes make it possible to compare action-only and future-conditioned inference within one model family. That comparison measures a change in inference path; by itself, it does not isolate the causal effect of future content or show that imagination improves behavior. Results from this backend alone do not establish how other WAMs work.

## Repository structure

- `src/` contains the shared request and result types, input handling, model registry, FastWAM adapter, tracing hooks, intervention helpers, and paired-inference utility. The package is installed under the import name `wam_interp`.
- `configs/fastwam/` contains the FastWAM Hydra configuration used by the adapter.
- `docs/` describes the architecture, research questions, and experimental controls.

FastWAM model code is an external runtime dependency and is not bundled in this repository. The FastWAM Python package, its dependencies and pretrained assets, and a compatible checkpoint must be available in the environment used to run inference.

## Install

From the repository root, install the scaffold and its FastWAM-facing dependencies:

```bash
python -m pip install -e '.[fastwam]'
```

This installs the scaffold, PyTorch, Hydra, OmegaConf, and NumPy. It does not install FastWAM, download model assets, or provide a checkpoint or simulator. Follow the FastWAM project’s setup instructions to make its runtime and required assets available in the same Python environment.

## Input format

The initial runner accepts one prepared observation in an `.npz` file with two arrays:

- `image`: a camera-preprocessed RGB image in CHW or HWC layout. `uint8` values in `[0, 255]` are scaled to `[-1, 1]`; floating-point values must already be in `[-1, 1]`. Height and width must be divisible by 16.
- `proprio`: a one-dimensional, already normalized state vector compatible with the selected checkpoint.

The image montage, camera order, image preprocessing, and state normalization must match the selected model and checkpoint. The runner does not apply dataset statistics, denormalize actions, convert gripper values, or step a robot environment.

## Run one observation

With FastWAM and its required assets available, run from the repository root:

```bash
python -m wam_interp.main \
  --task libero_optional_idm_2cam224_1e-4 \
  --checkpoint /path/to/checkpoint.pt \
  --observation /path/to/observation.npz \
  --prompt "pick up the object" \
  --condition first_frame \
  --seed 0 \
  --output results/first_frame.json
```

Use `--condition idm` to select Optional IDM’s future-denoising path. The default video frame count is 33; IDM requires `(num_video_frames - 1) % 4 == 0`. Use `--trace-module` one or more times to capture input and output tensors at named PyTorch module boundaries. Tracing is passive and does not expose all attention or cache internals; see [the architecture notes](docs/architecture.md).

The command performs one offline action inference call. It does not launch LIBERO, run episodes, or report task success. It writes normalized actions and run metadata to the JSON output; when tracing is enabled, it also writes a `.trace.pt` file beside that output.

## Research and evaluation

The project keeps representation content, causal action dependence, and task-level benefit distinct. A difference between inference modes is not sufficient evidence for any one of these claims. Controlled content interventions, suitable negative and compute-matched controls, and environment evaluation are needed for stronger conclusions. See [the research questions](docs/research_questions.md) and [the experiment protocol](docs/experiment_protocol.md) before interpreting results.

The current runner is a single-observation inference tool. It does not provide a LIBERO environment loop, task/trial management, observation collection, action post-processing, or success logging.

## References

- [From World Models to World Action Models: A Concise Tutorial for Robotics](https://arxiv.org/abs/2607.00836)
- [Fast-WAM: Do World Action Models Need Test-time Future Imagination?](https://arxiv.org/abs/2603.16666)
- [Faster-WAM: Efficient Inference-Time Future Conditioning for Robust World Action Models](https://arxiv.org/abs/2608.04404)
- [NVIDIA: Pretrained to Imagine, Fine-Tuned to Act](https://developer.nvidia.com/blog/pretrained-to-imagine-fine-tuned-to-act-the-rise-of-world-action-models/)
