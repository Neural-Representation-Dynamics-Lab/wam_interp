# Architecture and model integration

## Package boundaries

- main: orchestration only; constructs config, prepares observation, loads selected adapter, runs inference, records outputs.
- data: converts raw or stored observations into explicit model inputs. The first NPZ path accepts already-normalized proprio and camera-preprocessed pixels.
- models: model-neutral protocol and registry. Each model package owns its loader and translates a shared request to the model’s API.
- tracing: records data and passive hooks, without model-specific imports.
- interventions: scoped transformations at named module outputs. Later model backends should expose stronger semantic boundaries through explicit callbacks.
- experiments: paired inference helpers; environment evaluation is not yet implemented.
- docs: research questions, controls, and provenance requirements.

The research package source lives directly in `src/` and is installed under the Python import name `wam_interp`. The FastWAM Hydra configuration used by this project lives in `configs/fastwam`. FastWAM model code is an external runtime dependency; it is not bundled or copied into this repository. This keeps the research tools and model-specific integration separate while allowing other WAM adapters to be added independently.

## FastWAM adapter contract

The loader composes `configs/fastwam/sim_libero.yaml` with a selected task group, instantiates `cfg.model` through the FastWAM Hydra target, then calls `model.load_checkpoint(checkpoint)`. It does not reproduce checkpoint mapping, scheduler setup, or model initialization. The FastWAM package, its dependencies, pretrained assets, and a compatible checkpoint must be available in the runtime environment.

FastWAM Optional IDM exposes action_infer_mode values idm and first_frame. idm requires num_video_frames and uses the IDM implementation; first_frame delegates to FastWAM action-only inference. The ordinary FastWAM config supports first_frame but not the IDM selector. The adapter detects the method signature and raises a clear error if an unsupported mode is requested.

The shared observation prompt is a task instruction; the FastWAM adapter wraps it in FastWAM's robot-video instruction template. Other adapters may format language differently. Inputs to `infer_action` are the prompt, `[3,H,W]` image in `[-1,1]`, normalized proprioception, action horizon, frame count for IDM, inference steps, seed, and scheduler/guidance options. Output actions remain normalized. This package does not apply dataset statistics or environment gripper transforms.

## Trace limitations

Passive PyTorch module hooks capture calls to named modules. FastWAM MoT also invokes projections directly and calls functional scaled-dot-product attention, so wrapper hooks may not run and attention probabilities are not returned. Hooks on Q/K/V projection modules can be useful, but are not post-RoPE cached K/V. The recorder labels captures as module outputs. Exact per-layer post-RoPE cache, action-attention, denoising-step, and residual-stream traces require semantic observer points in the FastWAM runtime; do not infer them from generic module hooks.

Compilation is disabled by the scaffold when tracing is requested because Python hooks and compiled graphs do not compose reliably. Keep an uninstrumented production path for timing.

## Extension contract for another WAM

Implement the ModelAdapter protocol with:
- stable model name and checkpoint/config provenance;
- model-ready observation contract;
- named inference condition/mode support;
- normalized action output plus metadata;
- optional trace recorder integration;
- explicit unsupported-feature errors.

Do not make shared experiment code depend on FastWAM internals. Dataset/environment preprocessing and action denormalization belong to adapters, not the model-neutral runner.


