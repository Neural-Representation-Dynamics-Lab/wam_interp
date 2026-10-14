# Architecture and model integration

## Package boundaries

- main: orchestration only; constructs config, prepares observation, loads selected adapter, runs inference, records outputs.
- data: converts raw or stored observations into explicit model inputs. The first NPZ path accepts already-normalized proprio and camera-preprocessed pixels.
- models: model-neutral protocol and registry. Each model package owns its loader and translates a shared request to the model’s API.
- tracing: records data and passive hooks, without model-specific imports.
- interventions: scoped transformations at named module outputs. Later model backends should expose stronger semantic boundaries through explicit callbacks.
- experiments: paired conditions and future evaluation runners.
- docs: research questions, controls, and provenance requirements.

The research package lives in wam-interp/src/wam_interp. The official FastWAM implementation remains in parent src/fastwam. This avoids shadowing the upstream package and makes other WAM adapters independent.

## FastWAM adapter contract

The loader composes the parent configs/sim_libero.yaml with a selected task group, instantiates cfg.model through the repository’s Hydra target, then calls model.load_checkpoint(checkpoint). It does not reproduce checkpoint mapping, scheduler setup, or model initialization.

FastWAM Optional IDM exposes action_infer_mode values idm and first_frame. idm requires num_video_frames and uses the IDM implementation; first_frame delegates to FastWAM action-only inference. The ordinary FastWAM config supports first_frame but not the IDM selector. The adapter detects the method signature and raises a clear error if an unsupported mode is requested.

The shared observation prompt is a task instruction; the FastWAM adapter applies the same robot-video instruction template used by the parent evaluator. Other adapters may format language differently. Inputs to infer_action are prompt, [3,H,W] image in [-1,1], normalized proprio, horizon, frame count for IDM, steps, seed, and scheduler/guidance options. Output actions remain normalized. This package does not apply dataset stats or environment gripper transforms.

## Trace limitations

Passive PyTorch module hooks capture calls to named modules. FastWAM MoT also invokes projections directly and calls functional scaled-dot-product attention, so wrapper hooks may not run and attention probabilities are not returned. Hooks on Q/K/V projection modules can be useful, but are not post-RoPE cached K/V. The first scaffold deliberately labels these as module outputs. Exact per-layer post-RoPE cache, action-attention, denoising-step and residual-stream traces should be added through explicit optional observer callbacks in FastWAM MoT and IDM code; do not infer them from generic module hooks.

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


