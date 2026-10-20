# Architecture and model integration

## The shared inference boundary

`src/types.py` defines three model-facing objects and one raw environment object:

- `EnvironmentObservation` carries named raw cameras, named raw state values, the task instruction, and benchmark metadata. It is not resized or normalized.
- `PreparedObservation` carries the image and state after the selected model's preprocessing, plus the prompt and preprocessing metadata.
- `InferenceRequest` carries a prepared observation, condition, action horizon, inference settings, seed, and optional module paths to trace.
- `InferenceResult` carries a normalized action chunk, model provenance, and optional trace events.

`src/models/base.py` defines `ModelAdapter` for inference and `EvaluationModelAdapter` for closed-loop evaluations. The evaluation extension adds `inference_defaults()`, `prepare_observation()`, and `postprocess_actions()`. Those methods live with each model because request defaults, camera layout, image transforms, proprio statistics, action statistics, and gripper conventions are checkpoint-specific. Caller-provided inference options override the adapter defaults.

## Where code lives

- `src/evaluations/<benchmark>/`: benchmark tasks, simulator setup, reset/step loop, replanning, metrics, and benchmark results. No model loading or model-specific transforms.
- `src/models/<backend>/`: checkpoint/model loader, inference adapter, model-specific input transforms, normalization, and environment action decoding.
- `src/models/registry.py`: model loader and inference-adapter registries. Importing `wam_interp.models` imports the built-in FastWAM and LingBot-VA modules so their registration decorators execute.
- `configs/<backend>/`: backend configuration files when needed. FastWAM uses Hydra; other backends need not.
- `src/data/`: helpers for stored/prepared observations such as NPZ inputs; not a live simulator loader.
- `src/tracing/`: passive named-module output capture.
- `src/interventions/`: scoped edits at named module outputs.

The shared evaluation package uses lazy imports for simulator dependencies. Importing `wam_interp.evaluations.LIBERO` does not create a simulator or load a checkpoint. `evaluate_libero()` only imports LIBERO when it needs the suite and environment.

## Live LIBERO call flow

The current entry point is `src/evaluations/LIBERO/runner.py:evaluate_libero`. It receives a ready evaluation adapter and orchestrates:

1. Resolve the suite and task objects from LIBERO's benchmark registry.
2. Get fixed initial states for the selected task.
3. Create a task's `OffScreenRenderEnv` from the task's BDDL file.
4. Reset to an initial state and advance the configured no-op warm-up steps.
5. Convert the raw simulator observation to shared `EnvironmentObservation`.
6. Call `model.prepare_observation(raw)`; construct `InferenceRequest`; call `model.predict_action(request)`.
7. Call `model.postprocess_actions(result.actions, environment="libero")`.
8. Execute only the configured replanning prefix, observe again, and repeat until success, environment termination, or the step limit.
9. Return per-trial results and optionally write JSON/actions/videos.

The LIBERO frontend only names and packages environment-provided cameras and state. It does not resize, rotate, normalize, concatenate, denormalize, or map gripper values. The evaluator validates the environment's seven-value action shape, but delegates the actual command conversion to the adapter.

## FastWAM checkpoint path and adapter path

Model construction is a separate operation from evaluation. For FastWAM, `src/models/fastwam/loader.py:compose_fastwam_config()` composes the repository's `configs/fastwam/sim_libero.yaml` and selected task/model config. `load_fastwam()` instantiates the configured FastWAM model, applies the supplied checkpoint via `load_checkpoint()`, moves it to the requested device, and switches it to evaluation mode. The registry helper `load_model_adapter("fastwam", ...)` is a convenience for the generic inference adapter, but it does not provide LIBERO-specific transforms by itself.

For LIBERO, the caller constructs `FastWAMLiberoAdapter` in `src/models/fastwam/libero_adapter.py`, passing:

- the already-loaded raw model;
- resolved `cfg.data.train` as `data_config`;
- the matching dataset-statistics JSON or its parsed mapping.

That adapter composes the ordinary `FastWAMAdapter` with the missing evaluation-boundary transforms. In `prepare_observation()`, it flips each raw LIBERO camera image by 180 degrees (to match FastWAM's training convention), center-crops/resizes each camera according to `shape_meta.images`, concatenates views according to `concat_multi_camera`, makes `[1,C,H,W]` floats in `[-1,1]`, builds proprioception as position + quaternion-to-axis-angle + gripper position, then normalizes the state. The base `FastWAMAdapter.predict_action()` applies FastWAM's task prompt template and calls `infer_action()`.

In `postprocess_actions()`, the FastWAM adapter denormalizes the model's action tensor using the action statistics and selected normalization mode. It maps the gripper output to LIBERO's command sign convention and can binarize it. This code deliberately raises for unsupported `action_state_transforms` or incompatible shape metadata instead of producing plausible but incorrect actions.

The evaluator does not choose a checkpoint, instantiate a model, infer which stats file belongs to it, or load data statistics. The caller must pass matching checkpoint/config/statistics provenance. Record those values in `LiberoEvalConfig.run_metadata`.

## Tracing and interventions

Generic tracing registers PyTorch forward hooks on named modules. It sees only values that cross those module calls. FastWAM's MoT code can call projections directly and uses functional scaled-dot-product attention, so generic hooks do not necessarily see attention probabilities, post-RoPE keys/values, or each denoising update. Use a projection hook only when the projection output itself answers the question. Add model-specific observers for a deeper cache or denoising boundary.

`src/interventions/` can replace outputs at named module boundaries. It cannot change a value that never passes through the selected module. Compiled action inference is disabled for traced requests because Python hooks do not reliably compose with compiled graphs.

## Adding another model or evaluator

For a new model, implement its loader, adapter, preprocessing, normalization, and action decoding under `src/models/<name>/`. For a new benchmark, implement suite/environment mechanics and result handling under `src/evaluations/<benchmark>/`. Both use the shared types from `src/types.py`; the evaluator should call only the public adapter methods and should never branch on model names. Add a package mapping in `pyproject.toml`, lazy-load optional runtime dependencies, and document the external assets and transforms. See [the live LIBERO guide](libero_evaluation.md) for the concrete boundary and [the experiment protocol](experiment_protocol.md) for what to record.