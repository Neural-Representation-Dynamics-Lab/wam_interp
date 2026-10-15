# Architecture and model integration

## The boundary between shared and model-specific code

Evaluation code should be able to ask a model for an action without knowing how that model is implemented. The shared request and result types provide that boundary:

- `PreparedObservation` carries the image, normalized proprioception, task prompt, and optional input metadata.
- `InferenceRequest` adds the requested condition, action horizon, inference settings, and optional module paths to trace.
- `InferenceResult` returns the normalized action chunk, run metadata, and any captured trace events.
- `ModelAdapter.predict_action()` turns one request into one model inference call.

The adapter owns model-specific prompt formatting, argument names, inference modes, and unsupported combinations. The evaluator owns the environment, observation collection and preprocessing, action post-processing, stepping, and task-success measurement. Keeping those responsibilities separate lets a new benchmark use an existing adapter and lets a new model use an existing evaluator.

## Where code lives

- `src/types.py`: shared observation, request, and result data structures.
- `src/models/base.py`: the model adapter protocol.
- `src/models/registry.py`: loader and adapter registration, plus `load_model_adapter()`.
- `src/models/<name>/`: a backend's loader, adapter, and any model-specific runtime code.
- `configs/<name>/`: Hydra configuration groups when a backend needs them.
- `src/data/`: helpers for turning stored data into `PreparedObservation` objects.
- `src/experiments/`: small model-neutral inference helpers, such as paired conditions.
- `src/tracing/`: passive hooks and trace recording.
- `src/interventions/`: scoped changes at named module outputs.

Importing `wam_interp.models` imports the built-in backend adapters and loaders so their registration decorators run. Keep heavyweight imports such as PyTorch, Hydra, and model code out of the shared registry; load them in the backend path that needs them.

## FastWAM loading and inference

FastWAM's model code is in `src/models/fastwam/model_code/`. Its Hydra configuration groups are in `configs/fastwam/`. Checkpoints and pretrained assets are not stored in the repository.

For `load_model_adapter("fastwam", ...)`, the registry calls the FastWAM loader and wraps the resulting model with `FastWAMAdapter`:

1. The loader composes `configs/fastwam/sim_libero.yaml` with the selected `task` group. These task files choose a model config as well as task and training values; the name is not merely a LIBERO benchmark task identifier.
2. Hydra instantiates the selected model target in `src/models/fastwam/runtime.py`. The factory turns the composed config into the arguments expected by the corresponding FastWAM class and loads its pretrained base assets.
3. The loader applies the supplied checkpoint with the model's `load_checkpoint()` method, then moves the model to the requested device and evaluation mode.
4. The adapter translates an `InferenceRequest` into `infer_action()` arguments and returns the action tensor with metadata.

The task config chooses which FastWAM variant is constructed. For Optional IDM, the request's `condition` chooses between `first_frame` and `idm` inference on that already-selected model. A fixed IDM or Joint model does not become Optional IDM by changing `condition`; it supports its configured inference path. The adapter checks for unsupported combinations and the IDM frame-count constraint.

The config directory is resolved from the checkout containing the loader, so callers usually do not need to pass `repository_root`. An explicit root is used when it contains `configs/fastwam`; otherwise the loader falls back to the current checkout's config directory.

## Current input and output boundary

The NPZ helper accepts an `image` array and a `proprio` array. It converts `uint8` RGB pixels to `[-1, 1]`, accepts floating-point pixels only when already in that range, converts HWC images to CHW, and requires image dimensions divisible by 16. Proprioception must be a one-dimensional vector already normalized for the selected checkpoint. The task prompt is passed through the adapter's model-specific template.

FastWAM returns normalized actions. The single-observation runner writes those actions and metadata to a JSON file. It does not read dataset statistics, denormalize actions, convert gripper values, control replanning, start LIBERO or RoboTwin, or calculate success. An environment evaluator must own those steps and preserve both the raw model output and the action actually sent to the environment.

## Tracing and interventions

Generic tracing registers PyTorch forward hooks on named modules. It records module inputs and outputs and can summarize their shapes and dtypes. It does not guarantee visibility into every operation. FastWAM's MoT code can call projections directly and uses functional scaled-dot-product attention, so a module hook may not see attention probabilities, post-RoPE keys and values, or each denoising update.

Use the generic hook when a module boundary answers the question. For exact attention/cache values, per-layer action-to-video interactions, or denoising-step states, add explicit observer points in the FastWAM runtime. Keep these observers model-specific and pass their results through the shared recorder where possible. A hook on a Q/K/V projection is a projection output; do not label it as a cached post-RoPE value unless that is where the hook actually sits.

`src/interventions/` can replace outputs at named module boundaries. This is useful for scoped projection-level edits, but it cannot intervene on a value that never passes through the hooked module. Compiled action inference is disabled for traced requests because Python hooks do not reliably compose with compiled graphs.

## Adding a backend

1. Create `src/models/<name>/loader.py` to build the model and load its checkpoint. Register it with `register_model_loader("<name>")`.
2. Create `src/models/<name>/adapter.py`. Register an adapter with `register_model_adapter("<name>")` and implement `predict_action(request, recorder=None)`.
3. Translate the shared request into model inputs in the adapter. Return normalized actions, model/checkpoint details, and trace data through `InferenceResult`. Raise a clear error for a condition the model does not support.
4. Import the adapter and loader from `src/models/__init__.py`, so the registrations are available after `import wam_interp.models`.
5. Add `configs/<name>/` only if the backend needs configuration files. Keep environment-specific preprocessing and action transforms in the evaluator, not in the shared model registry.
6. Check that the backend can load a known checkpoint and produce an action for one prepared observation before connecting it to an environment loop.

The shared loader and adapter registries are deliberately small. A backend can use Hydra or a different configuration system; Hydra is a FastWAM choice, not a requirement for every model.
