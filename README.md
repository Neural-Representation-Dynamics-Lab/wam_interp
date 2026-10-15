# WAM Interpretability

This repository is for studying how world-action models use predicted futures when they choose actions. We want to understand three things separately: what a predicted future represents, whether the action computation uses that information, and whether using it helps on a task. A generated video may be useful to the model without being a faithful explanation of its decision, so the infrastructure is designed to test those claims rather than assume them.

FastWAM is the first model backend. Its Optional IDM variant can run either the `first_frame` action path or the future-conditioned `idm` path. Comparing those modes is a useful starting point, but the paths differ in more than future content; a mode comparison alone does not prove that the imagined content caused an action change or improved behavior.

## How the repository is organized

The main boundary is the model adapter. Shared experiment and evaluation code should describe an observation and an inference request, then call an adapter. It should not need to know a model's internal class names, checkpoint format, prompt template, or attention implementation. Each model backend owns those details and translates the shared request into its model's API.

That separation lets us add evaluation environments and interpretability tools without making them FastWAM-specific. It also makes it possible to compare models through the same outer workflow while keeping model-specific limits explicit.

```text
configs/fastwam/       FastWAM Hydra configs and task/model choices
docs/                  Architecture, research questions, and experiment protocol
src/types.py           Shared observation, request, and result types
src/models/            Adapter contract, registry, and model backends
src/data/              Loading and validating prepared observations
src/experiments/       Small helpers for paired model inference
src/tracing/           Passive module-output capture
src/interventions/     Scoped edits at named module outputs
```

FastWAM's model implementation is included under `src/models/fastwam/model_code/`, with its license alongside it. The Hydra configs are in the repository under `configs/fastwam/`. Model checkpoints and pretrained assets are still external inputs; they are not included here.

## Set up

From the repository root:

```bash
python -m pip install -e ".[all]"
```

This installs the repository and the dependencies currently declared for FastWAM. It does not download a checkpoint or pretrained model assets, install LIBERO or RoboTwin, or set up their simulators. The current development workflow uses an editable install from a checkout, so the repository's Hydra config files remain available to the loader.

In notebooks, restart the runtime after installing or changing packages. The running Python process can keep older modules loaded even after their files have been replaced.

For a complete Colab example that clones the repository, downloads the released FastWAM checkpoint, and loads it through the adapter, see [the FastWAM Colab guide](docs/fastwam_colab.md). The model-specific README at [src/models/fastwam/README.md](src/models/fastwam/README.md) explains the model variants and config naming.

The package import name is `wam_interp`:

```python
import wam_interp.models
```

## Run one prepared observation

The current runner makes one offline model call. It does not start an environment or run a benchmark episode. A checkpoint and its required pretrained assets must be available in your environment.

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

An observation NPZ contains two arrays:

- `image`: an RGB image in CHW or HWC layout. `uint8` pixels are converted to `[-1, 1]`; floating-point pixels must already be in that range. Height and width must be divisible by 16.
- `proprio`: a one-dimensional state vector already normalized for the selected checkpoint.

The camera order, image montage, resizing, and state normalization must match the checkpoint. The runner does not apply dataset statistics, convert normalized actions to robot commands, or step an environment. Its output actions are model-normalized.

The Hydra `--task` choice selects a task configuration, including its model configuration. It does not just name a benchmark task. For Optional IDM, `--condition` selects the inference path: use `first_frame` for action-only inference and `idm` for future-conditioned inference. Other FastWAM variants may only support their configured path; the adapter raises an error for unsupported combinations.

The same adapter API is available to Python evaluation code:

```python
from wam_interp.data.npz import load_observation_npz
from wam_interp.models import load_model_adapter
from wam_interp.types import InferenceRequest

adapter = load_model_adapter(
    "fastwam",
    task="libero_optional_idm_2cam224_1e-4",
    checkpoint="/path/to/checkpoint.pt",
)
observation = load_observation_npz("/path/to/observation.npz", "pick up the object")
result = adapter.predict_action(
    InferenceRequest(observation=observation, condition="first_frame")
)
```

An evaluation loop should call this interface, then handle its own environment setup, action post-processing, execution, and success measurement. That code does not exist yet.

## Add a model backend

Keep a new model's implementation and assumptions inside `src/models/<model_name>/`.

1. Add a loader that constructs the model from its config/checkpoint and register it with `@register_model_loader("<model_name>")`.
2. Add an adapter that registers with `@register_model_adapter("<model_name>")` and implements `predict_action(request, recorder=None)`.
3. Translate `InferenceRequest` into the model's inputs inside that adapter. Return an `InferenceResult` with the action chunk, useful provenance in `metadata`, and trace events when requested. Keep actions in the shared normalized representation; robot-specific transforms belong in the evaluation layer.
4. Import the backend from `src/models/__init__.py` so the registration decorators run when `wam_interp.models` is imported. Keep heavy model dependencies inside the loader or inference call where possible.
5. Add a model-specific config under `configs/<model_name>/` only if the backend needs configuration files. Keep shared evaluation code free of model-specific keys and classes.
6. Document the observation format, supported inference conditions, checkpoint/assets needed, and any unsupported features. Add a small load-and-predict check before wiring the adapter into an environment evaluator.

The common types and registry live in `src/types.py`, `src/models/base.py`, and `src/models/registry.py`. FastWAM is the example backend to follow, but its prompt formatting, Hydra choices, checkpoint loading, and IDM rules should not be copied into shared code.

## What to build next

The next infrastructure work is to connect the adapter to real evaluation loops:

1. **LIBERO evaluation.** Add task and trial management, observation preprocessing that matches the checkpoint, action denormalization and gripper mapping, replanning, success measurement, and saved run metadata. The existing LIBERO Hydra configs select model settings; they are not an evaluator.
2. **RoboTwin evaluation.** Add the corresponding camera/state preprocessing and action/environment interface for RoboTwin. Reuse the adapter contract, but keep environment-specific transformations in the evaluator.
3. **Mechanistic interpretability tools.** The generic hooks only see named PyTorch module boundaries. FastWAM also uses direct projections and functional attention, so exact attention/cache values, per-step denoising states, and meaningful MoT boundaries need explicit observation points in the model runtime. Add those where they answer a concrete research question.
4. **More model backends.** Add another WAM behind the same request/result contract before treating a FastWAM result as a general claim about WAMs.

When reporting results, keep representation content, causal influence on actions, and task benefit separate. The [research questions](docs/research_questions.md) explain the distinctions; the [experiment protocol](docs/experiment_protocol.md) describes what to record; and the [architecture notes](docs/architecture.md) describe where new code belongs.

## References

- [From World Models to World Action Models: A Concise Tutorial for Robotics](https://arxiv.org/abs/2607.00836)
- [Fast-WAM: Do World Action Models Need Test-time Future Imagination?](https://arxiv.org/abs/2603.16666)
- [Faster-WAM: Efficient Inference-Time Future Conditioning for Robust World Action Models](https://arxiv.org/abs/2608.04404)
- [NVIDIA: Pretrained to Imagine, Fine-Tuned to Act](https://developer.nvidia.com/blog/pretrained-to-imagine-fine-tuned-to-act-the-rise-of-world-action-models/)
