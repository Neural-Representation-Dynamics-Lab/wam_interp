# WAM Interpretability

This repository is research infrastructure for studying how world-action models use their internal representations to choose actions. FastWAM is the first model backend; the shared interfaces are intended to support other models and benchmarks without moving their implementation details into common code.

The main design rule is: **a benchmark evaluator controls the benchmark; a model adapter controls the model-facing transforms.** The evaluator does not load a model, checkpoint, tokenizer, or normalization file. It receives a ready-to-use adapter and calls its public methods.

## Repository map

```text
configs/fastwam/                 FastWAM Hydra model/task configuration
src/types.py                     Shared raw/prepared observation and inference types
src/models/                      Model adapters, loaders, and model-specific preprocessing
src/evaluations/LIBERO/          LIBERO suite, simulator, rollout, and result handling
src/data/                        Offline prepared-observation helpers (for example NPZ)
src/tracing/                     Passive module-output capture
src/interventions/               Scoped changes to named module outputs
docs/                            Architecture, experiment, and evaluation guidance
```

FastWAM's vendored model implementation is under `src/models/fastwam/model_code/`. Model checkpoints and pretrained video/text assets are external. LIBERO's simulator, task definitions, and MuJoCo assets are also external; the evaluator imports the upstream LIBERO package only when `evaluate_libero()` starts.

## Installation

Install this repository, the FastWAM backend, and the LIBERO simulator runtime:

```bash
python -m pip install -e ".[all]"
```

The `all` extra includes FastWAM's Python dependencies and the LIBERO simulator runtime (`robosuite>=1.4.0,<1.5`, `mujoco>=3.3,<3.10`, and the task/parser packages). The bounds keep LIBERO on robosuite's required 1.4 API and avoid the MuJoCo 3.10 API change that breaks robosuite 1.4. LIBERO's legacy requirements file pins Python 3.8 and PyTorch 1.11; do not install it into this environment because that would replace the model stack. The upstream LIBERO source checkout is still needed for task definitions, fixed initial states, and simulator assets. Clone and install that source without re-resolving its old dependencies:

```bash
mkdir -p third_party
git clone --depth 1 --no-checkout --branch master https://github.com/Lifelong-Robot-Learning/LIBERO.git third_party/LIBERO
git -C third_party/LIBERO fetch --depth 1 origin 8f1084e3132a39270c3a13ebe37270a43ece2a01
git -C third_party/LIBERO checkout --detach 8f1084e3132a39270c3a13ebe37270a43ece2a01
python -m pip install -e "./third_party/LIBERO" --no-deps --config-settings editable_mode=compat
```

The source checkout is pinned to LIBERO commit `8f1084e3132a39270c3a13ebe37270a43ece2a01` in the notebook and the manual setup above so collaborators use the same task definitions and initial states. The notebook downloads the released FastWAM checkpoint and statistics from Hugging Face. FastWAM then downloads any missing Wan VAE, text-encoder, and tokenizer assets from ModelScope, its default source, into its configured cache; the base video DiT download is skipped for this checkpoint. Network access is required, but no separate manual model-asset setup is needed.

The evaluator needs no LIBERO demonstration trajectories. Before evaluation, LIBERO needs a `config.yaml` with paths to its benchmark root, BDDL files, initial states, datasets, and assets. The notebook creates this config and checks a real environment reset and camera render before loading FastWAM. Its default run evaluates task 0 of `libero_spatial` for 50 full-length episodes and reports that task's success rate; it is not the full-suite average. See [the LIBERO evaluation guide](docs/libero_evaluation.md) for setup and comparison details.

## Run one prepared observation

The existing command-line runner is an offline, one-observation model call. It does not start a simulator or run benchmark episodes:

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

The NPZ file contains `image` and optional `proprio` arrays already arranged for the selected model. Its actions are returned in the model's normalized space. It is separate from live LIBERO evaluation, which gets observations from the simulator and asks the model adapter to prepare them.

## Run live LIBERO evaluation

`evaluate_libero()` accepts an adapter object. It manages suite/task selection, fixed initial states, reset and warm-up, closed-loop action replanning, environment stepping, success accounting, and result files. The adapter manages image/state preprocessing, model inference, and conversion from normalized model actions to LIBERO commands.

```python
from omegaconf import OmegaConf

from wam_interp.evaluations.LIBERO import LiberoEvalConfig, evaluate_libero
from wam_interp.models.fastwam.loader import compose_fastwam_config, load_fastwam
from wam_interp.models.fastwam.libero_adapter import FastWAMLiberoAdapter

# Model construction and checkpoint loading happen here, outside the evaluator.
fastwam_cfg = compose_fastwam_config("libero_optional_idm_2cam224_1e-4")
raw_model = load_fastwam(fastwam_cfg, "/path/to/checkpoint.pt", device="cuda")
fastwam_data = OmegaConf.to_container(fastwam_cfg.data.train, resolve=True)

adapter = FastWAMLiberoAdapter(
    raw_model,
    data_config=fastwam_data,
    dataset_stats="/path/to/dataset_stats.json",
)
result = evaluate_libero(
    model=adapter,
    config=LiberoEvalConfig(
        task_suite_name="libero_spatial",
        task_ids=(0,),
        num_trials=2,
        replan_steps=10,
        action_horizon=32,
        seed=0,
        output_dir="results/libero",
        inference_options={"condition": "first_frame", "num_inference_steps": 10},
        run_metadata={"checkpoint": "/path/to/checkpoint.pt"},
    ),
)
print(result.success_rate, result.output_path)
```

`FastWAMLiberoAdapter` is the FastWAM-side bridge. It rotates and resizes LIBERO camera images, concatenates the configured views, builds and normalizes proprioception from end-effector pose and gripper state, calls the existing FastWAM inference adapter, denormalizes the action chunk, and maps the gripper convention. These are checkpoint-specific operations, so they live under `src/models/fastwam/`, not in the evaluator.

See [the LIBERO evaluation guide](docs/libero_evaluation.md) for the full call flow, config fields, output format, dependencies, and adaptation notes. The `LiberoEvalConfig` defaults to task 0 in `libero_spatial`; set `task_ids=None` to evaluate every task in the selected suite. Defaults are deliberately explicit and can be changed per run.

## Adding another model

Put its loader, adapter, preprocessing, normalization, and action decoding in `src/models/<model_name>/`. Its evaluation adapter must implement:

- `inference_defaults() -> Mapping[str, Any]` for model-configured request values (caller overrides still win)
- `prepare_observation(EnvironmentObservation) -> PreparedObservation`
- `predict_action(InferenceRequest) -> InferenceResult`
- `postprocess_actions(actions, environment="libero") -> environment commands`

The evaluator passes raw cameras, named raw state values, and the task instruction to `prepare_observation`. `predict_action` should return model-normalized action chunks. `postprocess_actions` converts those actions into the target environment's command convention. Do not add model class names, checkpoint formats, image normalization, robot-state normalization, or action denormalization to `src/evaluations/LIBERO/`.

This adapter work is evaluation-specific and must be included whenever a model is added or connected to another evaluation. For each supported model/environment pairing, implement the required observation preparation and action conversion in that model's adapter. The evaluator should handle benchmark mechanics and pass observations through the adapter; it should not contain model-specific preprocessing or branching.

## Adding another evaluation

Use `src/evaluations/LIBERO/` as the example, but keep the new benchmark in its own folder, for example `src/evaluations/ROBOTWIN/`:

1. Put task selection, environment creation, reset/step calls, termination rules, metric calculation, and result serialization in that evaluation folder.
2. Convert environment observations into the shared `EnvironmentObservation` structure without resizing, normalizing, or reordering for a particular model.
3. Call the model adapter's `prepare_observation`, `predict_action`, and `postprocess_actions` methods. Add the matching environment-specific preparation and action conversion to each model adapter intended to work with this evaluation; the model backend owns those transforms.
4. Keep benchmark-specific command validation and episode semantics in the evaluator. Never load model weights from `evaluate_<benchmark>()`.
5. Make optional benchmark imports lazy, so importing shared interfaces does not require every simulator to be installed.
6. Add a guide under `docs/` and a clear dependency/setup note. Record suite/task IDs, initial-state choice, seeds, action horizon, replan interval, inference condition, checkpoint/statistics identity, and the final success metric.
7. Add the new package to `pyproject.toml`'s explicit package list and `package-dir` map.

## Research and tracing

Keep evidence about representation content, causal influence on actions, and task benefit separate. Generic PyTorch hooks only observe named module boundaries; they do not automatically expose functional attention values or post-RoPE caches. See [the architecture guide](docs/architecture.md) and [experiment protocol](docs/experiment_protocol.md) for limits and recording practices.

## References

- [From World Models to World Action Models: A Concise Tutorial for Robotics](https://arxiv.org/abs/2607.00836)
- [Fast-WAM: Do World Action Models Need Test-time Future Imagination?](https://arxiv.org/abs/2603.16666)
- [Faster-WAM: Efficient Inference-Time Future Conditioning for Robust World Action Models](https://arxiv.org/abs/2608.04404)
