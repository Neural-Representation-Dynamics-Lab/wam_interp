# Live LIBERO evaluation

This is the closed-loop simulator evaluator. It is separate from training and from the one-observation NPZ inference command.

## Install the evaluation environment

Use Python 3.10 or newer in a clean environment. From the repository root, install the project's dependencies:

```bash
python -m pip install -e ".[all]"
```

This installs the FastWAM runtime plus the LIBERO simulation dependencies used here: `robosuite>=1.4.0,<1.5`, `mujoco>=3.3,<3.10`, and the BDDL/task support packages. LIBERO's upstream `requirements.txt` is for an older Python 3.8/PyTorch 1.11 stack; do not install it into this environment. The robosuite bound retains LIBERO's 1.4 environment API, and the MuJoCo upper bound avoids the 3.10 `mj_fullM` signature change. The evaluator does not need LIBERO's demonstration dataset.

Clone LIBERO's official source checkout and install its package without allowing its legacy setup to resolve dependencies again:

```bash
mkdir -p third_party
git clone --depth 1 --no-checkout --branch master https://github.com/Lifelong-Robot-Learning/LIBERO.git third_party/LIBERO
git -C third_party/LIBERO fetch --depth 1 origin 8f1084e3132a39270c3a13ebe37270a43ece2a01
git -C third_party/LIBERO checkout --detach 8f1084e3132a39270c3a13ebe37270a43ece2a01
python -m pip install -e "./third_party/LIBERO" --no-deps
```

The notebook and this guide use the official LIBERO commit `8f1084e3132a39270c3a13ebe37270a43ece2a01` for repeatable task definitions and initial states.

The notebook runs task 0 of libero_spatial for 50 full-length episodes, matching FastWAM's single-task evaluation budget. Its reported success rate is task-specific, not the full-suite average. Compare against the same suite, task, checkpoint, and inference settings; exact reproduction also depends on matching seed and initial-state handling.

LIBERO needs a path config before `get_libero_path()` can be used. The notebook writes this automatically. For a manual setup, make a directory (for example, `./libero_config`) and put a `config.yaml` there with absolute paths like the following; set `LIBERO_CONFIG_PATH` to that directory before importing `libero.libero`:

```yaml
benchmark_root: /absolute/path/to/third_party/LIBERO/libero/libero
bddl_files: /absolute/path/to/third_party/LIBERO/libero/libero/bddl_files
init_states: /absolute/path/to/third_party/LIBERO/libero/libero/init_files
datasets: /absolute/path/to/empty-or-existing/libero_datasets
assets: /absolute/path/to/third_party/LIBERO/libero/libero/assets
```

Then export the config and renderer paths in that shell before starting Python:

```bash
export LIBERO_CONFIG_PATH="$(pwd)/libero_config"
export MUJOCO_GL=egl
```

The `datasets` directory may be empty for evaluation; it is included because upstream LIBERO expects the key in its config. The notebook runs a simulator construction, fixed-state reset, and camera-observation check before downloading/loading the model checkpoint, so renderer or path errors are caught early.

The model-loading cell downloads the released FastWAM checkpoint and normalization statistics. It directs the vendored loader to Hugging Face and caches missing Wan VAE, text-encoder, and tokenizer files under `checkpoints/wan_assets`. The base Wan video DiT is not separately downloaded for this released FastWAM checkpoint (`skip_dit_load_from_pretrain=true`). This setup requires outbound network access; it does not require a separate Wan install or a LIBERO demonstration-data download.

## What lives where

LIBERO's Python package, BDDL task definitions, fixed initial simulator states, and simulator assets come from its upstream source checkout. The `.[all]` extra installs the compatible MuJoCo/robosuite runtime, while the notebook clones and configures the upstream checkout. `evaluate_libero()` itself does not download any of these or load a policy checkpoint.

At run time, the evaluator asks LIBERO for a suite and a task. The task object provides its natural-language instruction and the names of its BDDL file and problem folder. The environment frontend resolves that task's BDDL file through `get_libero_path("bddl_files")` and creates `OffScreenRenderEnv`. The suite object returns initial states for a task through `get_task_init_states(task_id)`. These simulator states are not the demo-training dataset and are not loaded from an evaluation NPZ.

## The boundary between evaluator and model

The public entry point is `evaluate_libero(model, config=...)` in `src/evaluations/LIBERO/runner.py`. `model` must already be an evaluation-capable adapter. The function does not call a loader or inspect a checkpoint.

The shared contract in `src/models/base.py` is:

1. `prepare_observation(EnvironmentObservation)` turns raw benchmark cameras/state into the selected model's `PreparedObservation`.
2. `predict_action(InferenceRequest)` performs one inference call and returns an `InferenceResult` containing a normalized action chunk.
3. `postprocess_actions(actions, environment="libero")` converts that model's normalized representation into LIBERO's seven-value controller commands.

The first and third steps depend on the model and checkpoint, so they belong in `src/models/<backend>/`. LIBERO-specific suite, reset, stepping, replanning, termination, and metrics belong in `src/evaluations/LIBERO/`. The evaluator only checks the environment command shape `[time, 7]`; it does not know a FastWAM normalizer or gripper transform.

## One episode, in order

1. **Select a task.** `runner.py` creates a suite using `benchmark.get_benchmark_dict()[suite_name]()`, calls `get_task(task_id)`, and obtains initial states from the suite object. `task_ids=None` means all suite tasks; otherwise only the supplied IDs run.
2. **Create the simulator.** `environment.py` resolves `<bddl_files>/<problem_folder>/<bddl_file>` and constructs `OffScreenRenderEnv` at the configured camera resolution.
3. **Reset to a benchmark state.** Each trial calls `env.reset()` and `env.set_init_state(initial_state)`. If there are fewer stored initial states than trials, trial `i` uses `i % number_of_states`, so repeats are explicit and recorded by `initial_state_index`.
4. **Warm up, if configured.** The evaluator advances `num_steps_wait` steps with LIBERO's no-op command `[0, 0, 0, 0, 0, 0, -1]`. It does not ask the model to act during these steps. This preserves the rollout protocol used by FastWAM and gives the simulator a short settling interval before the first policy observation.
5. **Expose the raw observation.** The frontend maps LIBERO's `agentview_image`, `robot0_eye_in_hand_image`, end-effector position/quaternion, and gripper position into an `EnvironmentObservation`. Camera arrays stay in raw simulator form. The frontend does not rotate, crop, resize, normalize, or concatenate them.
6. **Prepare for the selected model.** The runner calls `model.prepare_observation(raw_observation)`. For FastWAM, `FastWAMLiberoAdapter` rotates each camera image by 180 degrees, center-crops/resizes each view to configured dimensions, concatenates the views in configured order, converts RGB pixels to `[-1,1]`, converts the `xyzw` quaternion to axis-angle, and normalizes the resulting state with the supplied dataset statistics.
7. **Make one prediction.** The runner builds an `InferenceRequest` from the prepared observation, `action_horizon`, and `inference_options`. It calls `model.predict_action(request)`. For FastWAM, its existing `FastWAMAdapter` wraps the instruction in FastWAM's prompt template and calls the already-loaded model's `infer_action()`.
8. **Decode the chunk.** The runner passes `InferenceResult.actions` to `model.postprocess_actions(..., environment="libero")`. The FastWAM adapter denormalizes with action statistics and maps the gripper to LIBERO's `-1=open`, `+1=close` convention (optionally binarizing it).
9. **Execute only a prefix.** The evaluator sends up to `replan_steps` actions from the predicted chunk to `env.step()`. It then discards the unexecuted remainder, observes again, and asks for another chunk. The final chunk is shortened when it would exceed the episode step limit.
10. **Stop and score.** Success comes from the environment's `check_success()` when available; the upstream reward/info success signal is the fallback. An environment termination without a success signal is recorded as termination, not automatically counted as success. The default maximum policy steps follow FastWAM's suite limits: 400 for spatial/object/goal, 700 for LIBERO-10/90. `max_steps` overrides them.
11. **Save results.** The function returns `LiberoEvaluationResult` with task and episode outcomes, success rates, seeds, initial-state indices, warm-up/policy step counts, and inference metadata. If `output_dir` is set, it writes a timestamped JSON result. Requested module traces are included with that inference call; use `trace_modules` sparingly because activation tensors can make result files large.

## Calling it with FastWAM

The adapter is not a checkpoint loader. An orchestration script loads the model, resolves its data config and normalization stats, constructs the FastWAM evaluation adapter, and then passes that adapter to the generic function:

```python
from omegaconf import OmegaConf
from wam_interp.evaluations.LIBERO import LiberoEvalConfig, evaluate_libero
from wam_interp.models.fastwam.loader import compose_fastwam_config, load_fastwam
from wam_interp.models.fastwam.libero_adapter import FastWAMLiberoAdapter

cfg = compose_fastwam_config("libero_optional_idm_2cam224_1e-4")
raw_model = load_fastwam(cfg, checkpoint="/path/to/checkpoint.pt", device="cuda")
data_config = OmegaConf.to_container(cfg.data.train, resolve=True)
adapter = FastWAMLiberoAdapter(
    raw_model,
    data_config=data_config,
    dataset_stats="/path/to/dataset_stats.json",
)
result = evaluate_libero(
    model=adapter,
    config=LiberoEvalConfig(
        task_suite_name="libero_spatial",
        task_ids=(0,),
        num_trials=5,
        action_horizon=32,
        replan_steps=10,
        inference_options={"condition": "first_frame", "num_inference_steps": 10},
        run_metadata={"checkpoint": "/path/to/checkpoint.pt"},
        output_dir="results/libero",
    ),
)
```

The shared evaluator defaults to `condition="first_frame"`, which runs FastWAM action-only inference. There is one reproducibility detail: the original FastWAM `eval_libero_single.py` defaults Optional IDM to `action_infer_mode="idm"` when that argument exists (unless its config overrides it). To reproduce that original evaluator setting, pass `inference_options={"condition": "idm"}`. The adapter derives FastWAM's frame count from the task config and validates the mode; the evaluator only forwards options.

The FastWAM adapter reads only the configuration and stats required for model-side transforms. The shown LIBERO config has a single merged state/action key, one or two cameras, a `video_size`, `concat_multi_camera`, `norm_default_mode`, and `use_stepwise_action_norm`. It currently rejects enabled `action_state_transforms` rather than silently skipping them. Add such transforms inside the model backend if the checkpoint config needs them.

## Configuration fields

- `task_suite_name`: one of `libero_spatial`, `libero_object`, `libero_goal`, `libero_10`, or `libero_90`.
- `task_ids`: task indices, or `None` for every task in the suite.
- `num_trials`: episodes per task. Initial states cycle when the requested count is larger than the suite's stored list.
- `max_steps`: policy action steps per episode; `None` uses the upstream FastWAM suite limits described above. Warm-up steps are recorded separately.
- `num_steps_wait`: no-op simulator steps before asking for the first model prediction.
- `replan_steps`: number of predicted actions executed before observing and inferring again.
- `action_horizon`: number of actions requested from the model on each inference call.
- `seed`: base seed. The evaluator derives a stable seed per task/trial; absent an explicit `inference_options["seed"]`, each inference call uses that episode seed plus its call index.
- `camera_resolution`: width and height passed to the LIBERO offscreen renderer.
- `inference_options`: fields of shared `InferenceRequest` except `observation` and `action_horizon`. Examples include `condition`, `num_video_frames`, `num_inference_steps`, `sigma_shift`, `text_cfg_scale`, and `compile_action_infer`.
- `run_metadata`: caller-provided provenance, such as checkpoint path/hash, dataset-statistics path/hash, source revision, and external asset versions. The evaluator does not guess these values.
- `output_dir`: where the evaluator writes its JSON result.

## Adding another benchmark evaluator

Create `src/evaluations/<BENCHMARK>/` for benchmark mechanics and result types. Define its own config and `evaluate_<benchmark>(model, config, ...)` function. Keep observation acquisition and task semantics there, but convert each raw observation into `EnvironmentObservation` and delegate model-specific preprocessing/inference/action decoding to the adapter. Use lazy imports for simulator packages so the rest of `wam_interp` imports without those packages. Add the package mapping in `pyproject.toml` and link the new guide from the README.

A backend must opt into that environment by implementing `postprocess_actions(..., environment="<benchmark>")`; this makes action semantics explicit without putting model names or checkpoint assumptions in evaluation code.
