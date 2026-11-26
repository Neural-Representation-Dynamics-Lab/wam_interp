# E1 first, trajectory correspondence afterward

## Scope and readiness

The local setup supports a sequential four-condition E1 harness, native
FastWAM IDM future extraction/replacement, artifact recording, and trajectory
analysis. The synthetic CPU rehearsal validates plumbing, not WAM behavior.
No checkpoint, simulator assets, or GPU are needed for the rehearsal.

The concrete LIBERO pilot is now implemented in `scripts/e1_libero.py`:
environment/controller recreation, deterministic warm-up and replay verification,
native-vs-observer-vs-identity preflight, candidate generation, review manifests,
four-condition E1 execution, frame recording, and offline coordinate annotation.
Only real checkpoint/simulator validation and scientific review await GPU access.
Other backends are not instrumented by this change. Optional IDM is the first
integration candidate, not a claim that it is the best model for the study.

## Transport and A100 execution

Transport this checkout including the uncommitted/new files. A plain git clone
does not yet contain this work. `python scripts/package_e1.py` builds
`results/e1_handoff/source_overlay.zip`, excluding checkpoints, environments,
simulator checkouts and generated results. On the target, clone the repository,
check out the `base_revision` in the archive's `HANDOFF_SOURCE.json`, then extract
the archive over that clean checkout. Keep the `.git` directory for provenance.
The handoff manifest includes a SHA-256 hash for every source file. Run from the
repository root on Linux:

```bash
python3 scripts/verify_e1_handoff.py
bash scripts/setup_e1_linux.sh
source .venv-e1/bin/activate
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
export LIBERO_CONFIG_PATH="$PWD/results/libero_config"
export DIFFSYNTH_DOWNLOAD_SOURCE=huggingface
python scripts/e1_libero.py check
python scripts/e1_libero.py prepare --output results/e1_live/prepare
```

The setup script pins the released Optional IDM checkpoint and the existing
team LIBERO revision. It requires Python 3.10 and a CUDA-12.8-compatible driver.
Set `E1_PYTHON=/path/to/python3.10` if it is not on PATH. Native model
loading obtains missing Wan/T5 assets through the existing backend downloader;
allow network access and sufficient disk/RAM for the first load. Asset paths and
GPU choice can be adjusted in `configs/experiments/e1_libero.json`.
The Linux installer has not been executed on this Windows laptop.
Its Bash syntax has been checked locally. The exact target environment is
separate from the Windows test environment; local tests are not an installation
test of that Linux environment. Loaded Wan/T5/tokenizer asset contents are hashed
into provenance so a silent base-asset change invalidates the prepared context.

`prepare` first verifies simulator replay, then compares native inference,
capture-only inference and identity replacement. Any action difference above
the configured absolute tolerance stops preparation. It saves decoded baseline
and candidate PNG sequences, candidate hashes, inference latency and peak GPU
allocation. This latency includes extraction overhead, not pure model latency.
Candidate actions are discarded and never executed during donor selection.

Inspect `baseline_frames/` and `candidate_*_frames/`. Edit `review.json` to label
the baseline and candidates, document nuisance changes, accept valid candidates,
choose same/opposite donors, and set the expected behavioral sign. Verify the
video frame/control-step mapping before marking it verified. The initial mapping
is 0,4,...,32 from the native config, and remains an assumption until verified.
Do not accept donors merely to complete the pipeline. If no valid contrast
exists, the scientifically correct outcome is an unsuccessful donor search.

```bash
python scripts/e1_libero.py run --prepared results/e1_live/prepare --output results/e1_live/run
```

This performs E1 and exports imagined/executed frame pairs. After E1, open the
four `*_frame_pairs.html` files, click the gripper/object landmark in each image
or explicitly mark it missing, and download each annotation JSON. Analyze them:

```bash
python scripts/analyze_e1.py --annotations results/e1_live/run/baseline_annotations.json results/e1_live/run/identity_annotations.json results/e1_live/run/same_property_annotations.json results/e1_live/run/opposite_property_annotations.json --e1-summaries results/e1_live/run/e1_summary.json --output results/e1_live/analysis.json
```

Move downloaded annotation JSONs into the run directory first. Pages display
only the fixed external camera, the left half of the native two-camera montage.
Execution NPZs also retain native camera arrays and simulator gripper poses for
later independent calibration. Object coordinates are manually annotated from
video; no unvalidated object-pose estimator is presented as ground truth.
Use **Save draft** and **Resume JSON** for partial annotation sessions. A draft
from another state, condition or context is rejected. The browser tool has been
tested for image loading, landmark clicks, draft round-tripping, missing-point
handling, completed export and downstream metrics.

At execution time, the baseline future must reproduce the future that was
reviewed during preparation. Identity replacement must reproduce both action
predictions and execution. A failure stops before the donor branches; investigate
it rather than relaxing tolerances to make a result pass.
Aggregation rejects mismatched property/behavior units, model, checkpoint,
precision, protocol or expected effect direction. Repeated cases are averaged
within state before forming descriptive across-state means.

For several initial states, review `configs/experiments/e1_cases.json` (a small
exploratory pilot, not a powered study) and run sequentially:

```bash
python scripts/e1_campaign.py prepare --output results/e1_campaign
# Review each case's prepare/review.json before execution.
python scripts/e1_campaign.py run --output results/e1_campaign
```

Completed cases are skipped on resume. Failed/partial cases are retained for
inspection and never overwritten. Separate processes release GPU memory between
cases. E1 requires semantic review between stages; trajectory analysis requires
annotations afterward. Neither human assessment can be truthfully automated by
assuming the requested semantic difference exists.

## Local commands

From the repository root, using an environment with this checkout installed:

```powershell
# Existing local environment (already contains NumPy, pytest and torch):
.\.venv-validation\Scripts\python.exe -m pytest tests/test_e1.py -q
.\.venv-validation\Scripts\python.exe scripts/run_e1.py rehearse --output results/e1_cpu_pilot
.\.venv-validation\Scripts\python.exe scripts/run_e1.py trajectory --tracks results/e1_cpu_pilot/trajectory_fixture.npz --metadata results/e1_cpu_pilot/trajectory_fixture.json --output results/e1_cpu_pilot/trajectory_summary.json
```

Use a fresh output path for each run; the commands refuse overwrites. Results
are under the existing ignored `results/` directory. Never treat synthetic
effects as experimental results. On another machine, the lightweight rehearsal
needs `python -m pip install -e ".[research]"`; hook tests additionally need torch.
The actual model environment needs the existing FastWAM/LIBERO dependencies.

To regenerate an inventory and machine-readable local test report:

```powershell
.\.venv-validation\Scripts\python.exe scripts/check_e1_readiness.py --tests --output results/e1_readiness/new_report.json
```

The report deliberately lists absent checkpoints and simulator packages as
pending even when local tests pass. Optional browser verification needs
Playwright and a browser; it is not required for inference:

```powershell
.\.venv-validation\Scripts\python.exe scripts/check_e1_annotation_ui.py --channel chrome --output results/e1_readiness/new_browser_check
```

## E1 protocol

1. Preregister the property, semantic labels, signed behavioral measurement,
   expected direction, task/state selection, and donor exclusion rules.
2. Record one resolved recipient context, including checkpoint/statistics/config,
   observation and simulator-state hashes, prompt, action seed and precision.
3. Extract candidate futures under identical observation/history/instruction.
   Record candidate-generation seeds separately. Label before inspecting policy
   effects. Do not change the recipient action seed to obtain a donor.
4. Review a same-property and opposite-property donor. Record nuisance changes:
   replacing an entire future does not guarantee that only one property changed.
5. Restore the full recipient state before baseline, identity, same-property and
   opposite-property runs. Execute a fixed action prefix without replanning.
6. Compare signed physical behavior across the four runs. A success drop alone
   is not evidence of semantic use. Stop and diagnose a failed identity control.
7. Repeat across initial states/tasks. Aggregate at the initial-state level;
   repeated seeds/chunks are not independent samples. The current report is
   descriptive per-state, with no significance or confidence-interval claims.
8. Run the imagined-versus-executed trajectory follow-up after E1. Coordinates
   and video references can be recorded during E1 and annotated afterward.

## Native future boundary

```python
from wam_interp.experiments.futures import predict_with_future

# adapter is the existing FastWAMAdapter around a loaded Optional IDM model.
# request uses condition="idm", compile_action_infer=False, native bf16 settings.
result, capture = predict_with_future(adapter, request)
original = capture["original"]  # CPU torch tensor [B,C,latent_time,H,W]
used = capture["used"]

# Donor is already independently reviewed and matched to this context.
patched_result, patched_capture = predict_with_future(adapter, request, donor=donor)
```

The boundary is after video denoising and before the action-conditioning K/V
prefill. Replacement only modifies latent time slices 1 onward; slice zero is
the observed anchor and is checked exactly. These are VAE latent indices, not
raw-frame numbers. The native VAE must decode `used` to inspect what actually
conditioned the action. Capture alone retains native output; hooks are removed
on errors, nesting is rejected, and calls must be sequential. This boundary
does not expose every functional attention operation.

Save bf16 tensors as `tensor.float().numpy()` when using NPZ and record both
inference dtype and storage dtype. Do not change inference to fp16 to save disk
space. Retain native resolution/horizon during the first checkpoint validation.

## Session and artifact contract

`wam_interp.experiments.e1.Session` defines three methods:

- `snapshot()`: save simulator/controller state, adapter episode history, model
  caches, and all relevant RNG/generator states. A seed or `qpos` alone is not a
  complete snapshot. Return a reusable snapshot that restore does not mutate.
- `restore(snapshot)`: restore the same complete state before every branch.
- `run(donor)`: infer through the explicit boundary, decode actions using the
  benchmark adapter, execute one chunk, and return `Branch`. `behavior` must be
  the declared physical measure (e.g. signed gripper displacement), not a model
  action value. Preserve normalized actions, decoded commands and executed prefix.

The existing adapter's observation/action transforms should be reused. The
existing shared evaluator is unchanged. A Session factory can own a simulator
and call its existing reset/step helpers without changing evaluator behavior.

`wam_interp.experiments.fastwam_session.FastWAMChunkSession` supplies native
inference, future capture/replacement, action decoding and prefix selection.
Pass the already loaded evaluation adapter, prepared request, `execute_steps`,
and three callbacks: `snapshot_fn()`, `restore_fn(state)`, `rollout_fn(commands)`.
The rollout callback returns `behavior`, `executed_commands`, optional `metadata`
and `trajectories`. Implement and validate these simulator callbacks on the
target machine; no simulator state format is guessed by this helper.

```python
summary = run_e1(session, donors={"same_property": same, "opposite_property": opposite},
                 manifest=manifest, output="results/e1/state_001")
```

For custom backends, `scripts/run_e1.py run --factory package.module:make_session --manifest manifest.json
--donors donors.npz --output results/e1/state_001` is the equivalent CLI once the
factory is implemented. The supplied concrete LIBERO pilot above needs no custom
factory. The generic CLI imports Python code from the specified module.

The rehearsal writes a complete example `manifest.json`. For real runs replace
every synthetic context value, set `synthetic` to false, calculate each donor's
`context_id` with `context_id(manifest["context"])`, and supply reviewed labels and
nuisance notes. Store task/trial IDs, history hashes, inference settings, asset
hashes, camera transforms, software versions and working-tree identity in context.
Use file checksums, not filenames alone, for checkpoint/statistics identity.
`context_id` checks supplied provenance, not whether the caller recorded it truthfully.

Each run saves the manifest, donor arrays, per-branch used future/actions/commands,
branch metadata, descriptive effects and a complete/failed status. Session.run
should include video/state artifact paths, termination reason, effective
precision and inference latency in metadata. Store large videos separately.

## Trajectory follow-up

Use one NPZ per condition and landmark, with arrays `times`, `imagined`,
`executed`, and optional `valid`. Coordinates must be [time,2] or [time,3] in
the same units and reference frame, with strictly increasing timestamps.
Use NaNs/visibility flags for absent detections. JSON metadata declares landmark,
coordinate frame, units, executed_until, condition, state_id and synthetic status.

Start with independently annotated gripper/object points in the fixed external
camera. Validate automated tracking against annotations and simulator projections.
An object's mask centroid is not necessarily its simulator body origin. Do not
infer imagined coordinates from the tested policy's own action head. Wrist
camera motion and crop/flip transforms require explicit calibration.

The analysis reports ADE, endpoint error, displacement error, stationary-baseline
ADE, error over time and visibility coverage. An invisible endpoint produces
null FDE, not the last visible point. All-missing tracks remain null, not zero.
`executed_until` excludes predictions after replanning/termination. No time
warping or interpolation is applied. Verify the 32-action/9-frame FastWAM mapping
from actual timestamps rather than aligning arrays by index.

Optional `Branch.trajectories` entries keyed `gripper`/`object` save these arrays
and metadata automatically. Contact/grasp/lift agreement should be annotated
separately; geometry alone cannot establish a successful interaction.

## First A100 session

1. Confirm available VRAM and the matching checkpoint/native config.
2. Load one model in bf16, eager mode; batch size one. Measure peak memory.
3. Compare native inference with capture-only and identity replacement on exactly
   the same restored input/RNG state. Measure numerical repeat variability first.
4. Decode saved futures with the native VAE; verify camera and time mappings.
5. Validate simulator/controller restore by replaying the same commands twice.
6. Only then collect/review donors and run a small E1 pilot. Record null results.

Keep donor tensors on CPU between runs and analyze video in a separate stage.
No weights, simulator checkouts, caches or generated results belong in commits.
