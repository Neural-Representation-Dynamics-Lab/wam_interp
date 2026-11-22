# Frozen replication across tasks and new states

This extends the simulator-video pilot using FastWAM Optional IDM, bf16, on the
existing A100. It tests task 0 (soup/tomato, living-room scene 2) and task 7
(soup/cream cheese, living-room scene 1). Each task has two calibration states
(16,17) and eight prespecified test candidates (20–27). These test IDs have not
been used in the preceding local E1 development runs. This does not imply that
the model has never seen related data during training or that no teammate has
ever evaluated those benchmark initial states.

`configs/experiments/e1_visual_replication.json` is frozen before calibration.
`scripts/e1_visual_replication.py` enforces the following stage boundaries:

1. Build calibration references and fresh exact simulator replays. Both target
   approaches must satisfy the existing 2 cm goal error, 1 mm object-motion,
   zero robot–scene contact and 10 cm target-distance-advantage limits. Render
   all three predefined colour strengths with identical physics/camera/depth.
2. Encode all calibration videos using the native temporal VAE. Inspect every
   source/decoded clip. No donor policy action is executed in calibration.
3. Select one valid colour strength using only its latent RMS relative to the
   motion contrast, averaged over calibration states/targets. The candidates
   multiply tabletop red and blue channels by 0.9, 0.75 or 0.5. Choose the level
   nearest a ratio of one in mean absolute log-ratio; ties favour the weaker
   level. Record all candidates. This is magnitude calibration, not semantic,
   salience or spatial-support matching.
4. Freeze the selected strength and its evidence. Only then build test
   references. A failed test pair is excluded without replacement, with both
   references and reasons retained. Require at least four eligible states per
   task; otherwise stop before policy inference.
5. Run native, exact-identity and native RGB-round-trip controls for both
   current-image appearances and both action seeds. The maximum is 192
   control branches. Any baseline failure stops all donor execution.
6. Visually review every encoded test future before outcomes. Freeze all
   exclusions. Execute the full target-motion × future-appearance × current-
   appearance factorial for eligible states, with two recipient seeds. This
   contributes at most 256 donor branches, for a maximum of 448 total.

The source, protocol, reference files, codec files, calibration decision and
review are hashed at their stage boundaries. No silent restart, missing review
or source change is allowed. Independent human review is pending; AI-assisted
screening enables exploratory work under the user's standing authorization.

The robot always receives the task's original instruction. Donors contain only
simulator RGB; no donor language, target label, coordinates or actions enter the
policy. All branches execute 32 controls without replanning. Current-image tint
is an input intervention, while physical simulator execution stays unchanged.
Robot–scene contacts are sampled at every integration step, and object poses,
gripper state and simulator state are recorded live alongside camera/EEF data.
Edited depth buffers are now saved as well as compared during rendering.

The primary contrast is target-b minus target-a endpoint movement along the
initial a-to-b tabletop axis, for original current and future appearance.
Average recipient seeds within state. Report each task separately and an
equal-task average with a state-stratified percentile bootstrap. Report
appearance effects, appearance-consistency interactions, trajectory/orientation
alignment, contacts, exclusions and input magnitudes as secondary outcomes.

Sixteen candidate test states is a bounded replication, not a justified sample
size for a final confirmatory claim. The baseline's 1 cm endpoint / 5 mm path
limits are engineering checks; they are not a validated task-success or
semantic-equivalence margin. Do not infer invariance from a nonsignificant
appearance effect or treat repeated seeds as independent states.

The reference helper inherits legacy descriptive fields from its original
single-task configuration. The active task, target mapping and state are the
explicit arguments/metadata in each case; the replication protocol controls
the partition and candidate IDs. No historical result is used as the native
baseline for a new test state.

## Prepared Linux commands

The existing checkpoint, Wan assets, statistics and LIBERO installation are
required. These commands do not download or replace them.

```bash
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export LIBERO_CONFIG_PATH="$PWD/results/libero_config"
export DIFFSYNTH_DOWNLOAD_SOURCE=huggingface
OUT=results/new_visual_replication
.venv-e1/bin/python scripts/e1_visual_replication.py prepare --output "$OUT"
.venv-e1/bin/python scripts/e1_visual_replication.py calibration-references --output "$OUT"
.venv-e1/bin/python scripts/e1_visual_replication.py calibration-codecs --output "$OUT"
# Inspect every calibration sheet and fill calibration/codec_review.json.
.venv-e1/bin/python scripts/e1_visual_replication.py select --output "$OUT"
.venv-e1/bin/python scripts/e1_visual_replication.py test-references --output "$OUT"
.venv-e1/bin/python scripts/e1_visual_replication.py controls --output "$OUT"
# Inspect every test sheet and fill test/codec_review.json.
.venv-e1/bin/python scripts/e1_visual_replication.py donors --output "$OUT"
```

Use a fresh output directory. Failures and rejected states are results, not
instructions to retry until accepted. Local checks: 102 E1 tests passed,
including leakage guards, outcome-free calibration, state-level aggregation
and tracking-hook cleanup on failure.

## Reusing the saved results without loading the model

The result bundle includes `analyze.py`, `audit_controls.py`, `plot_results.py`
and `report.py`. Run them from a checkout matching the recorded source snapshot using
the configured Python environment and the full extracted bundle. The numeric
audit imports only NumPy-based experiment helpers; it does not load Torch,
MuJoCo, Robosuite or model weights. Plotting additionally requires Matplotlib.
`report.py` consumes an existing `replay_audit.json`; it does not perform replay.

```bash
python results/e1_visual_replication_20261121/audit_controls.py
python results/e1_visual_replication_20261121/analyze.py
python results/e1_visual_replication_20261121/plot_results.py
python results/e1_visual_replication_20261121/report.py
python results/e1_visual_replication_20261121/factorial_report.py
```

These commands rewrite derived audits/reports, so retain the supplied archive
and use a separate extracted copy when reproducing them. `replay_audit.py` needs
the pinned simulator environment and is intentionally not restartable over
existing replay records. Generating new policy outputs needs the recorded
checkpoint, base assets and GPU environment.

For a new study, reuse the NumPy functions in
`wam_interp.experiments.controlled_replication` and the video/physics capture
helpers in `simulator_video` and `simulator_libero`, then freeze a fresh protocol
and output directory. The experiment's existing source archive is evidence;
do not extract it over a working checkout with unrelated edits.

The independent-review page shows source/decoded clips without policy outcomes
and leaves all decisions unselected. A later human review is a separate,
post-hoc artifact and must not replace the pre-outcome AI screening record.

`FACTORIAL.md` and `factorial_summary.json` report all prespecified current- and
future-appearance conditions, including their interaction. The compact
`summary_bundle.zip` is for discussion and includes the numerical summaries
and figures. The full `replication_completed.zip` is needed to audit raw arrays;
neither bundle contains model weights or simulator assets.
