# Controlled simulator-video futures

This experiment replaces prompt-generated donors with scripted, physically
executed simulator continuations. Only their camera videos enter the native
video VAE. No donor language, controller commands, simulator coordinates, or
target labels are passed to the recipient policy. The policy keeps the original
instruction to put both soup and tomato in the basket.

The positive factor is **which target region the gripper approaches**. It is a
motion/region composite, not isolated object identity or a successful full task.
The appearance factor multiplies the tabletop material's red and blue channels
by 0.9, leaving green and alpha unchanged. A weaker 0.95 variant is also recorded
in the reference bank, but is not part of the first policy pilot. This is a broad
surface tint, not a localized texture patch or a magnitude-matched intervention.

## Reference construction and checks

`scripts/e1_simulator_pairs.py` creates six references: states 3/4/5, soup/tomato
approach, 32 controls with frames sampled at 0,4,...,32. The native OSC controller
receives bounded position commands, no rotation commands and an open-gripper
command. Goals are 14 cm above each object's center. These are pre-contact
approaches; they do not demonstrate a complete grasp or task.

`src/experiments/simulator_libero.py` captures all appearance variants inside the
native camera sensor call, since a direct render after `env.step()` need not
match the cached image returned by that step. Every original capture must equal
the evaluator observation exactly. Material changes are restored in `finally`;
re-rendering the original must be exact. RGB variants share exact physics,
camera poses, depth buffers and sensor timestamps. Depth is the native render
buffer, not a claimed metric-depth estimate.

Segmentation hides RGB-invisible diagnostic sites. Table coverage is computed
from geometry IDs rendered at 4x resolution and max-pooled to the native camera
grid; a fixed 2-pixel boundary allowance handles antialiasing. All changed RGB
pixels must fall within that support. Both cameras must have a visible edit.
The native RGB renderer and observation convention are unchanged.

Each trajectory is replayed with the exact recorded commands in a fresh,
unedited simulator. All captured cameras, proprioception and recorded physics
arrays must match. Initial images/EEF state must also match the preceding native
experiment. Robot–object/table contacts are checked at every integration step.
Acceptance requires <=2 cm endpoint goal error, <=1 mm object translation, zero
robot–scene contact steps, and >=10 cm advantage in final XY distance to the
intended versus other target. These are development validity gates.

Two rejected attempts are preserved. The first exposed invisible-site mask
holes and a slow soup approach. The second fixed the controller and sites but
the coarse mask still missed a few subpixel table fragments. The third uses
4x segmentation coverage and passes all six references. Controller timing was
repaired before any policy results; no acceptance threshold was relaxed.

## Policy factorial

`scripts/e1_simulator_policy.py` crosses two reference motions, two future
appearances, and two current-image appearances. Current-image and future tint
are independent factors. This includes both matched appearance and mismatch
conditions. All physical executions start from the same original simulator;
the current-image tint is applied to the prepared policy input only. The window
is open loop, so subsequent camera observations do not feed back into these
32 actions. This is not a closed-loop edited-scene benchmark.

`encode_reference_video()` in `src/experiments/simulator_video.py` normalizes
native uint8 simulator frames using FastWAM's observation arithmetic, jointly
encodes all nine frames, and restores the exact recipient observed latent
anchor. The first source frame is explicitly selected to match the current
appearance. Repeated encoding must be exact. Decoded clips are inspected before
donor policy execution; the geometry record alone cannot validate what the VAE
preserves.

The bounded pilot is 84 branches: 3 states x 2 recipient seeds x 2 current
appearances x (native, identity, native round-trip, four donor variants).
All 36 native/identity/round-trip controls run first. Every round-trip control
must stay within the existing 1 cm endpoint / 5 mm mean-path engineering bounds.
An actual AI-assisted review is required before the 48 donor branches. Human
review is deferred for this exploratory work. A failure is preserved and stops
the dependent stage; runners do not silently restart.

The primary readout is the tomato-minus-soup EEF endpoint contrast projected
onto the initial soup-to-tomato direction in the tabletop XY plane. Report it
for each current/future appearance cell, averaging seeds within state. The
appearance readout compares future tint versus original at fixed motion and
current image, including its interaction with current-image appearance. Report
pixel/latent distances and individual arms. No statistical equivalence or
general invariance claim follows from a small or nonsignificant effect here.

## Prepared Linux environment

Use the existing FastWAM Optional IDM bf16 checkpoint, base assets, statistics,
LIBERO environment and previous native results. Paths are in the two JSON
protocols under `configs/experiments/`. Nothing downloads weights or assets.
Choose fresh output directories; the example reference path is the accepted
development build, not a command to overwrite it.

```bash
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export LIBERO_CONFIG_PATH="$PWD/results/libero_config"
export DIFFSYNTH_DOWNLOAD_SOURCE=huggingface
.venv-e1/bin/python scripts/e1_simulator_pairs.py --output results/new_reference_build
# Set the policy protocol's references path to the accepted, reviewed build.
.venv-e1/bin/python scripts/e1_simulator_policy.py controls --output results/new_policy_run
# Inspect all source/decoded sheets and fill codec_review.json with actual evidence.
.venv-e1/bin/python scripts/e1_simulator_policy.py run --output results/new_policy_run
```

All reference/source/input hashes, simulator revisions, model context, review
status, failures, native controls, action arrays and execution traces are kept.
These three states are development data. Semantic selectivity across properties,
task success, language-path effects, and other architectures remain separate
questions.

## Observed development run, 21 November 2026

The A100/bf16 run `results/e1_simulator_policy_20261121` completed all 84 branches.
All 12 native round-trip gates passed (0.947–2.038 mm endpoint drift), with exact
identity interventions and exact historical native baselines. All 24 reference
codecs received AI-assisted visual review before the 48 donor branches;
independent human review remains pending. The local E1 test set passed 97 tests.

For original current and future appearance, the signed tomato-minus-soup shift
was 19.343, 17.564 and 13.491 cm in states 3, 4 and 5 respectively, averaging
16.799 cm after averaging seeds within state. All 24 state/seed/appearance
contrasts had the expected sign. Future tabletop tint produced 1.453–2.349 mm
mean 3D endpoint changes across the four fixed-motion/current-image cells.
This is a single mild nuisance factor with unmatched pixel and latent edit
magnitudes, not evidence of general invariance or isolated object semantics.

Alignment is incomplete: mean error to the corresponding simulator reference
endpoint was 21.695 mm for soup and 44.304 mm for tomato. The tomato-conditioned
path under-reaches its reference, even though its endpoint is closer to that
reference than to the soup reference. These are pre-contact approaches, not
full task-success measurements or generated-video 3D trajectory estimates.

The result folder contains the complete report, numerical analysis, per-branch
arrays, source and protocol snapshots, visual review, reference audit, and
deterministic contact replay audit. `audit_and_analyze.py`, `audit_references.py`
and `replay_contacts.py` there reproduce the saved-data checks in the prepared
Linux environment; the replay refuses to overwrite existing audit records.
The final evidence archive also retains both rejected reference builds.
