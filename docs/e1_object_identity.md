# Validate target reassignment before testing future-video semantics

The preceding material-only pilot did not establish a perceived identity swap.
This experiment tests complete native visual assets: translate the soup and
tomato visual meshes between XY slots while retaining their own UVs, textures,
materials, orientations and height. Only renderer-visible geometry positions
change inside a temporary context. Restore them before any simulation step.
The physical collision geometry and simulator object identities remain original.
This is a short causal input probe, not a benchmark success evaluation.

`ObjectAssetCapture` records original, full-asset-swapped and material-only RGB
inside the native sensor call. It requires exact render restoration, fixed
robot/camera/physics state, and unchanged unambiguous robot pixels. Unlike a
material swap, a full mesh swap legitimately changes object depth and silhouette;
depth and RGB outside the union of edited-object supports must remain exact.
See MuJoCo's [scene-update documentation](https://mujoco.readthedocs.io/en/3.3.2/programming/visualization.html).
The installed Robosuite renderer was also inspected: it calls `mjv_updateScene`
from the current `mjData` immediately before rendering.

Six layouts are fixed in `configs/experiments/e1_object_identity.json` before
policy inference. They use initial robot states 34–39 of LIBERO Object task 0,
with independent explicit XY offsets of up to 3 cm for soup, tomato and cream.
All other objects stay in place. These are six synthetic arrangements on one
table, not six independently sampled scenes. Warm-up remains 30 controls;
every branch reconstructs the simulator, reapplies the same layout and repeats
warm-up. Saved physical positions verify that the arrangements actually differ.

## Development validation

States 34–35 are development layouts. For both action seeds (0 and 1), run the
native policy under soup, tomato and untouched cream instructions with original,
full-asset-swapped and material-only current images. Repeat original-image
branches with an identity intervention. This is 48 rollout branches, each 32
controls with no replanning, using FastWAM Optional IDM in bf16 with ten steps.
There is no donor instruction and no replacement future in the native branches.

Both layouts must pass both seeds, under the full-asset edit:

- Original tomato-minus-soup endpoint separation along their original XY axis
  is at least 80 mm; after the edit it is at most -80 mm.
- Soup shifts at least +80 mm and tomato at most -80 mm under the identical edit.
- The untouched cream endpoint changes by no more than 20 mm in 3D.
- No robot–scene contact occurs, and physical object motion is at most 1 mm.

These are prespecified operational criteria for noun-selective target
reassignment, not proof of an internal identity code or a statistical
equivalence test. Current-image edits also change the native generated future.
The material-only condition is a diagnostic; it cannot replace a failed
full-asset condition. Failure stops the future-video experiment. Spatial priors,
cue recognition, instruction dependence and counterfactual image compatibility
remain possible explanations for a failure.

## Held-out fixed-motion test, conditional on validation

States 36–39 use four different frozen layouts. Reuse the fixed scripted soup
approach, crossed with original/full-asset-swapped future videos and original/
swapped current images. The same three recipient instructions and two seeds are
used, with the recipient instruction fixed within every future-video pair.

Run 144 native/identity/round-trip control branches first. The same operational
identity criterion must pass on at least three of the four layouts, in both
seeds, before future interventions. Retain all four layouts in the primary
future analysis even if one fails native validation, and report that failure.
No outcome-based replacement is permitted. All reference geometry, baseline,
codec and actual pre-outcome visual-review gates must also pass. The existing
10 mm endpoint/5 mm path round-trip limits and decoded-gripper registration
limits stay fixed. Then run all 96 future interventions.

The primary future statistic remains half the soup-minus-tomato response to the
same future swap, projected along the original target axis, averaged across
current layouts and seeds within physical arrangement. Bootstrap arrangements,
not frames or seeds. Report every factor cell, untouched-target responses,
native validation, contacts and all exclusions. Independent human review remains
pending when screening is AI-assisted. Full assets change shape and appearance
together; a successful result does not isolate which visual property mattered.

## Commands

Use the existing pinned Linux/A100 environment and checkpoints, with
`MUJOCO_GL=egl`, `PYOPENGL_PLATFORM=egl`,
`LIBERO_CONFIG_PATH="$PWD/results/libero_config"`, and
`DIFFSYNTH_DOWNLOAD_SOURCE=huggingface`.

```bash
OUT=results/e1_object_identity_new
.venv-e1/bin/python scripts/e1_object_identity.py prepare --output "$OUT"
.venv-e1/bin/python scripts/e1_object_identity.py references --output "$OUT"
# Actually inspect all six source arrangements and fill source_review.json.
.venv-e1/bin/python scripts/e1_object_identity.py development --output "$OUT"
# Continue only if development_validated; otherwise report the failed gate.
.venv-e1/bin/python scripts/e1_object_identity.py controls --output "$OUT"
# Inspect all decoded clips and fill heldout/codec_review.json before donors.
.venv-e1/bin/python scripts/e1_object_identity.py donors --output "$OUT"
```

The runner snapshots all inference source and both partition protocols before
reference construction. Changing scientific thresholds or source requires a new
frozen run; keep the failed run. Generated evidence, weights and simulator assets
stay outside Git. Local regression suite after implementation: 112 tests passed.

The first frozen run stopped before policy inference: two reference approaches
ended 2.38 and 2.72 cm from their goals while their Y commands remained capped
at 0.6. Version 2 changes the common reference controller to a 12-control ramp
and 0.8 cap for all six layouts. All six paths then passed. Layouts, horizon,
2 cm approach threshold and behavioural criteria are unchanged. The failed
version is retained. Version 2 also records explicit current-image, proprioception,
initial-state and layout hashes before each native inference condition.
