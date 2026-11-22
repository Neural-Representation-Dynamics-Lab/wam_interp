# Fixed-motion material cues and recipient instruction

This bounded feasibility study uses FastWAM Optional IDM on LIBERO Object task
0 initial states 30–33. Its physical scene contains soup, tomato and cream cheese.
The three recipient instructions use official LIBERO Object wording but share
the same task-0 simulator state. They are diagnostic instruction conditions,
not three benchmark evaluations. Four states cannot establish a general effect.

For one scripted, non-contact approach, exchange the native material
IDs of soup and tomato only during camera rendering. Geometry, UV coordinates,
physics, robot path, gripper state and camera poses stay fixed. Original and
edited depth must be identical; unambiguous source robot-interior pixels and
restored rendering must be exactly identical. Mixed robot/object boundary pixels
within the existing two-pixel object margin are counted separately. Save full
and interior robot/gripper masks, segmentation and the original
and edited videos from the native sensor invocation.

This changes object-label/material cues. It is not guaranteed that applying a
native texture through the other object's UV map changes complete perceived
object identity. Inspect that explicitly after temporal VAE encoding; keep human
review marked pending when only AI screening has been completed.

The inference factorial has two future cue layouts, two current-image layouts,
one fixed depicted motion, three recipient instructions and two seeds. The
same visual edit is relevant under soup/tomato instructions and nominally
irrelevant under the untouched cream-cheese instruction. The recipient prompt
is constant within each paired visual intervention; no donor prompt is encoded.

There are at most 144 native/identity/RGB-round-trip control branches and 96
donor branches. All controls finish before donor execution. Frozen baseline
engineering limits remain 10 mm endpoint / 5 mm average path difference. Reject
invalid reference or codec cases before donor outcomes, without replacements,
and require at least three of four states. Every donor clip needs actual visual
review. No automatic acceptance or outcome-based exclusion is allowed.

Beyond exact source motion, compare decoded videos within gripper masks. Use a
small ±3-pixel registration search and a 5/255 mean absolute difference limit;
measurable patches must have at least 25 interior pixels and RGB standard
deviation at least 3. Require at least four measurable camera/frame patches per
pair, shifts no greater than one pixel, and report all unmeasurable patches.
These are image-space feasibility checks, not proof that internal robot features
are identical. The whole-video codec MAE limit is 5/255.

The primary statistic is half the difference between the soup-instruction and
tomato-instruction responses to the same future cue swap, projected onto the
original soup-to-tomato tabletop axis. Average both current layouts and seeds
within state before bootstrapping whole states for the fixed motion path. Report
each noun-specific direction, each factor cell, the cream instruction response,
native language responsiveness, contacts, reference alignment and magnitudes.

A positive interaction with the predicted opposite noun-specific signs would
support instruction-dependent use of these future appearance cues. It would
not prove complete object semantics: instruction-dependent gains and anomaly
responses remain alternatives. A null interaction is inconclusive if the cues
are not recognized or the fixed depicted motion dominates the policy. No
practical-equivalence threshold or full-task-success inference is claimed.

Future-only edits create current/future cue inconsistency. The complete current
by future layout factorial measures this dependence; averaging layouts does not
make each individual video physically continuous. Both geometry and physical
object identity remain original during execution, and only 32 controls execute
without replanning. These are causal input probes, not deployable rollouts or a
new benchmark score.

## Execution

Use the existing pinned Linux simulator environment, released Optional IDM
checkpoint and statistics. Set `MUJOCO_GL=egl`, `PYOPENGL_PLATFORM=egl`,
`LIBERO_CONFIG_PATH="$PWD/results/libero_config"`, and
`DIFFSYNTH_DOWNLOAD_SOURCE=huggingface`.

```bash
OUT=results/e1_material_cues_new
.venv-e1/bin/python scripts/e1_material_cues.py prepare --output "$OUT"
.venv-e1/bin/python scripts/e1_material_cues.py references --output "$OUT"
.venv-e1/bin/python scripts/e1_material_cues.py controls --output "$OUT"
# Actually inspect source/decoded videos, then fill codec_review.json.
.venv-e1/bin/python scripts/e1_material_cues.py donors --output "$OUT"
```

The source snapshot and protocol are frozen before reference construction.
There is no silent stage restart. Keep failed feasibility versions with their
errors; repair the instrument only in a newly frozen run. Model weights, caches,
simulator assets and generated arrays remain outside version control.

Rendering follows the installed Robosuite wrapper and MuJoCo 3.3.2's separation
of simulation state and visualization. See the [official visualization
documentation](https://mujoco.readthedocs.io/en/3.3.2/programming/visualization.html).
The intervention uses temporary `geom_matid` assignments around render calls,
with restoration in `finally`; no simulation step occurs under edited materials.

The first frozen reference attempt stopped with zero policy branches because
supersampled robot coverage included mixed object-edge pixels. In its state-30
diagnostic, 26 robot-covered pixels changed, all inside the object boundary
margin; zero changed outside it. Version 2 records that overlap and tests the
unambiguous interior separately. It does not alter/composite rendered pixels,
relax depth or geometry equality, or replace failed states based on behaviour.

Version 2 then stopped because the wrapper omitted returning the native sensor
image. Version 3 fixed that software error and added a regression test. Its
rendering checks passed, but every tomato approach stopped 7.9–8.8 cm from the
goal, failing the unchanged 2 cm endpoint gate. Version 4 narrows the pilot to
the soup approach, which passed in all four states; no policy outputs existed
when this change was made. Keep every failed version. The four initial states
share effectively the same object arrangement and vary robot initial conditions,
so intervals do not describe generalization across scenes. Local tests: 108 pass.

Pre-outcome inspection of version 4 also found that the material exchange
darkens the cans' lids as well as changing their side patterns. The small label
text is unreadable in these images. Acceptance is therefore for visible
material-cue differences and coarse motion preservation, not validated complete
object identity. Record this distinction in every readout; an interaction could
still reflect instruction-dependent sensitivity to these combined visual cues.
