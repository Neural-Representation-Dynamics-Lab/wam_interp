# Native donor search after the equal-norm diagnostic

This exploratory follow-up uses actual FastWAM Optional IDM samples without
interpolating or rescaling their video latents. It addresses the semantic
dilution seen when the previous diagnostic interpolated alternative futures.
Independent human review remains pending; AI-assisted exploratory review was
authorized by the user. No automatic push or PR is part of the experiment.

## Frozen acquisition

The authoritative protocol is `configs/experiments/e1_native_search.json`.
Use LIBERO-10 task 0 and previously unused initial states 13, 14 and 15. The
recipient always receives the original soup-and-tomato task instruction. Four
single-target instruction variants for soup and four for cream cheese, each
with donor seeds 10–13, produce 32 native futures per state and 96 overall.
Recipient action seeds are 0 and 1. Donor actions are discarded and never
executed or used to select candidates.

Native generation does not guarantee semantic validity. Some instruction
variants may be outside the training instruction distribution. The actual
decoded video, not the prompt name, determines its provisional visual label.

## Bounded matching before policy effects

For each state, compare every intended soup/cream pair's future-only latent
difference norm from each recipient baseline. Both recipient seeds must meet
the 20% norm tolerance. Rank eligible pairs by the largest normalized norm
discrepancy across seeds, breaking ties by candidate IDs. The first eight pairs
per state form the fixed visual-review budget. Preserve the entire bank,
all pair rankings and the count of eligible pairs outside that shortlist.

Review all nine native frames and both camera views of every shortlisted
candidate. Full-resolution review sheets use neutral candidate aliases;
the mapping and instruction provenance remain available for audit. This is
not a claim of independent or fully blinded review. Require a clear approach
to the intended object and reject unclear semantics, major generated appearance
changes or unidentifiable motion landmarks.

For semantically accepted candidates, place the same visible palm-housing
corner in the nine external-camera frames. No missing coordinates may be
interpolated. Match path length and endpoint displacement within 15%, and
maximum cumulative normalized motion-progress difference within 0.15. These
are image-space descriptors, not proof that all nuisance variation is removed.
Numerically eligible pairs also need explicit pair-level gripper-orientation
and scene-appearance review. Manual/AI coordinates require human validation;
near-threshold decisions should be described as provisional.

Select the passing pair with the smallest maximum normalized discrepancy
across matching criteria and recipient seeds, then candidate IDs. Freeze all
state selections before any recipient comparisons. Every shortlisted candidate
must be reviewed even if another pair fails. Do not execute unmatched pairs.
A no-match result means no qualifying pair in this bounded shortlist; it does
not establish that no match exists elsewhere in the bank or model distribution.

## Execution and estimands

For each qualifying state and recipient seed, execute baseline, identity,
native soup donor and native cream donor from independently recreated identical
simulator/controller state. Preserve the observed latent anchor. The existing
E1 runner verifies baseline-future reproduction and baseline/identity replay,
records actual command prefixes and saves all camera/EEF arrays. Audit exact
native donor insertion and finite outputs before interpreting an effect.

The predefined primary contrast projects the alternative-minus-soup executed
EEF endpoint difference onto the initial simulator-world XY direction from
the soup body to the cream-cheese body. Positive means a shift toward the
alternative object relative to the soup-donor branch. Record the initial body
positions and direction before policy effects. This defines the axis from
scene geometry, not observed action effects.

World-y contrast, 3D endpoint separation and mean 3D trajectory separation are
secondary. The existing per-case `e1_summary.json` retains its world-y measure;
the new primary is in `native_pair_comparison.json`. Average recipient seeds
within each state first. Do not pool these new projections with earlier
world-y outcomes or treat frames/seeds as independent states.

## Commands

From the prepared Linux repository with matching assets and environment:

```bash
source .venv-e1/bin/activate
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
export LIBERO_CONFIG_PATH="$PWD/results/libero_config"
export DIFFSYNTH_DOWNLOAD_SOURCE=huggingface
python scripts/e1_native_search.py prepare --output results/e1_native_search_new
# Review the frozen shortlist and complete state_*/review.json.
# Never label AI-assisted work as independent human review.
python scripts/e1_native_search.py select --output results/e1_native_search_new
python scripts/e1_native_search.py run --output results/e1_native_search_new
```

Source and protocol hashes are locked during preparation. Do not edit inference
source between preparation and execution. Output directories and interrupted
executions cannot be silently overwritten or restarted. Track the original PID.
No model loads for the run stage if no state qualifies.

Local tests:

```bash
python -m pytest tests/test_e1.py tests/test_e1_handoff.py \
  tests/test_e1_libero.py tests/test_e1_replication.py \
  tests/test_e1_equal_norm.py tests/test_e1_native_pairs.py -q
```
