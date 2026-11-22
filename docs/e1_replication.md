# Exploratory E1 replication and contact follow-up

This extends the completed three-state pilot. All new results remain exploratory:
the user authorized execution while independent human review is deferred.
The review page for the earlier pilot is in the local results directory at
`results/e1_a100/native_counterfactual/independent_review.html`.

## Fixed design

The authoritative configuration is `configs/experiments/e1_replication.json`.
It selects task 0, previously untested initial states 3 through 12, recipient
action seeds 0 and 1, and donor generation seeds 2 through 9. Donor seeds are
distinct from the recipient seeds. The primary pair is always
`soup_tomato_seed_2` and `cream_butter_seed_2`; invalid semantics reject a state
rather than trigger selection using policy outcomes.

The original instruction stays fixed in all recipient arms. Prepare 32 native
donors per state using the four previously specified instruction recipes. Reuse
that bank across recipient seeds, with separate baseline futures and per-seed
latent-difference measurements. Context and content hashes guard this reuse.

A fifth arm is available only when an additional same-target native donor
passes all matching criteria against the alternative donor:

- Future-latent difference norm within 20%, excluding the observed anchor.
- Visible-palm image-space path length and endpoint displacement within 15%.
- Maximum difference in cumulative normalized movement progress at the nine
  aligned frames no greater than 0.15.
- All nine palm coordinates visible and visually verified; no interpolated
  missing landmarks. Semantic validity and nuisance notes are also required.

Among eligible controls select the smallest maximum normalized discrepancy;
break ties by candidate ID. The primary same-target donor is not eligible as
the additional control. Every norm-eligible candidate requires a semantic
review, including candidates rejected as unclear or invalid. Motion matching
is evaluated only for accepted same-target candidates; a norm failure alone
already excludes a candidate from the joint criteria.

Freeze all state/seed selections before executing any replication effects.
Save all rejections and denominators. A no-match result permits the four-arm
exploratory comparison but **must not be described as a matched experiment**.
These tolerances are operational design choices, not guarantees that all
confounds have been controlled. An audit of the original pilot found that
alternative donors had about 2.5–2.7 times the same-donor latent-change norm.

## Commands on the prepared server

Run from `/home/ubuntu/wam-e1-20261119` after overlaying the matching source:

```bash
source .venv-e1/bin/activate
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
export LIBERO_CONFIG_PATH="$PWD/results/libero_config"
export DIFFSYNTH_DOWNLOAD_SOURCE=huggingface
python scripts/e1_replication.py prepare --output results/e1_replication_20261120
# Inspect all primary donors and any norm-eligible possible controls.
# Complete state_*/selection_review.json, including actual review method.
python scripts/e1_replication.py run --output results/e1_replication_20261120
python scripts/e1_replication.py contact --output results/e1_replication_20261120
```

Do not mutate inference source/config files between preparation and execution:
the source fingerprint is part of the recipient context. Use a fresh output
for a changed protocol. A live job must be polled by its PID; an unchanged log
or observation timeout is not permission to restart it. Interrupted acquisitions
and executions require inspection rather than silent overwrite.

`prepare_status.json`, `run_status.json`, and `contact_status.json` record stage
outcomes. Preparation never executes donor actions. The complete recipient
baseline/identity simulator arrays must match before effects are accepted.
CPU test doubles verify contracts; only server runs validate real inference.

## Contact follow-up

For fixed states 3, 4, and 5, execute the native policy sequentially for at most
12 chunks of 32 actions, stopping on simulator termination or task success.
At each replan, capture its clean imagined video before execution, then record
the actual EEF, gripper joints, both cameras, object body positions, and contact
geom names at every executed control. Use the native preprocessing for paired
imagined/actual sample frames at controls 0,4,...,32 within each chunk.

For the first object-motion correspondence pilot, inspect the earliest chunk
per episode in which a task target (alphabet soup or tomato sauce) moves at
least 1 cm. Retain all chunks and explicitly report cases with no target motion
or insufficient visibility. This conditional sample does not estimate overall
video-prediction accuracy. Movement or a collision is not proof of a grasp.
Review contact geometry and visuals before calling a segment a grasp.

Compare the same visible object/gripper landmarks in the same camera at the
same timestamps. A world-space object body center is not interchangeable with
an image-space cap landmark. Do not silently use simulator positions as imagined
coordinates or annotate through occlusion. Report missing coverage and a
stationary-object baseline alongside ADE/FDE. Keep world-space executed traces
as separate exact measurements.

## Analysis and remaining review

Average the two recipient-seed contrasts within each initial state first, then
report across states with explicit accepted/rejected counts. The two seeds are
repeated measurements, not additional independent states. Keep matched and
unmatched contrasts separate. Archive the fixed protocol, selection freeze,
source, asset identities, raw traces and annotations with any report.

Independent human review remains pending until an actual human supplies it.
AI-assisted donor or landmark inspection must never be relabeled as that review.
No automatic PR, push, or external message is part of these experiments.

The 2026-11-20 run retains world-y displacement as its predefined primary
outcome. After inspecting the first completed state, a secondary analysis of
full 3D endpoint and trajectory separation was added because movement changed
substantially along another axis. This is explicitly post-hoc: its timestamp,
trigger and estimators are recorded in `secondary_analysis_plan.json` beside
the results. It must not replace the primary outcome or be described as
preregistered. Neither measurement alone establishes target-specific semantics.

## Completed exploratory run

The 2026-11-20 A100 run completed all 20 state/seed comparisons (80 branches),
with exact identity replay and verified future insertion. Nine of ten state
means had the expected world-y sign; the mean contrast was -3.01 cm. The
post-hoc 3D endpoint separation averaged 14.20 cm. No additional native control
passed every matching criterion, so target-specific semantic causality remains
unestablished.

All three native contact episodes reached simulator task success. Approximate
AI-assisted gripper annotations covered all 27 paired frames, with 1.70 px mean
error versus 15.82 px for the stationary baseline. Object-lid visibility covered
only six stationary-phase pairs and none of the four sampled pairs at the
1 cm motion threshold; moving-object correspondence is inconclusive. Independent
human review remains pending by the user's explicit choice.

The local [result report](../results/e1_replication_20261120/README.md) contains
the full denominators, limitations, figures and evidence links. The results
directory is not part of an automatic Git commit or PR.
