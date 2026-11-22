# E1 equal-norm diagnostic and visible-object follow-up

This is a development follow-up to the completed ten-state replication. It
addresses unequal latent perturbation sizes and object-landmark visibility.
Independent human review remains pending by agreement with the user. It does
not replace the native-donor replication or establish a semantic mechanism.

## Controlled latent experiment

`configs/experiments/e1_equal_norm.json` fixes states 3, 4 and 5, recipient
seeds 0 and 1, and the same reviewed native donor IDs used previously. These
are previously used states, not a held-out confirmation set. The recipient
prompt, current observation, checkpoint, inference options and simulator
initialization remain fixed.

For recipient baseline future B, original-target donor S and alternative donor
A, compute future-only differences S-B and A-B. Exclude latent time index zero,
which represents the observation. Define R as the smaller difference norm.
At radii 0.5R and R, insert B plus a direction scaled to that radius. The four
directions are S-B, A-B and two Gaussian directions with fixed seeds 101 and
102. All eight interventions are frozen before any new policy effects. The
random directions are fixed controls, not a sample sufficient to characterize
the distribution of noise sensitivity.

Each state/seed case also includes baseline, identity and both full native
donors: twelve branches per case, 72 in total. No radius extrapolates either
native donor ray. Nonetheless, interpolation and random changes can produce
off-manifold or semantically diluted futures. Equal global Euclidean norm
does not match spatial, temporal or channel-specific changes.

The runner checks realized bf16 perturbation size and direction, anchor
preservation, finite outputs and exact command prefixes. It requires exact
historical baseline actions and simulator arrays, and exact baseline/identity
replay. The final independent array audit also compares both native donor
branches against the earlier experiment, including all stored camera frames.

The primary outcome is alternative-minus-original EEF world-y displacement
at each common radius. Full 3D endpoint/trajectory separation and the two
random-direction controls are declared secondary measurements. Average the
two action seeds within each state before aggregation. Do not treat seeds,
frames or controls as independent states.

## Reproduce on the prepared Linux server

Run inside the prepared repository with the same checkpoint, base assets,
statistics and LIBERO installation. Copy the new source files before prepare;
do not change source while the run is active.

```bash
source .venv-e1/bin/activate
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
export LIBERO_CONFIG_PATH="$PWD/results/libero_config"
export DIFFSYNTH_DOWNLOAD_SOURCE=huggingface
python scripts/e1_equal_norm.py prepare --output results/e1_equal_norm_new
python scripts/e1_equal_norm.py run --output results/e1_equal_norm_new
```

Preparation requires the earlier replication's raw baseline camera traces,
donor latents and provenance. These are retained on the original server;
the old compact result bundle intentionally omits some camera streams. Missing
inputs fail before model loading. Preparation freezes source and every input
with hashes. Existing outputs and interrupted executions cannot be silently
overwritten or restarted. Poll the original process until it exits.

CPU validation:

```bash
python -m pytest tests/test_e1.py tests/test_e1_handoff.py \
  tests/test_e1_libero.py tests/test_e1_replication.py tests/test_e1_equal_norm.py -q
```

## Object motion from existing episodes

Export paired views from every completed contact chunk without new inference:

```bash
python scripts/e1_contact_visibility.py \
  --source results/e1_replication_20261120 \
  --output results/e1_visibility_new
```

The exploratory measurement uses chunk 2 in states 3, 4 and 5, following the
earlier first-motion chunk. It tracks the visible lower rim of the soup can
in the external camera. This is a view-dependent image contour, not the old
cap landmark, a fixed material point or a 3D object center. The wrist camera
was inspected but does not show this landmark reliably; coordinates are never
mixed across cameras.

Selection was made after viewing saved videos, before numerical measurement.
An enlarged-pixel QA pass rejected occluded and ambiguous outlines and corrected
off-center preliminary marks. Both drafts are preserved. The final dataset has
18/27 identifiable pairs, including 17/24 sampled moving pairs. Six generated
outlines are ambiguous near a background object; three pairs are occluded.
Missing generated boundaries may themselves reflect prediction failures, so
conditional coordinate error must always be accompanied by coverage.

The stationary baseline uses the actual initial landmark and is unavailable
when that initial landmark is occluded. It is available for only state 4; do
not compare its average against all three episodes as if denominators matched.
Annotations are AI-assisted, with a declared 3-pixel per-point sensitivity
radius, not a calibrated confidence interval. Human review remains pending.

The actual run's source snapshots, complete audits, plots, raw-file indexes
and review artifacts are stored under `results/`. No push or PR is automatic.
