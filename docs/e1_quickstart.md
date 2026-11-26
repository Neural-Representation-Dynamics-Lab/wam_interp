# E1 pilot handoff

This is the FastWAM Optional IDM / LIBERO pilot. E1 runs first; coordinate
comparison follows. Other backends and a definitive research model choice are
outside this first runnable implementation.

This guide reproduces the original pilot. For current priorities, the modular
tooling map and cross-model planning commands, start with
[e1_multimodel.md](e1_multimodel.md). Historical output folders should not be reused.

## Transfer

1. Copy the latest source-overlay ZIP to the Linux GPU machine.
2. Clone the team repository and check out the base revision in
   `HANDOFF_SOURCE.json` inside the ZIP.
3. Extract the ZIP over that clean checkout, retaining `.git`.
4. In the checkout, run `python3 scripts/verify_e1_handoff.py`.

## Install and prepare

```bash
bash scripts/setup_e1_linux.sh
source .venv-e1/bin/activate
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
export LIBERO_CONFIG_PATH="$PWD/results/libero_config"
export DIFFSYNTH_DOWNLOAD_SOURCE=huggingface
python scripts/e1_libero.py check
python scripts/e1_libero.py prepare --output results/e1_live/prepare
```

Python 3.10 and a compatible NVIDIA driver are prerequisites. The installer
downloads the released checkpoint and pinned official Wan VAE, text encoder,
and tokenizer assets. The pilot uses these originals because the optional
DiffSynth converted-weight mirror returned an access error during deployment.
Headless LIBERO rendering also requires EGL and NVIDIA's graphics userspace
library matching the installed driver. A compute-only driver installation can
pass `nvidia-smi` yet fail EGL initialization. On the tested Ubuntu server,
`libegl1`, `libosmesa6`, and `libnvidia-gl-580-server=580.105.08-0lambda0.22.04.1`
provided rendering for driver 580.105.08; select the matching package for your
host rather than installing this version blindly. Check that
`/usr/share/glvnd/egl_vendor.d/10_nvidia.json` exists.
Read `configs/experiments/e1_libero.json` before running: its defaults are one
task/state, native bf16, 32 actions, eight candidate seeds and world-x gripper
displacement as the behavioral measure.

## Review, then E1

For the separately labeled counterfactual-instruction protocol, use
`--config configs/experiments/e1_counterfactual_libero.json` on both `prepare`
and `run`, with fresh output directories. This task-0 configuration asks the
donor generator to follow different target sets using native benchmark wording
(soup/tomato, cream-cheese/butter, soup/cream-cheese, plus reordered tomato/soup),
while keeping the recipient policy's original task instruction, image,
proprioception, and action seed fixed. It measures world-y gripper displacement,
since the targets are separated primarily along that axis. Donor prompts and seeds are recorded
in each candidate's immutable generation provenance. Review the generated
motion itself; a changed prompt is not evidence that the model obeyed it.

This is a whole-future replacement experiment. It does not isolate a single
semantic variable: target choice can covary with motion magnitude, timing,
object appearance, and other latent content. Report it separately from
same-instruction seed variation, and do not infer semantic specificity from
an action change alone. Initial server review used an AI-assisted visual
inspection; independent human review remains advisable for research claims.

Inspect decoded baseline and candidate frames. In `prepare/review.json`, assign
semantic labels, accept valid candidates, note nuisance differences, select
same/opposite donors and the expected effect sign, and verify frame timing.
The program refuses incomplete review. If no valid contrasting donor exists,
record that outcome instead of forcing a selection.

```bash
python scripts/e1_libero.py run --prepared results/e1_live/prepare --output results/e1_live/run
```

## Trajectory follow-up

Open each `run/*_frame_pairs.html`, mark gripper/object landmarks or missing
points, and save the exported annotation JSONs into the run folder. Draft export
and import are supported. Then:

```bash
python scripts/analyze_e1.py --annotations results/e1_live/run/baseline_annotations.json results/e1_live/run/identity_annotations.json results/e1_live/run/same_property_annotations.json results/e1_live/run/opposite_property_annotations.json --e1-summaries results/e1_live/run/e1_summary.json --output results/e1_live/analysis.json
```

For multiple cases use `scripts/e1_campaign.py prepare` then `run`, both with
`--output results/e1_campaign`. Review every case between those stages.

## What is verified

CPU protocol and failure tests, local CUDA bf16/RNG checks, and browser
annotation/export checks. On 2026-11-19, the counterfactual-instruction pilot
also completed real FastWAM Optional IDM inference and LIBERO simulation on an
A100 40GB in bf16: task 0, initial states 0/1/2, four arms per state, 32 controls
per arm. Observer and identity action errors were zero; identity execution
poses and rendered frames matched the baseline exactly. All exported action
and latent arrays were finite. This validates this narrow runtime path, not
other models, full task success, or benchmark performance.

The generated report is `results/e1_a100/native_counterfactual/README.md` in
the experiment owner's local checkout (results are intentionally not included
in the source overlay). It records the exploratory effects, approximate
AI-assisted image annotations, failed donor searches, and remaining research
controls. Independent semantic/landmark review and broader replication remain
necessary before publication claims.

Full protocol, data contracts and troubleshooting: [e1_setup.md](e1_setup.md).
