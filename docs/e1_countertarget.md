# Two alternative native futures under a fixed recipient instruction

This is a separate adaptive follow-up to the native soup/cream search. That
search generated 96 futures and found zero norm-eligible pairs across its
768 intended-target pairs. Preserve those negative results and their original
source archive. The follow-up was designed after the first two states failed
the gate and before any recipient policy comparisons.

The new frozen protocol is `configs/experiments/e1_countertarget.json`.
Generate 48 additional native futures: three states (13,14,15), two instruction
variants each for cream cheese and butter, and donor seeds 14–17. Recipient
action seeds remain 0 and 1. There is no interpolation, latent rescaling or
additional bank expansion in this bounded follow-up.

Both donors now describe targets outside the unchanged original soup-and-tomato
recipient instruction. Comparing two alternative targets avoids treating a
near-baseline donor as the sole control for a much larger alternative change.
It tests short-chunk directional coupling, not the benefit of correct imagination
or task success. The recipient instruction is never replaced by either donor
instruction during policy comparisons.

Matching and review use the same numerical tolerances as the first search:
20% future-latent norm difference for both recipient seeds, 15% visible-palm
path length and endpoint displacement differences, and maximum normalized
movement-progress difference 0.15. Review at most the eight closest norm-eligible
pairs per state, retaining all other pairs and candidate imagery. Prompts only
specify intended targets; actual labels require native video review. Numerically
eligible pairs require orientation/appearance review. No unmatched comparisons
are executed. Independent human review remains pending by user agreement.

Freeze all selections before executing baseline, identity, donor A (cream) and
donor B (butter). Restore the environment/controller independently for every
branch and preserve the current observed latent anchor. Fail on any baseline
future mismatch, identity action/camera mismatch, changed native donor insertion,
nonfinite values or invalid action shape. Save executed command prefixes and
both camera streams. The matching JSON retains legacy `same`/`alternative`
keys internally; in this follow-up they refer to donors A/B and do not claim
that donor A matches the recipient's intended behavior.

The primary contrast is the butter-minus-cream EEF endpoint projected onto the
initial world-XY vector from the cream-cheese body to the butter body. Positive
means movement shifted toward the butter relative to the cream-donor branch.
Record that vector before outcomes. World-y, full endpoint separation and
trajectory separation are secondary. Average the two recipient seeds within
each state before aggregating; these are the same development states used in
the first search, not an independent confirmation dataset.

On the prepared Linux server:

```bash
source .venv-e1/bin/activate
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
export LIBERO_CONFIG_PATH="$PWD/results/libero_config"
export DIFFSYNTH_DOWNLOAD_SOURCE=huggingface
python scripts/e1_countertarget.py prepare --output results/e1_countertarget_new
# Inspect the frozen shortlist; complete state_*/review.json truthfully.
python scripts/e1_countertarget.py select --output results/e1_countertarget_new
python scripts/e1_countertarget.py run --output results/e1_countertarget_new
```

Do not edit inference source after acquisition starts. Source and protocol
archives are frozen, and review/input hashes are checked before execution.
Interrupted runs require inspection, not silent overwrite or restart.
Results and instructions are local deliverables; no automatic push or PR.
