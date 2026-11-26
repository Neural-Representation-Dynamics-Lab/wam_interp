# Later-state E1 feasibility pilot

This separate exploratory round replays the first 32 native commands from the
existing state-3/4/5 contact episodes, then acquires a new cream/butter donor bank.
It uses three previously studied development states, not a held-out sample.
The saved command prefixes and every camera/proprioception observation are
hash-identified and checked exactly during replay, including intermediate steps.
Simulator and controller are recreated on each restore before warm-up and replay.
Merely setting a later MuJoCo qpos is not an adequate restoration.

The frozen protocol is `configs/experiments/e1_later_state.json`: four prompt
recipes, four donor seeds (18-21), three states, 48 native futures total. Recipient
action seeds are 0 and 1. The original soup/tomato instruction remains unchanged.
Use the existing 20% norm gate and first-eight-pairs review limit per state, then
the existing all-nine-frame semantic, motion and appearance checks. No latents
are scaled or interpolated. No new policy comparison is permitted for a failed
pair. Prompt text alone never establishes a semantic label. AI-assisted review
is authorized for exploration; independent human validation remains pending.

The primary effect, if a pair qualifies, projects the butter-minus-cream executed
EEF endpoint difference onto the initial world-XY cream-to-butter direction at
the replayed observation. Average seeds within each state first. No new effect
estimate exists if no pair qualifies. This bank must not be expanded or its
thresholds relaxed in response to poor selection yield.

On the prepared Linux environment with EGL and the existing LIBERO configuration:

```bash
python scripts/e1_countertarget.py prepare --protocol configs/experiments/e1_later_state.json --output results/e1_later_state_20261120
# Review the fixed shortlists and complete state_*/review.json truthfully.
python scripts/e1_countertarget.py select --output results/e1_later_state_20261120
python scripts/e1_countertarget.py run --output results/e1_later_state_20261120
```

Keep the replay input NPZ files and manifest with the source/protocol archive.
They are experiment data, not Git source. Do not edit inference source between
acquisition and execution. A current source overlay is necessary because prior
archives intentionally preserve the code used for the prior experiments.
