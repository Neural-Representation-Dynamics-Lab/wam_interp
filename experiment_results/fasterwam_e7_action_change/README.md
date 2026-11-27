# Faster-WAM action-change control for the partial mask (Roadmap E7)

Masking 90% of the first future frame's keys costs a relative 0.34 of the ID successes and 0.71 of the OOD ones (`../fasterwam_e7_partial`). These runs ask whether that OOD excess is just OOD rollouts tolerating an action error of that size less well: they add the mask's own action change to the normal chunk, without any mask, and compare the drop in both arms.

Checkpoint, protocol, states and seeds as in the pilot (`../fasterwam_e7_pilot`): `hustvl/FasterWAM@6bf9471` (`libero/step_021700.pt`), bf16, LIBERO-Plus environment, ID n = 50 and OOD n = 140, seed 42. Every JSON records the repository at `c1fb12e` (branch `feat/e7-partial-mask`, clean). `record_deltas.py` and `inject_deltas.py` ran from that checkout but were not part of it; the copies here are the ones that ran, with only their docstrings updated. The analysis scripts read this folder layout.

## Step 1, record

`record_deltas.py` runs `normal` and, at every call, also computes the chunk `future1_mask_90` and `current_mask_90` would produce from the same input, storing the change over the 10 executed steps in normalized actions (10 x 7) and the decoded gripper flips (`id/normal_deltas`, `ood/normal_deltas`). Its outcomes equal the pilot `normal` in all 50 ID and 140 OOD episodes. `analyze_deltas.py` summarizes the changes (`deltas_summary.json`).

| `future1_mask_90` change | ID (50 episodes) | OOD (140) | OOD minus ID |
|---|---|---|---|
| mean abs, executed steps | 0.0642 | 0.0647 | +0.0004 [-0.0033, 0.0041] |
| gripper dimension, mean abs | 0.036 | 0.071 | |
| calls with a gripper flip | 7.4% | 11.0% | +3.6 pp [1.6, 5.6] |

The change is largest in translation (x, y, z). `current_mask_90` changes actions by 0.019 ID and 0.021 OOD.

## Step 2, inject

`inject_deltas.py` runs `normal` with no mask and adds one recorded `future1_mask_90` change to the executed steps of each chunk. The bank is the ID recording (1208 calls) or the OOD recording (5545 calls); a same-arm bank includes the recipient's own episode.

- `inject_*`: each call draws one change independently from all calls of the bank.
- `seq_*`: each episode takes one bank episode that the normal run solved and replays its changes in call order, cycling if it runs longer, so consecutive calls keep the recorded carry-over.

Draws depend only on the episode's first request seed and the call index, so reruns are identical.

Success with 95% Wilson interval, relative drop against the pilot `normal` with a bootstrap 95% interval over paired units, and paired normal-only / condition-only trials (`noise_summary.json`, `analyze_noise.py`):

| Condition | ID | Drop | Paired | OOD | Drop | Paired | OOD minus ID drop |
|---|---|---|---|---|---|---|---|
| `inject_idbank` | 39/50 [0.65, 0.87] | 0.22 [0.12, 0.34] | 11 / 0 | 82/140 [0.50, 0.66] | 0.20 [0.10, 0.31] | 29 / 8 | -0.02 [-0.17, 0.13] |
| `inject_oodbank` | 37/50 [0.60, 0.84] | 0.26 [0.14, 0.38] | 13 / 0 | 80/140 [0.49, 0.65] | 0.22 [0.11, 0.33] | 34 / 11 | -0.04 [-0.20, 0.13] |
| `seq_idbank` | 37/50 [0.60, 0.84] | 0.26 [0.14, 0.38] | 13 / 0 | 67/140 [0.40, 0.56] | 0.35 [0.23, 0.46] | 46 / 10 | +0.09 [-0.08, 0.25] |
| `seq_oodbank` | 33/50 [0.52, 0.78] | 0.34 [0.22, 0.48] | 17 / 0 | 60/140 [0.35, 0.51] | 0.42 [0.30, 0.53] | 54 / 11 | +0.08 [-0.10, 0.25] |
| `future1_mask_90` | 33/50 [0.52, 0.78] | 0.34 [0.22, 0.48] | 17 / 0 | 30/140 [0.15, 0.29] | 0.71 [0.62, 0.79] | 74 / 1 | +0.37 [0.21, 0.52] |
| `current_mask_90` | 50/50 [0.93, 1.00] | 0.00 | 0 / 0 | 101/140 [0.64, 0.79] | 0.02 [-0.07, 0.10] | 11 / 9 | +0.02 [-0.07, 0.10] |

The mask rows repeat `../fasterwam_e7_partial` with this script's bootstrap draws, so their intervals can differ from that summary in the second decimal.

## Reading

- An action change of the mask's size and profile, drawn per call, costs about 0.2 to 0.26 in both arms, with no OOD minus ID gap. OOD rollouts do not tolerate this error less, and the OOD bank (more gripper flips) does not cost more than the ID bank.
- Replayed in call order, the change costs 0.26 to 0.34 ID and 0.35 to 0.42 OOD, a gap of +0.08 to +0.09 whose interval includes zero, against +0.37 for the mask.
- So neither the error's size, per-dimension profile, timing within the chunk or gripper flip rate, nor OOD fragility accounts for the mask's OOD drop of 0.71. Keeping the recorded carry-over between calls raises the OOD drop to 0.42, still below it.
- Left open: the injected change does not depend on the scene, while the mask's change does. This design cannot test whether that dependence carries the OOD excess.

## Caveats

- One seed per state.
- The 90% level was picked from an open-loop sweep on ID states (`../fasterwam_e7_partial`).
- `noise_summary.json` was regenerated after the `seq_*` runs finished; an earlier version without them gave intervals that differ in the second decimal for the other rows.
- `deltas_summary.json` was computed with Python 3.10. Python 3.12 sums floats differently and changes its values only beyond the 13th significant digit.

`analyze_deltas.py` and `analyze_noise.py` are standard library only and read the JSONs here, in `../fasterwam_e7_partial` and in `../fasterwam_e7_pilot`.
