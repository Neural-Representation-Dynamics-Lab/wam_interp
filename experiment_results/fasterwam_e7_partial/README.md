# Faster-WAM partial future masks (Roadmap E7)

The pilot (`../fasterwam_e7_pilot`) puts the full first-future-frame mask at the floor in both arms. These runs grade that mask, and the same-size current-frame mask, to compare the ID and OOD drop off the floor.

A partial mask `<mask>_<percent>` sets the action rows of the attention masks at the 8 condition layers (0, 4, ..., 28) to False for that share of one frame's 98 key columns (rounded). The columns come from one fixed random order (subset seed 0), so the subsets are nested across percentages and identical in frames 0 and 1: `current_mask_90` and `future1_mask_90` remove the same positions. `checks.json` lists the columns of every partial mask.

The levels 90 and 95 were picked from an open-loop sweep before the closed-loop runs: `sweep_open_loop.py` measures the executed-step and later-step action change for 5 to 100% of each frame on one state of ID tasks 0, 2, 4, 6 and 8 (`sweep_open_loop.json`), and `sweep_positions.py` drops one position of future frame 1, or keeps only one, on the same states (`sweep_positions.json`).

Checkpoint and protocol as in the pilot: `hustvl/FasterWAM@6bf9471` (`libero/step_021700.pt`), bf16, LIBERO-Plus environment, 30 warm-up steps, replan every 10 steps, action horizon 32, 700 max policy steps, seed 42. Code: `src/interventions/future_conditioning.py` and `scripts/e7_future_ablation.py` at `c1fb12e` (branch `feat/e7-partial-mask`), recorded in each JSON. ID is the 10 original libero_10 tasks with initial states 0 to 4 (n = 50) and OOD the pilot's 140 LIBERO-Plus variants. Level 0 is the pilot `normal` run and level 100 the pilot full mask, so every level pairs with the pilot on states and seeds.

## Success

Success with 95% Wilson interval, and relative drop = (normal successes minus condition successes) / normal successes with a bootstrap 95% interval over paired units. Paired against `normal`: trials only `normal` solved / trials only the condition solved.

| Condition | ID | Drop | Paired | OOD | Drop | Paired |
|---|---|---|---|---|---|---|
| `normal` | 50/50 | | | 103/140 | | |
| `future1_mask_90` | 33/50 [0.52, 0.78] | 0.34 [0.22, 0.48] | 17 / 0 | 30/140 [0.15, 0.29] | 0.71 [0.62, 0.80] | 74 / 1 |
| `future1_mask_95` | 13/50 [0.16, 0.40] | 0.74 [0.62, 0.86] | 37 / 0 | 11/140 [0.04, 0.14] | 0.89 [0.83, 0.95] | 93 / 1 |
| `future1_mask` | 0/50 | 1.00 | 50 / 0 | 0/140 | 1.00 | 103 / 0 |
| `current_mask_90` | 50/50 | 0.00 | 0 / 0 | 101/140 [0.64, 0.79] | 0.02 [-0.07, 0.10] | 11 / 9 |
| `current_mask_95` | 49/50 | 0.02 [0.00, 0.06] | 1 / 0 | 96/140 [0.60, 0.76] | 0.07 [-0.01, 0.15] | 13 / 6 |
| `current_mask` | 49/50 | 0.02 [0.00, 0.06] | 1 / 0 | 94/140 [0.59, 0.74] | 0.09 [-0.01, 0.18] | 18 / 9 |

OOD minus ID relative drop for `future1_mask`: 90% +0.37 [0.21, 0.52], 95% +0.15 [0.02, 0.29], 100% 0 (floor).

OOD success per factor, out of 20 (levels 2 and 4 together):

| Factor | `normal` | `future1_mask_90` | `future1_mask_95` | `current_mask_90` | `current_mask_95` |
|---|---|---|---|---|---|
| Background Textures | 6 | 1 | 0 | 4 | 4 |
| Camera Viewpoints | 17 | 3 | 0 | 15 | 13 |
| Language Instructions | 19 | 5 | 4 | 20 | 19 |
| Light Conditions | 20 | 6 | 2 | 20 | 20 |
| Objects Layout | 19 | 9 | 2 | 14 | 15 |
| Robot Initial States | 13 | 4 | 2 | 14 | 15 |
| Sensor Noise | 9 | 2 | 1 | 14 | 10 |

`summary.json` also has the paired contrast `current_mask` success minus `future1_mask` success per level.

## Checks done after the results

Not in `summary.json`; computed from the same JSONs.

- Task mix. OOD variants map to base tasks unevenly (5 to 20 each). Reweighting ID per-task retention under `future1_mask_90` by the OOD task mix gives 0.65 (raw 0.66), against 0.28 OOD (29 of the 103 `normal` successes).
- Factors where OOD `normal` is at the ceiling still show the gap. Light Conditions keeps 20/20 under `current_mask_90` and 6/20 under `future1_mask_90`; Language Instructions keeps 5 of its 19 `normal` successes. ID keeps 33/50. Fisher exact p = 0.008 and 0.006.

## Checks

Before the runs, on the GPU (`checks.json`): the patched model with no condition and `resample` with the original noise reproduce the unpatched actions bitwise, a mask leaves the video cache unchanged, and the partial columns are listed. CPU tests on the branch check that the subsets are nested and match between frames.

## Caveats

- The current-frame control matches the number of keys, not the action change. `../fasterwam_e7_action_change` adds an action change of the mask's size without any mask.
- One seed per state. The 90% level was chosen on 5 ID states.
- The future positions start as pure noise, so the future K/V is computed only from the current frame, instruction, proprio and noise, as in the pilot.

`python analyze_partial.py` (standard library only) recomputes `summary.json` from the JSONs here and in `../fasterwam_e7_pilot`.
