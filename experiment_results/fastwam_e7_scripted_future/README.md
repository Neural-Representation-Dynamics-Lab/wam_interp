# Fast-WAM scripted future (Roadmap E7)

If the imagined future is replaced by a scripted video of the arm approaching one of two objects, does the executed chunk follow the video, and does it follow it as well out of distribution?

Model: Fast-WAM Optional IDM (`yuanty/fastwam`, `libero_optional_idm_2cam224.pt`), condition `idm`, bf16, not the Faster-WAM model of the mask runs. Its training config lists all four LIBERO suites. Code: `scripts/e7_scripted_future.py` and `configs/experiments/e7_scripted_future.json` on top of `7a678f9`, uncommitted when they ran; the copies that ran are in `source/`. The refactored script was committed later as `db1b5cd` (branch `feat/e7-future-content`). The scripted motion is raxec's `run_reference` from `scripts/e1_simulator_pairs.py`, rendered in the same scene, and the video goes in through `encode_reference_video` and the E1 chunk session (PR #18).

## Setup

- Per start: 30 warm-up steps, then one 32-control chunk from the same observation in three branches: the model's own future (`native`), and the future replaced by a scripted pre-contact approach to 14 cm above the first or the second object of a fixed pair. The scripted clip is nine paired 224 px camera views with frame zero set to the policy's observation; it is encoded with Fast-WAM's own VAE and its future latents replace the model's, with the observed first latent frame kept. The pair puts the task's first object first. Two action seeds per start, averaged.
- Measure: the end-effector shift after the chunk along the tabletop axis from the first object to the second, as a share of their separation (0.13 to 0.30 m). "toward" is the second-object branch minus the first-object branch.
- Starts: ID is 5 states for each of the 10 original libero_10 tasks; OOD is the 140 LIBERO-Plus variants of the E7 pilot.
- Checks every used start passed (the script stops otherwise): replay matches, scripted frame zero equals the policy's observation, the observed latent frame is unchanged, the scripted future goes in as given, and the native branch equals the native inference. Anchor error before restore was 0 in every branch. `checks/` holds the check run on task 0, state 0 (repeated native branch), and `refactor_check/` the same check with the committed `db1b5cd` code; `checks.json` and both seed files are byte-identical.

## Used starts

A start is dropped if a scripted approach misses raxec's E1 limits (goal error 2 cm, object moved 1 mm, contact). Every drop came from the scripted controller, not the policy.

| Arm | Used / starts | Dropped |
|---|---|---|
| ID | 32 / 50 | task 2 (moka pot or frypan) 5, task 3 (bowl) 4, task 8 (moka_pot_2) 5, task 9 3, task 5 (book, contact) 1 |
| OOD | 89 / 140 | task 2 20, task 3 19, task 8 7, task 9 3, task 5 (contact) 2 |

Tasks 2 and 8 have no used start in either arm, and task 3 has one in ID and none in OOD.

## Results

End-effector shift as a share of the object separation, mean with 95% bootstrap interval. Task-matched reweights ID to the OOD task counts and bootstraps within tasks; every OOD task occurs in ID. It was added during the analysis because the dropped starts leave the arms with different task mixes.

| Branch | ID (32) | OOD (89) | OOD minus ID, raw | OOD minus ID, task-matched |
|---|---|---|---|---|
| native future | -0.54 [-0.62, -0.46] | -0.45 [-0.50, -0.40] | +0.09 [-0.01, 0.19] | +0.09 [0.05, 0.13] |
| first-object future | -0.61 [-0.68, -0.54] | -0.58 [-0.62, -0.54] | +0.03 [-0.05, 0.11] | +0.05 [0.02, 0.07] |
| second-object future | +0.16 [0.10, 0.23] | +0.17 [0.13, 0.21] | +0.01 [-0.07, 0.08] | +0.01 [-0.01, 0.03] |
| toward (second minus first) | 0.77 [0.76, 0.79] | 0.75 [0.73, 0.76] | -0.02 [-0.05, 0.00] | -0.04 [-0.05, -0.02] |

In metres, toward is 0.182 ID and 0.174 OOD.

- Both scripted branches end closer to the shown object than to the other one in every used start and seed, ID and OOD.
- The native chunk ends closer to the first object than to the second in 64 of 64 ID and 166 of 178 OOD seed runs.
- The two action seeds differ by at most 0.006 (ID) and 0.032 (OOD) in toward share.

OOD by factor, share of the separation:

| Factor | n | native | first | second | toward |
|---|---|---|---|---|---|
| Background Textures | 12 | -0.33 | -0.54 | +0.21 | 0.75 |
| Camera Viewpoints | 16 | -0.35 | -0.58 | +0.16 | 0.73 |
| Language Instructions | 14 | -0.49 | -0.57 | +0.20 | 0.76 |
| Light Conditions | 14 | -0.48 | -0.60 | +0.17 | 0.77 |
| Objects Layout | 13 | -0.61 | -0.68 | +0.13 | 0.81 |
| Robot Initial States | 7 | -0.42 | -0.52 | +0.22 | 0.74 |
| Sensor Noise | 13 | -0.46 | -0.56 | +0.11 | 0.67 |

## Reading

- The chunk follows the scripted future in both arms: a future showing the other object moves the endpoint 0.75 to 0.77 of the separation toward it, and always closer to it.
- Given the same scripted future, the endpoint differs little between ID and OOD (task-matched +0.01 for the second object, +0.05 for the first, toward -0.04 of the separation).
- The native chunk differs more (+0.09): out of distribution it moves less far toward the first object. Background Textures and Camera Viewpoints show the largest native shortfall (-0.33 and -0.35, against -0.54 ID), while their scripted branches stay near the other factors.

## Caveats

- Fast-WAM, while the masks and injections ran on Faster-WAM.
- One chunk from one point per start, not a closed-loop success rate.
- The scripted video is rendered in the OOD scene, so it carries the scene's visual shift.
- Because the pair puts the task's first object first, the native and first-object branches nearly coincide by design.
- Tasks 2, 3 and 8 are missing.
- The native shift can be short because the arm heads elsewhere or because it moves more slowly, and this measure does not separate the two.

## Files

- `id/rows.json`, `ood/rows.json`: one row per start, with the approach checks, per-seed axis shifts and the reason for any drop.
- `id/<start>/seed_<k>.npz`, `ood/<start>/seed_<k>.npz`: per branch, the end-effector positions over the chunk (33 x 3) and the executed actions (32 x 7). The per-start contact sheets of the scripted approaches (242 JPEGs, 79 MB) are not included.
- `analyze_scripted.py` (standard library only) recomputes `summary.json` from the two `rows.json` files. `summary.json` was computed with Python 3.12; Python 3.10 sums floats differently and changes its values by at most 4e-16.
