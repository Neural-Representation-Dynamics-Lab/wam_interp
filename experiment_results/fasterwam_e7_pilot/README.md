# Faster-WAM future-conditioning ablation pilot (Roadmap E7)

Faster-WAM's action expert reads a K/V cache built in one DiT pass over [current frame latent, pure-noise future latents], held at the 8 condition layers (0, 4, ..., 28). The cache has 3 latent frames of 98 tokens: the current frame (0-97) and two future frames (98-195, 196-293). Each mask condition sets the action rows of those layers' attention masks to False for one span of key columns, so the action tokens give those keys no weight. `resample` keeps the current frame and swaps in the future K/V built from the same input with future noise seed + 10000.

| Condition | Keys removed |
|---|---|
| `normal` | none |
| `current_mask` | current frame (0-97) |
| `future1_mask` | first future frame (98-195) |
| `future2_mask` | second future frame (196-293) |
| `future_mask` | both future frames (98-293) |
| `resample` | none; future K/V from new future noise |

`current_mask`, `future1_mask` and `future2_mask` remove the same number of keys.

Released checkpoint `hustvl/FasterWAM@6bf9471` (`libero/step_021700.pt`, task config `libero_fasterwam_2cam224_1e-4`). The model card doesn't list its training data; the upstream LIBERO data config (`configs/data/libero_2cam.yaml` at `8366781`) lists the spatial, object, goal and 10 suites (no-noops demos), and the shipped `dataset_stats.json` counts 1712 episodes. Run with `scripts/e7_future_ablation.py` at repo revision `0d01f53` (branch `feat/e7-future-ablation`, clean checkout, recorded per file as `model.source_state`), in the LIBERO-Plus environment: bf16, MuJoCo 3.3.2, robosuite 1.4.1, torch 2.7.1, one H100 80GB, 256px render, 30 warm-up steps, replan every 10 steps, action horizon 32, 10 inference steps, 9 video frames, 700 max policy steps, seed 42.

- ID: the 10 original libero_10 tasks, built from the LIBERO-Plus checkout's BDDL and init files (byte-identical to the original LIBERO ones), initial states 0-4 (n = 50).
- OOD: LIBERO-Plus libero_10 variants, 10 per factor and level for the 7 factors at levels 2 and 4, drawn with sample seed 0 (n = 140, one trial each).

Every condition runs the same variants, initial states and seeds, so trials pair up across folders.

## Success

Success with 95% Wilson interval. Paired by trial against `normal`: trials only `normal` solved / trials only the condition solved, exact McNemar p.

| Condition | ID | Paired | OOD | Paired |
|---|---|---|---|---|
| `normal` | 50/50 [0.93, 1.00] | | 103/140 [0.66, 0.80] | |
| `resample` | 50/50 [0.93, 1.00] | 0 / 0 | 99/140 [0.63, 0.78] | 11 / 7, p = 0.48 |
| `current_mask` | 49/50 [0.90, 1.00] | 1 / 0, p = 1 | 94/140 [0.59, 0.74] | 18 / 9, p = 0.12 |
| `future2_mask` | 50/50 [0.93, 1.00] | 0 / 0 | 103/140 [0.66, 0.80] | 10 / 10, p = 1 |
| `future1_mask` | 0/50 [0.00, 0.07] | 50 / 0, p = 1.8e-15 | 0/140 [0.00, 0.03] | 103 / 0, p = 2.0e-31 |
| `future_mask` | 0/50 [0.00, 0.07] | 50 / 0, p = 1.8e-15 | 0/140 [0.00, 0.03] | 103 / 0, p = 2.0e-31 |

Every failure under `future1_mask` and `future_mask` ran to 700 steps. `summary.json` has success per factor and level (`ood_cells`).

The pre-specified contrast, future-mask effect minus size-matched current-mask effect, is the mean over trials of (`current_mask` success minus `future1_mask` success): 0.98 [0.94, 1.00] ID and 0.67 [0.59, 0.75] OOD, OOD minus ID -0.31 [-0.39, -0.22] (bootstrap over trials, 10,000 resamples). Both arms are at the floor under `future1_mask`, so this difference only reflects the lower `normal` and `current_mask` success OOD. Relative to `normal`, the drop is 100% in both arms.

## Action change

During the `normal` runs, every model call was repeated under each other condition and the mean absolute difference from the `normal` chunk was recorded, over the 10 executed steps and over the whole 32-step chunk (`future_ablation.action_abs_diff` in each call's metadata). "Later steps" are steps 11-32, which are never executed, derived from the two. Values are per-episode means averaged over episodes, with a bootstrap 95% interval over episodes. Re-running `normal` gives 0 on every call.

| Condition | ID executed | OOD executed | ID later steps | OOD later steps |
|---|---|---|---|---|
| `resample` | 0.0058 [0.0055, 0.0061] | 0.0087 [0.0081, 0.0094] | 0.0067 [0.0064, 0.0071] | 0.0120 [0.0108, 0.0132] |
| `current_mask` | 0.0240 [0.0231, 0.0250] | 0.0247 [0.0239, 0.0256] | 0.0118 [0.0113, 0.0122] | 0.0144 [0.0138, 0.0152] |
| `future1_mask` | 0.1018 [0.0971, 0.1066] | 0.0960 [0.0926, 0.0994] | 0.0375 [0.0356, 0.0395] | 0.0387 [0.0374, 0.0401] |
| `future2_mask` | 0.0071 [0.0068, 0.0075] | 0.0089 [0.0082, 0.0096] | 0.0998 [0.0957, 0.1038] | 0.0964 [0.0931, 0.0996] |
| `future_mask` | 0.1316 [0.1256, 0.1379] | 0.1298 [0.1245, 0.1350] | 0.1943 [0.1856, 0.2032] | 0.1954 [0.1877, 0.2028] |

Masking the first future frame mostly changes the executed steps, and masking the second mostly changes the later steps, which replanning discards. Per OOD factor, the `future1_mask` executed change ranges from 0.077 (Background Textures) to 0.106 (Camera Viewpoints).

## Checks

Before the runs, on the GPU (`checks.json`): the patched model with no condition and `resample` with the original noise both reproduce the unpatched actions bitwise, and a mask leaves the video cache unchanged.

## Caveats

- The future positions start as pure noise, so the future K/V is computed only from the current frame, instruction, proprio and noise. These runs can't tell a predicted future apart from a second encoding of the current scene.
- Removing keys also changes how the remaining attention weight is spread. The size-matched `current_mask` and `future2_mask` controls show that removing 98 keys alone does not break the policy.
- OOD `normal` succeeds on 74%, so success-based contrasts between ID and OOD are bounded by it.
- Faster-WAM rollouts aren't bitwise reproducible, so paired trials share seeds and initial states, not exact trajectories. OOD `resample` already separates 18 of 140 pairs.
- Intervals treat trials as independent; trials of the same task are not clustered.

## Files

One JSON per condition and arm, as written by `evaluate_libero_plus()` (OOD) and the script's `evaluate_original()` (ID), with the run settings under `config.run_metadata`. Paths in them follow our server layout. `analyze.py` (standard library only) reads them and writes `summary.json`.

```bash
python scripts/e7_future_ablation.py --arm checks
python scripts/e7_future_ablation.py --arm id --conditions normal,current_mask,future2_mask
python scripts/e7_future_ablation.py --arm id --conditions future_mask,future1_mask,resample
python scripts/e7_future_ablation.py --arm ood --conditions normal,current_mask,future2_mask
python scripts/e7_future_ablation.py --arm ood --conditions future_mask,future1_mask,resample
python experiment_results/fasterwam_e7_pilot/analyze.py
```
