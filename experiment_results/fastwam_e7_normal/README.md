# Fast-WAM Optional IDM normal on the E7 episodes (Roadmap E7)

Where does Fast-WAM Optional IDM fail without any intervention, on the same episodes as the Faster-WAM pilot (`../fasterwam_e7_pilot`)? These failures are the starting set for the OOD rescue, which replaces the imagined future with a correct one.

Model: Fast-WAM Optional IDM (`yuanty/fastwam`, `libero_optional_idm_2cam224.pt`, sha256 in each JSON), task config `libero_optional_idm_2cam224_1e-4`, bf16. Every call uses condition `idm`, so the model generates the future video (10 inference steps, sigma shift 1.0, 9 video frames) and the action expert reads all video tokens. Its training config lists all four LIBERO suites.

Code: `scripts/e7_fastwam_normal.py`, run at `db1b5cd` (branch `feat/e7-future-content`) with the runner uncommitted, as each JSON's `source_state` records; the copy that ran is `source/e7_fastwam_normal.py`. The committed `b0ba9de` adds two things that do not change the episodes or the inference: it checks `--task-ids` before the model loads, and it saves each rollout's actions and simulator states for later replay.

Protocol as in the pilot: LIBERO-Plus environment, 30 warm-up steps, action horizon 32, replan every 10 steps, 256 px render, 700 max policy steps, seed 42. ID is the 10 original libero_10 tasks with initial states 0 to 4 (n = 50). OOD is the pilot's 140 LIBERO-Plus variants (7 factors, levels 2 and 4, 10 per cell, sample seed 0), one trial each, run in two halves of 70 (two report files in `ood/normal`).

`analyze_normal.py` checks that every run completed and that every episode has the pilot's variant, episode seed and initial state (instruction, factor and level too for OOD), then pairs the two models.

## Success

Success with 95% Wilson interval. Paired against the pilot's Faster-WAM `normal`: trials only Fast-WAM solved / trials only Faster-WAM solved, exact McNemar p.

| Arm | Fast-WAM | Faster-WAM (pilot) | Both / neither | Only Fast / only Faster | p |
|---|---|---|---|---|---|
| ID | 50/50 [0.93, 1.00] | 50/50 [0.93, 1.00] | 50 / 0 | 0 / 0 | 1 |
| OOD | 108/140 [0.70, 0.83] | 103/140 [0.66, 0.80] | 88 / 17 | 20 / 15 | 0.50 |

OOD by factor, n = 20 each (Fast-WAM at level 2 and level 4 in brackets):

| Factor | Fast-WAM | Faster-WAM | Only Fast / only Faster | Neither | p |
|---|---|---|---|---|---|
| Background Textures | 9 (5, 4) | 6 | 4 / 1 | 10 | 0.38 |
| Camera Viewpoints | 13 (7, 6) | 17 | 2 / 6 | 1 | 0.29 |
| Language Instructions | 19 (10, 9) | 19 | 1 / 1 | 0 | 1 |
| Light Conditions | 15 (8, 7) | 20 | 0 / 5 | 0 | 0.06 |
| Objects Layout | 19 (9, 10) | 19 | 1 / 1 | 0 | 1 |
| Robot Initial States | 18 (10, 8) | 13 | 5 / 0 | 2 | 0.06 |
| Sensor Noise | 15 (8, 7) | 9 | 7 / 1 | 4 | 0.07 |

Per-factor p values are unadjusted and none is below 0.05.

## Failures

`failures.json` lists every OOD failure with its factor, level, base task, episode seed and whether Faster-WAM also failed.

- 32 OOD failures, every one ran to 700 policy steps. Faster-WAM also fails on 17 of them (Background Textures 10, Sensor Noise 4, Robot Initial States 2, Camera Viewpoints 1).
- By factor: Background Textures 11, Camera Viewpoints 7, Light Conditions 5, Sensor Noise 5, Robot Initial States 2, Objects Layout 1, Language Instructions 1.
- No ID failure, so this run gives no ID counterpart for the rescue.

## Caveats

- One trial per variant, so these runs do not separate systematic failures from single stochastic outcomes.
- Fast-WAM generates its future video, while the pilot's Faster-WAM builds the future from pure noise in one pass. The comparison is between two models, not two settings of one.

## Files

- `analyze_normal.py` (standard library only) recomputes `summary.json` and `failures.json` from `id/normal`, `ood/normal` and the pilot's `normal` runs.
- `check_replay.py` replays rollouts saved by the committed runner and compares every simulator state with the recording. The seed 42 run here predates the recording, so it has no rollouts to replay; the script is included for the runs that do.
