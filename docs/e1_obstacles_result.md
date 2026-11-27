# E1 obstacle results

On the registered LIBERO-10 task-0 starts, FastWAM Optional IDM encoded which side of the route held an imagined barrier. The executed gripper shifted 0.75 mm laterally toward the barrier side at control 16, averaged over mirrored placements. A visually matched off-route edit produced a similar total 3D shift. There was no evidence of anticipatory steering toward the open side in this first action chunk.

## What changed

The four conditions had a barrier in neither scene, the future only, the live scene only, or both. A fifth condition put a visible future barrier off the route. The recipient instruction, starting state, and policy seed were paired. Each future began with the recipient’s actual observation. The two-camera footage followed the same reference motion until the last pre-contact sample and held that view through the tail. Continuing the scripted reference verified contact with the on-route solid and no contact with the off-route solid. FastWAM and Long-WAM used their own source motion. LingBot-VA and OpenWAM used FastWAM source footage at one selected cache site each. [Methods and reproduction](e1_obstacles.md)

## FastWAM held-out results

All 11 registered starts on task 0 qualified with action seed 1. The decoder used development states 0 and 1 and identified the barrier side on 11/11 held-out starts. The clear-video endpoint drifts were 10.1–14.8 mm under the 20 mm limit. Positive steering is displacement toward the open side relative to the paired baseline. Each start contributes one mean over both placements. Intervals resample starts.

| Paired contrast at control 16 | Mean signed steering (mm) | 95% interval (mm) |
|---|---:|---:|
| Future-only minus clear | −0.747 | −1.019 to −0.467 |
| Both minus live-only | −0.678 | −0.965 to −0.395 |
| Live-only minus clear | +0.054 | −0.078 to +0.187 |
| Off-route future minus clear | −0.530 | −0.720 to −0.347 |

Mean total 3D displacement was **3.17 mm on-route** and **3.03 mm off-route**. The off-route signed effect fell inside the ±2 mm equivalence margin. Its total 3D displacement exceeded the margin (90% interval 2.78–3.30 mm). The exploratory paired difference in signed steering between on- and off-route edits was −0.22 mm (95% interval −0.44 to +0.01). These readouts measure different components of the gripper’s response. [Numerical summary and per-start figure](https://github.com/Neural-Representation-Dynamics-Lab/wam_interp/pull/21) · [Frozen design](../configs/experiments/e1_obstacle_fastwam_task0.json) · [3D analysis supplement](../configs/experiments/e1_obstacle_unsigned_supplement.json)

### Independent sign check

We took the world-space vector from each tested barrier center to its mirrored center. Projecting the paired control-16 displacement onto this vector gives positive values for steering toward the open side. This calculation uses neither the model scorer nor its side flag. It reproduced all **88 individual values** (11 starts × 2 sides × 4 contrasts), with a maximum difference below 0.000000001 mm. For the future-only condition the descriptive side means were −1.582 mm on the plus side and +0.088 mm on the minus side. The published −0.747 mm describes their average. Shorter distance to a physical solid, choice of route, and collision each require separate measurements.

After running the held-out reproduction command, check the actual trajectories with:

```bash
python scripts/e1_obstacle_sign_audit.py --report results/obstacle_reproduction/heldout.json \
  --qualified results/obstacle_reproduction/held/qualified
```

## Other models and unsuccessful checks

| Model or check | Result | Scope |
|---|---|---|
| Long-WAM IDM | Mean future-only steering about +0.06 mm at control 16 across five eligible starts from tasks 0 and 3. | Only one of two reserved task-0 starts qualified for the independent side check. Development evidence. |
| LingBot-VA | Mean future-only steering −0.011 mm for four task-0 starts and +0.0015 mm for one task-3 start at control 8. | The two reserved calibration examples had readable side at the tested cache layer. |
| OpenWAM | Corresponding means −0.017 mm and −0.024 mm. | The same limited calibration and single-site interpretation applies. |
| Faster-WAM | All seven matched clean-video edits left the tested future K/V and actions identical. Its native live-scene response could change. | The released one-pass schedule starts the future at pure noise. This check establishes the limit of that edit at this boundary. |
| Calibration and exclusions | An early clear-video variant produced about 80 mm of endpoint drift and was discarded. A first-frame mismatch found in development was corrected and affected cases rerun. A second FastWAM task failed to qualify for held-out inference. | Independent human review of the clips remains outstanding. Only FastWAM task 0 reached held-out confirmation. |

## How this fits the other E1 evidence

| Property | Predicted response | Evidence to date |
|---|---|---|
| Depicted arm motion | Shift the executed reach with edited motion. | With their own source videos on three LIBERO tasks, FastWAM shifted +26.15 mm (47/48 starts) and Long-WAM +6.61 mm (48/48). Only 11/48 Long-WAM starts qualified with shared FastWAM footage. [Study design](e1_confirmation.md) |
| Irrelevant appearance | Preserve the trajectory within a predefined bound. | Under the registered room-wall tint, mean trajectory RMS was 1.03 mm for FastWAM and 0.29 mm for Long-WAM on own footage. Both upper bounds were below 5 mm. Stronger tabletop tints in a separate experiment changed reaches. [Visual replication](e1_visual_replication.md) |
| Target region and language | Reach toward the depicted approach. | Native donor swaps changed actions. Donor prompt and scene information can travel together. Scripted RGB removed donor language, while depicted arm motion and target region still varied together. [Simulator-video test](e1_simulator_video.md) |
| Object identity | Reverse the instructed target after exchanging complete visual assets. | The native behavioral positive control passed in 0/2 development layouts. Target-specific future use remains unresolved. [Identity protocol](e1_object_identity.md) |
| Object location | Follow the target more than the distractor with depicted arm motion held fixed. | Seven of ten two-seed held-out starts qualified. Signed shifts were −3.73 mm for the book and −7.20 mm for the mug. In a separate physical relocation, 0/8 arrangement–seed cases met the native first-chunk target-following criterion. [Fixed-motion results](e1_a100_fixed_motion_results.md) · [Physical relocation](e1_physical_relocation.md) |
| Obstacles | Steer toward the open side and preserve the reach for an off-route obstacle. | The mean future-only effect went toward the barrier side. Both placements shifted the 3D endpoint. The native replanning positive control has a separate protocol and result below. |
| Spatial relations, reachability, object state, interaction dynamics | Select an available route, change approach to a reachable goal, skip a completed action, or compensate for a slip, respectively. | The studies above do not independently isolate these properties. A validated live-scene response and matched visual edit would make each test interpretable. |
| Blending and donor searches | Diagnose intervention magnitude, coherence and stimulus availability. | Blends vary several properties. An unsuccessful donor search leaves the semantic hypothesis unresolved. [Equal-norm diagnostic](e1_equal_norm.md) |

## Native avoidance pilot

The [prospective pilot](e1_native_avoidance.md) tests the model’s own plans under repeated observation of a real obstacle. Its conditions, clearance and contact rules, appearance control, and stopping rule were fixed before the new policy runs.


**Neither scene passed the native positive control.** A scripted detour crossed the barrier plane without robot contact on every side and seed (8/8). Replaying the unobstructed commands into the on-route barrier produced contact in 8/8. Six comparisons also passed the off-route replay requirement. With its normal observations and native futures, FastWAM contacted the on-route solid in every comparison, including all six physically qualified comparisons.

| LIBERO task | Seed | Barrier side | Physical checks | First native contact (control) | Clearance gain (mm) | Off-route trajectory RMS (mm) |
|---|---:|---|---|---:|---:|---:|
| 0 | 2 | minus | Pass | 32 | -1.32 | 9.89 |
| 0 | 2 | plus | Pass | 33 | +0.46 | 7.41 |
| 0 | 12 | minus | Pass | 33 | -0.24 | 6.53 |
| 0 | 12 | plus | Pass | 33 | -0.33 | 6.42 |
| 3 | 2 | minus | Pass | 33 | +7.27 | 6.26 |
| 3 | 2 | plus | Off-route replay hit | 31 | -5.58 | 11.76 |
| 3 | 12 | minus | Pass | 30 | +7.45 | 6.88 |
| 3 | 12 | plus | Off-route replay hit | 30 | -0.90 | 11.02 |

Both seeds used the same development start per task. Seeds and mirror conditions are paired repeats. On task 3's minus side the gripper reference point gained more than 7 mm of minimum clearance, yet a robot geometry contacted the bar. All eight native reaches crossed the center plane. All eight off-route RMS shifts exceeded the prospective 5 mm bound. The 48-control horizon measures the initial approach with replanning. It does not establish eventual task success or what these policies would do in a different barrier layout.

The study ran 50 policy or replay episodes on an NVIDIA A100-SXM4 40 GB using BF16, MuJoCo 3.3.2 and robosuite 1.4.1. The pilot protocol was frozen in commit `aecbca0` (design SHA-256 `d0d198ba219bbcb29af00d7bc9958a0edae37e15579d149cf8d6e8e30acee10a`). Independent inspection of the saved commands, trajectories, contact steps and gate decisions confirmed all eight comparisons. The four-arm experiment with future replacement at each replan was conditional on passing this screen and did not advance. This decision applies to these two fixtures.
