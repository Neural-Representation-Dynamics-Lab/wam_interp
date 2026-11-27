# E1 obstacle interventions on LIBERO

Read the [results, sign audit, and interpretation across E1 properties](e1_obstacles_result.md).

Does a visible obstacle in a model's future change its initial reach toward the open side? This is a first-chunk LIBERO experiment. A positive signed lateral displacement means the gripper moved away from the obstructed side. The primary comparison changes the future while the current scene, instruction, action seed and robot state are fixed.

## Conditions and measurement

| Live scene | Future after frame zero | Paired comparison |
|---|---|---|
| Clear | Clear | Reference for future-only and live-only |
| Clear | On-route bar | Effect of an obstacle in the future |
| On-route bar | Clear | Effect of seeing the obstacle now |
| On-route bar | Same on-route bar | Additional effect of the future when the bar is already visible |
| Clear | Off-route bar | Sensitivity to a visible obstacle outside the swept path |

Every case includes mirrored sides, native action replay and a clear-video sham. The obstacle is a synthetic, fixed blue crossbar with visual and collision geometry. Scene objects settle from the released initial state before enabling barrier collision. A wrist view can occlude a bar even when it is visible in the external camera, so visibility is measured after the two camera transformations. Each live recipient supplies frame zero and its own history.

Use the model's recorded 32-control unobstructed path to place the bar. Re-executing that path must hit the on-route bar after the control-16 readout and miss the off-route placement throughout. The depicted clip takes native frames every four controls up to the last such sample before first contact, with at least 24 controls retained. It repeats that final view through the remaining samples. Joint positions and gripper positions must agree within 1e-8 across matching clips. Each sample needs at least 100 changed pixels. Select the first off-route pair with no contact and mean changed area between one quarter and four times the corresponding on-route edit. Position candidates are ordered in the off-route script. Record every attempt. Eligible images can still differ in local contrast or occlusion.

Measure displacement in simulator world coordinates at control 16 for the two IDM backends. Control 8 is available to all action horizons, including the 12 executable controls in LingBot's first chunk. Record forward displacement, minimum gripper-point clearance to the oriented box, actual collision steps, commands and full trajectories. Minimum point clearance is distinct from geometry contact. Treat contact after an intervention as a behavioral outcome. Use one paired value per start when aggregating the mirrored sides. Task completion is outside the single-chunk readout.

## Model interfaces

| Backend | Pilot intervention |
|---|---|
| FastWAM Optional IDM | Encode nine camera montages with the native VAE and replace clean future latents. Preserve latent frame zero. |
| Long-WAM IDM | Render its own reference. Preserve clean history, encode the future and apply matched noise at the native imagination timestep. |
| Faster-WAM one-pass | The released path starts at sigma 1. Schedule-matched noising removes the clean RGB signal exactly. A direct clean substitution would test a different noise regime. The scheduler regression test records this feasibility limit. The checkpoint audit measures whether real edited and unedited encodings give identical future K/V and actions at the one-pass boundary. It also measures live-scene steering. |
| LingBot-VA | Encode 13 native camera observations with a separate streaming VAE cache. During a donor pass, supply the three future latent slots at the terminal predicted-video write. Replay the selected layer's future K/V in an ordinary recipient pass. Both CFG rows keep their original observed slot. |
| OpenWAM | Encode the native nine-frame composite. At the last joint denoising evaluation, mix the clean future with the recipient's original Gaussian at the native sigma. Capture the middle layer's future K/V, then replay only those rows on the recipient's action-facing attention path. |

Check `supports_prepared_frames` for the selected intervention site. The public `FutureAdapter.from_prepared_frames(request, image_tensors)` accepts camera frames transformed by the loaded backend. Pass `noise_seed` for Long-WAM. OpenWAM also needs a noninitial `stage` on the adapter. Faster-WAM reports its native one-pass limitation explicitly. Use `predict_with_future(request, donor)` for recipient replay.

Use fixed-seed native capture and identity replay before interpreting an edited branch. In development the sham endpoint must be within 20 mm of native. The shared-fixture cache pilot tests whether the model can use a rendered donor. Signed K/V differences alone do not establish that the policy represents left versus right. For a side decoder, fit a mean normalized difference on independent training starts, then predict the orientation of unseen matched pairs. A successful transfer to a new scene requires its own test. Report usable and attempted starts and the independent unit for every interval.

## Reproduce on Linux

### FastWAM smoke and held-out study

Run these commands from the root of a checkout named `wam_interp` on a BF16-capable GPU. On a fresh Linux machine, the setup script installs the Python 3.10 LIBERO environment, fetches the pinned FastWAM Optional IDM checkpoint, its statistics, and the Wan VAE, text encoder and tokenizer. An existing matching environment and asset directory can be reused. [Setup script](../scripts/setup_e1_linux.sh)

```bash
bash scripts/setup_e1_linux.sh             # once on a fresh checkout
source .venv-e1/bin/activate
export PYTHONPATH="$(dirname "$PWD"):${PYTHONPATH:-}"
export LIBERO_CONFIG_PATH="$PWD/results/libero_config"
export DIFFSYNTH_MODEL_BASE_PATH="$PWD/checkpoints"
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl

bash scripts/run_e1_obstacle_fastwam.sh smoke results/obstacle_reproduction
bash scripts/run_e1_obstacle_fastwam.sh heldout results/obstacle_reproduction
```

The smoke tests one development start (task 0, state 0, seed 0) through source rendering, off-route placement, calibration and paired behavior. It exits with an error if the case fails. The held-out command verifies the original design hashes, runs development and the 11 registered task-0 starts with seed 1, and writes `results/obstacle_reproduction/heldout.json`. It fits the side decoder on development states 0 and 1 only. Each run can resume from its own output directory. Use a new directory if you change the checkpoint, settings or code. [Frozen design](../configs/experiments/e1_obstacle_fastwam_task0.json) · [Prespecified 3D analysis](../configs/experiments/e1_obstacle_unsigned_supplement.json)

**Reference result.** On an A100 with BF16, all 11 registered starts passed the replay and geometry checks. The decoder identified the obstacle side on 11/11 starts. At control 16, the mean shift was 0.75 mm toward the future-only barrier (95% bootstrap interval 0.47–1.02 mm). Mean 3D displacement was 3.17 mm for on-route edits and 3.03 mm for off-route edits. The off-route 90% interval of 2.78–3.30 mm exceeded the registered 2 mm margin for total displacement. These are first-chunk measurements on one task and one held-out policy seed.

### Other backends and individual stages

The individual stage commands below also run Long-WAM with its own reference motion. Use the matching checkpoint and statistics for each backend. OpenWAM uses its own environment. [Confirmation setup](e1_confirmation.md)

```bash
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
export PYTHONPATH="$(dirname "$PWD"):${PYTHONPATH:-}"
spec=configs/experiments/e1_obstacle_dev.json
out=results/obstacle_pilot
for model in fastwam longwam; do
  python scripts/e1_confirm.py references --backend "$model" --manifest "$spec" --out "$out/refs"
  python scripts/e1_obstacle.py sources --backend "$model" --manifest "$spec" --refs "$out/refs" --out "$out/base"
  python scripts/e1_obstacle_offroute.py --backend "$model" --manifest "$spec" --source "$out/base" --out "$out/qualified"
  python scripts/e1_obstacle_pilot.py --backend "$model" --manifest "$spec" --out "$out/qualified"
done
python scripts/e1_obstacle_report.py "$out/qualified" --out "$out/development.json"
```

For each cache backend, activate its environment and select a source directory whose `sources.json` passed. Replace the path below with that directory. To make an own-plan source, first run `e1_obstacle_cache_reference.py --backend MODEL --manifest SPEC --out REFS`, then run the same sources and off-route stages with that backend name. LingBot consumes every executed observation across native replans when generating a reference.

```bash
python scripts/e1_obstacle_cache.py --backend lingbot_va --source "$out/qualified/fastwam/task_0/state_0/seed_0" --out "$out/cache_shared"
python scripts/e1_obstacle_cache.py --backend openwam --source "$out/qualified/fastwam/task_0/state_0/seed_0" --out "$out/cache_shared"
python scripts/e1_obstacle_fasterwam.py --source "$out/qualified/fastwam/task_0/state_0/seed_0" --out "$out/fasterwam_real"
python -m pytest -q tests/test_e1_obstacles.py tests/test_e1_future_adapter.py tests/test_e1_confirmation.py
```

The development manifest contains six starts on two tasks and one action seed. For a second-room development check, `e1_obstacle.py sources --station 32` places the bar at the final native reach. Keep that variant separate from the registered station-28 results. For a new IDM replication, `e1_obstacle_freeze.py` records the independent development checks and proposed fresh-start allocation. Save a new manifest for each new study. The published task-0 replication uses the original frozen design above.
