# E1 paired motion and background study

This experiment asks whether moving the depicted gripper changes the executed reach and whether changing a visible background colour changes the same reach. The recipient instruction, real scene, starting image, history and action seed are fixed within each comparison. Each model receives the same LIBERO RGB clips. Run this on the released FastWAM Optional IDM and Long-WAM IDM checkpoints. The shared adapter also accepts native future features for Faster-WAM, LingBot-VA and OpenWAM. Those three interfaces do not yet have a validated external RGB intervention.

## LIBERO

Use the repository's LIBERO simulator environment, a CUDA device supporting bf16, and the two released LIBERO bundles. The FastWAM loader reads `configs/experiments/e1_libero.json`. The Long-WAM loader reads `checkpoints/longwam/libero_idm/{config.yaml,model.pt,dataset_stats.json}`. Install the research dependencies plus SciPy, Matplotlib and Pillow. Run from the repository root with the same PYTHONPATH, model asset cache, LIBERO configuration and EGL settings as the native benchmark smoke. The numerical outputs go to a new study directory.

```bash
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
root=results/e1_confirmation
mkdir -p "$root"
python - "$root" <<'PY'
import json, sys
from pathlib import Path
design=json.loads(Path("configs/experiments/e1_confirmation.json").read_text())
dev=dict(tasks=design["tasks"], seeds=design["seeds"], split="development",
         states={str(t):design["development_states"] for t in design["tasks"]})
out=Path(sys.argv[1])/"dev.json"
if out.exists() and json.loads(out.read_text())!=dev: raise RuntimeError("Study directory belongs to a different design")
out.write_text(json.dumps(dev))
PY
python scripts/e1_confirm.py references --backend fastwam --manifest "$root/dev.json" --out "$root/dev"
python scripts/e1_confirm_sources.py --workers 3 --manifest "$root/dev.json" --out "$root/dev"
python scripts/e1_confirm_source_replay.py --workers 3 --manifest "$root/dev.json" --out "$root/dev"
for model in fastwam longwam; do
  python scripts/e1_confirm.py calibrate --backend "$model" --manifest "$root/dev.json" --out "$root/dev"
done
for model in fastwam longwam; do
  python scripts/e1_confirm_parity.py --root "$root" --backend "$model"
done
bash scripts/run_e1_confirmation.sh "$root"
```

The launcher freezes the held-out state IDs and sample-size rationale, generates and verifies the six stimuli, tests paired feature directions for both models, and executes the randomized branches on fresh simulator resets. Identical inputs to FastWAM's pre-intervention video denoiser can reuse its output within one request. Each action branch still runs the action policy. Replay verifies a hash over both native cameras, robot state and instruction at every reset. A soft reset that cannot reproduce the recipient falls back to a fresh native environment.

The motion readout is half the right-left difference in end-effector world x after 32 controls. The colour readout is the maximum of four full three-dimensional trajectory RMS distances. Each compares warm or cool to neutral at fixed motion. A response is practically bounded when its one-sided 95% upper limit is below 5 mm. This is a prespecified engineering scale for a pre-contact reach. It is one quarter of the smallest allowed donor displacement. The intervals describe the tested tasks, checkpoints and starts. Both random seeds must qualify for a start to count. The feature checks and behavior coverage gates apply separately to each checkpoint and property.

Per-camera motion and colour magnitudes are in `held/pixel_contrasts.json`. Read `held.json`, `implementation_hashes.json`, `calibration_summary.json`, `held/*_feature_gate.json` and `{fastwam,longwam}_results.json`. Plots and a blinded stimulus sheet are under `figures/`. Give reviewers the sheet alone. The key contains the arm labels. Raw clips, masks, actions and trajectories stay in the results directory. With a failed feature check, a retained physical readout is descriptive for that property. If both properties fail before action measurement, the receipt records that exclusion.

## Long-WAM reference-source comparison

The additional Long-WAM LIBERO study uses the registered 48 starts, two seeds and the same eight branches. Here the simulator records Long-WAM's native action chunk to make the reference video. The first study used a simulator recording of FastWAM's native action chunk. Long-WAM's recipient instruction, checkpoints, history, and paired action and video-noise seeds are identical across the two studies.

```bash
bash scripts/run_e1_longwam_reference.sh "$root" results/e1_longwam_reference
```

This reads the native branch of completed instruction-matched trials and recomputes any missing native reference. Start selection is independent of the original arm and background outcomes. The branch trajectories, own-source feature probes and sham checks go to the new output directory. Read `longwam_results.json` for the independent study and `paired_source_comparison.json` for differences on starts usable with both sources. The latter also gives the number of starts usable with each source per task. When sharing one A100 between this comparison and RoboTwin, both launchers can use the same `E1_GPU_LOCK` path.

## RoboTwin

Use the dedicated RoboTwin simulator environment and verify the NVIDIA Vulkan renderer. Install the simulator revision from `RoboTwinEvalConfig`, obtain the official objects and embodiment assets, and run `python script/update_embodiment_config_path.py` in that checkout before expert screening. Use Long-WAM's released RoboTwin 2.0 IDM config, model and statistics. The release revisions and checksums are in `src/models/longwam/loader.py`. The 14-D native joint commands, head and two wrist cameras, and checkpoint history all go through the RoboTwin adapter.

```bash
screen=results/e1_confirmation/robotwin
study="$screen/study"
mkdir -p "$screen" "$study"
for task in move_can_pot place_can_basket; do
  python scripts/e1_robotwin_screen.py --root "$ROBOTWIN_ROOT" --task "$task" --start 220100 --attempts 12 --accepted 4 --out "$screen/dev_$task.json"
  python scripts/e1_robotwin_screen.py --root "$ROBOTWIN_ROOT" --task "$task" --start 220200 --attempts 20 --accepted 12 --out "$screen/held_$task.json"
  python scripts/e1_robotwin_clip.py --root "$ROBOTWIN_ROOT" --weights "$ROBOTWIN_IDM" --task "$task" --screen "$screen/dev_$task.json" --out "$screen/smoke_$task"
done
python scripts/e1_robotwin_manifest.py --screen-dir "$screen" --out "$study"
args=(--root "$ROBOTWIN_ROOT" --weights "$ROBOTWIN_IDM" --screen-dir "$screen")
python scripts/e1_confirm_robotwin.py calibrate "${args[@]}" --manifest "$study/dev.json" --out "$study/dev"
```

After expert screening and the native smoke, `bash scripts/run_e1_robotwin_confirmation.sh "$screen" move_can_pot` runs the remaining stages for that task. Repeat for `place_can_basket`. It publishes the number of development starts passing both source and sham gates. Two of the four development starts are needed for stage progression. A successful task rollout, exact capture/replay, and an unedited video with small physical drift answer different questions. The expert labels the active arm. The donor controller solves for a world-x displacement of the active arm while preserving orientation through the articulation's native kinematics. The recipient always receives joint commands. The wall mask comes from the full actor ID in each camera. Every physics substep is checked for robot-prop, wall or table contact. Source checks include both grippers, actual object drift, colour visibility after model preprocessing and elapsed simulation time.

For eligible tasks with sufficient development coverage, run the following stages. Record any omitted task and its failed development gate. Keep the accepted and rejected expert seeds.

```bash
for stage in sources features; do
  python scripts/e1_confirm_robotwin.py "$stage" "${args[@]}" --manifest "$study/held.json" --out "$study/held"
done
python scripts/e1_confirm_analysis.py gate --root "$study" --backend longwam
python scripts/e1_confirm_robotwin.py behavior "${args[@]}" --manifest "$study/held.json" --out "$study/held"
python scripts/e1_confirm_analysis.py report --root "$study" --backend longwam
python scripts/e1_confirm_figures.py --root "$study" --out "$study/figures"
```

LIBERO tests 32 fixed-rate controls. RoboTwin's joint planner can spend different numbers of physics substeps on the same number of commands. Read the timestamps and both-arm measurements before comparing magnitudes between simulators. These interventions measure pre-contact reach behavior. They do not establish target-identity understanding or a benefit to full-task success.
