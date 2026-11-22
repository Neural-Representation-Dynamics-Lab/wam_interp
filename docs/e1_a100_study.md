# Running the paired LIBERO E1 study

Run from the repository root on Linux with a 40 GB A100, the Optional IDM checkpoint, its matching statistics, official LIBERO initial states and MuJoCo 3.3.2. The [model handoff](e1_multimodel.md) gives the backend installation and asset paths. Use the OpenWAM environment for OpenWAM and the transformers 4 environments for the other backends.

```bash
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
export LIBERO_CONFIG_PATH="$PWD/results/libero_config"
export DIFFSYNTH_DOWNLOAD_SOURCE=huggingface DIFFSYNTH_MODEL_BASE_PATH="$PWD/checkpoints"
python -u scripts/e1_native_replan.py
python scripts/analyze_native_replan.py results/e1_native_replan
python -u scripts/e1_sliding_calibration.py
python -u scripts/e1_sliding_targets.py
python -u scripts/e1_replan_alignment.py
python -u scripts/e1_large_swap_replan.py
python -u scripts/e1_motion_control.py
python scripts/report_e1_a100_study.py results results/e1_summary.json
python scripts/e1_blind_review.py results results/e1_review
```

The native run resolves the official book and bowl instructions by name. It draws a second location from the same task's released initial states and keeps the recipient's object orientation, other bodies, robot and instruction. Each of four states enters four episodes with the same action seeds, one for each layout and Optional IDM mode. The shared evaluator uses 30 settling steps, an action horizon of 32 and replans after 10 actions up to 700 controls. Its untouched task predicate scores each episode. The per-episode task/state in the result and the study protocol identify the actual cases. The generic evaluator config stored with each case also contains unused default suite-selection fields.

The first target contact is sampled at control boundaries. The signed contact ratio projects the change in gripper position at first contact onto the target's translation and divides by the translation length. A ratio of one means the two shifts agree along that axis. Finger posture, first-contact timing and success also matter. The development rule requires paired success and a ratio of at least 0.5 in three of four IDM states. Initial bowl outcomes were visible when this rule was set.

The first visual manipulation compares physical slides of the target and a distractor caused by an initial planar velocity after a common observed frame. The simulator-only calibration selects speeds by geometry before intervention outcomes. The separate prescribed-render test displays a smooth three-centimetre translation over 24 controls on one shared native robot trajectory. It varies the displayed geometry while leaving the underlying robot and object dynamics on their native replay. It audits visibility, sampled geometry clearance, changed pixels, native camera restoration and unedited video encoding. This tests visual sensitivity to externally moved objects. Both edits keep the task instruction fixed. Comparisons across objects retain differences in their appearance and affordances.

Each forecast is compared to its own executed controls at offsets 4 and 8 when available. The official VAE decodes the saved latent. OpenCV tracks local features near the simulator-projected end effector and object center in the external camera. Forward/backward consistency and agreement with the measured execution filter correspondences. Results include the stationary and preceding-motion baselines, coverage and missing endpoints. Independent landmark review remains necessary for a confirmatory tracking result. Install `h5py` if the robosuite camera utilities import requires it.

To check a new backend, take its native CLI command from `scripts/e1_model_matrix.py baseline` and append the following options. Run one backend per GPU. Each report lives under the backend's output directory.

```bash
--e1-feature-audit results/e1_model_audits/BACKEND/parity.json \
--e1-scene-protocol results/e1_native_replan/protocol.json
```

The first three native decisions compare native, capture-only and identity actions from identical state and noise. An additional zeroing intervention measures sensitivity at the selected site. Only the identity arm is executed. The subsequent scene probe collects future slots from a physically shifted target or distractor and puts those slots into the original recipient, preserving recipient context and action noise. It records the first ten executed controls in world coordinates. Select the same native condition as the baseline, use exact eager inference, and preserve backend-specific warm-up, history updates and action horizons. The scene probe measures the consequences of visual scene differences encoded at a future-facing site. It cannot distinguish every contribution of present versus predicted content to that site.

The default experiment outputs are local. To resume the native experiment, use the same protocol and source file. The runner verifies the protocol and skips episodes with a complete result. Save prior output directories before rerunning a changed experiment.


The separate large-swap diagnostic reruns the four previously studied soup/tomato layouts with their frozen offsets. It uses the official LIBERO-Object soup instruction and task predicate, four states, original versus exchanged XY slots, and same-weight first-frame versus IDM with full ten-step replanning. Each of the sixteen episodes uses seed zero and the native 400-control limit. It records first object contact, first soup contact, task success, inference latency, peak allocated memory and world trajectories. These are previously studied stress layouts.

For the held-out book analysis, [the recorded analysis specification](../configs/experiments/e1_slide_analysis.json) defines the state-level unit, two-seed requirement and five-millimetre nuisance margin. The two modes' native timing includes instrumentation. In particular, the native IDM timing includes the forecast copy and save at each replan.

Give an independent reviewer only the generated `blind_review.zip`. It contains paired edit animations, shuffled forecast sheets, instructions and two CSV forms. Keep `review_answer_key.json` until their annotations are returned. The edit audit samples seed zero from each available state. The forecast sheets cover the first, fourth and ninth decisions of each episode when available. Each review item is blind to policy endpoint and task success.

The secondary robot-motion control compares two real trajectories of the arm while the book and mug stay in place. Four book development states at seed zero use a fixed list of action biases. The first bias that shifts the gripper two to four centimetres in each direction while satisfying the contact and object-stability limits supplies the donors. An unedited encoding must pass the same two-centimetre endpoint check. The wrist viewpoint moves with the arm, so this manipulation includes egocentric camera motion.
