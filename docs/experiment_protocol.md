# Experiment protocol

These rules are for comparing model inference paths and, later, evaluating those paths in an environment. The current runner only accepts one prepared observation and returns one normalized action chunk. It does not run LIBERO or RoboTwin episodes. Keep model inference, environment control, and the research claim clear in both code and result files.

## Before a run

Save enough information to reproduce the exact call:

- repository revision and model backend/variant;
- checkpoint identity or hash and any pretrained assets it depends on;
- resolved model config and the selected task/model groups;
- dataset and normalization-stat locations;
- benchmark, task split, trial, and observation identity when using an environment;
- prompt, inference condition, action horizon, video frame count, and inference steps;
- random seed, device, dtype, and compile setting;
- preprocessing and post-processing versions, including camera order and action/gripper transforms;
- the normalized model action, the action after post-processing, and the action prefix actually executed.

Do not assume that a seed controls every random draw. Check the selected model path and simulator path, and record any source of randomness that remains uncontrolled.

## Paired inference

For a paired comparison, hold the observation, prompt, checkpoint, normalization statistics, action noise seed, action horizon, and scheduler fixed unless one of those is the intervention. Record exactly which settings changed between calls.

Use explicit condition names:

- `first_frame`: action inference conditioned on the initial observation;
- `idm`: FastWAM's future-conditioned IDM path;
- a named intervention, such as a future replacement at a specific boundary;
- a compute-matched control that retains the same processing depth while removing the scene-specific information of interest.

The Optional IDM `first_frame` and `idm` calls use different inference paths. A difference between them is evidence about the paths as a whole; it does not isolate future content from additional computation or architectural changes. Fixed IDM and Joint checkpoints have their own native conditions and should not be treated as if they were Optional IDM.

## Environment evaluation

An evaluator should call the model through `load_model_adapter()` and `predict_action()`. It should own environment creation, task and trial selection, observation conversion, robot-specific action transforms, replanning, stepping, and success measurement. Keep the adapter output so action changes can be separated from effects introduced by post-processing or execution.

For LIBERO and RoboTwin, preserve the exact image montage and camera order, proprioception normalization, action normalization, gripper convention, execution horizon, and reset/task protocol expected by the checkpoint. A benchmark config file is not an environment evaluator: report success only after episodes have actually been run with the intended setup.

At minimum, retain per-trial success, the normalized action chunk, the post-processed command, the executed action prefix, and latency. Save task/trial identifiers and enough simulator and preprocessing information to repeat the run. Keep failed runs and null results.

## Interventions and traces

For every intervention, record its definition, the tensor or frame boundary, layer and denoising step, token or frame range, and intervention magnitude. Include matched donor and recipient inputs for activation patches and state the patch direction. Validate image edits after decoding and, where possible, inspect their latent or feature-space effect.

Generic module hooks only see values that pass through the selected modules. They do not automatically expose FastWAM's functional attention, post-RoPE caches, or all per-step states. Name a trace according to the value actually captured; add model-specific observers when a research question needs a deeper boundary.

## Interpreting results

Separate these claims:

1. the future representation contains scene-specific information;
2. the action computation causally uses that information;
3. using it improves task performance or robustness.

A changed action does not by itself mean the change is useful. Include negative controls and, when compute differs, a compute-matched control. Keep exploratory traces distinct from confirmatory runs, record alternative explanations, and do not discard null interventions. The [research questions](research_questions.md) describe the claims and candidate experiments in more detail.
