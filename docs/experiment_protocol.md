# Experiment protocol

These rules apply to inference-path comparisons and closed-loop benchmark runs. Keep the model call, environment control, and scientific claim distinguishable in code and in saved records.

## Before a run

Record enough provenance to reconstruct the run:

- repository revision, model/backend/variant, checkpoint identity or hash, and external pretrained assets;
- resolved model config and dataset-statistics path/hash;
- benchmark, suite, task IDs, trial IDs, and initial-state indices;
- prompts, inference condition, action horizon, video frame count, inference steps, and replan interval;
- environment and inference seeds, device, dtype, and compilation setting;
- preprocessing and action-decoding version, including camera order and gripper mapping;
- the normalized model action, decoded command, and prefix actually executed when retaining action traces.

`LiberoEvalConfig.run_metadata` is caller-supplied provenance. The evaluator does not infer checkpoint hashes, statistics identity, or simulator versions. Do not assume a seed controls every source of randomness; record what the selected model and simulator seed.

## Paired inference

For a paired comparison, hold the observation, prompt, checkpoint, normalization statistics, action noise seed, horizon, and scheduler fixed unless one is the intervention. Record exactly which settings changed.

Use explicit conditions such as `first_frame`, `idm`, a named intervention, or a compute-matched control. FastWAM Optional IDM's `first_frame` and `idm` modes differ in inference path, so a mode comparison alone does not isolate the causal effect of future content. Fixed IDM and Joint checkpoints have their own native paths.

## Closed-loop evaluation

`evaluate_libero(model=adapter, config=...)` consumes an already-loaded evaluation adapter. The caller composes model configuration, loads weights and external assets, reads matching statistics, and constructs the adapter before calling the evaluator. The evaluator then owns LIBERO suite/task selection, simulator construction, fixed initial-state selection, no-op warm-up, replanning, stepping, termination, and success metrics.

The adapter owns model-specific raw observation preparation, normalized model inference, and conversion of normalized actions to environment commands. Preserve the expected camera orientation/order, image size/montage, proprioception statistics, action statistics, gripper convention, and action horizon. A configuration file alone is not evidence that an episode ran.

For LIBERO, results record each initial-state index, trial seed, success flag, termination reason, policy and warm-up steps, inference count, total reward, and inference metadata. Keep the run's checkpoint/statistics/source identities in `run_metadata`.

## Interventions and traces

For each intervention, record its definition, tensor boundary, layer and denoising step, token/frame range, and magnitude. Include matched donor/recipient inputs for activation patches and state the patch direction. Validate image edits after decoding and, when possible, inspect the latent/feature-space effect.

Generic module hooks only see values that pass through selected modules. They do not automatically expose functional attention, post-RoPE cache values, or all denoising states. Name a trace after the value actually captured, and add backend-specific observers when the research question needs a deeper boundary.

## Interpreting results

Keep these claims separate:

1. a future representation contains scene-specific information;
2. the action computation causally uses that information;
3. using it improves task performance or robustness.

A changed action does not by itself establish usefulness. Include negative controls and, when compute differs, a compute-matched control. Keep exploratory traces distinct from confirmatory runs, retain null interventions, and state alternative explanations. See [the research questions](research_questions.md) and [the LIBERO evaluator guide](libero_evaluation.md).
