# Experiment protocol

## Before running

Record repository revision, model family/variant, checkpoint hash, model assets, resolved config, dataset/stats paths, simulator version, task split, observation identity, prompt, seed, inference steps, action horizon, dtype/device, and compile mode.

For each paired comparison, keep observation, prompt, action noise seed, stats, model checkpoint, action horizon, and action scheduler fixed unless that factor is the intervention. Confirm seed controls every relevant random draw in the selected path.

## Conditions

Use explicit names rather than a single “imagination on/off” label:
- first-frame action path;
- denoised future-conditioned path;
- future representation ablated/corrupted at a named boundary;
- matched-scene future/KV swap;
- content-selective edit;
- compute-matched content-destroyed control.

Report which model paths are architecturally different. Do not attribute a difference solely to imagined content when inference depth, number of forward passes, cache reuse, or guidance also changes.

## Outcomes

At minimum retain normalized action chunk, post-processor action chunk, actually executed prefix, task success/action error, and latency. For interventions, include edit definition, tensor/frame boundary, layer/step/token range, and intervention magnitude. Report both model-level action changes and environment consequences; a changed action is not automatically better or worse.

## Evidence discipline

Separate exploratory traces from preregistered/confirmatory runs. Verify image edits after decoding. For activation patches, use matched donor/recipient inputs and report patch direction. Include negative and compute-matched controls. Preserve failed and null interventions. Claims about scene-specific representation, causal action use, OOD benefit, and adaptive selection require separate evidence.

