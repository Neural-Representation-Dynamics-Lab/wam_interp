# Research questions: is imagined future a faithful action trace?

## Framing

The project studies a causal claim: whether action generation depends on imagined future content, which parts of that content are used, and under what conditions. Calling generated video “chain of thought” is a useful analogy to test, not an established equivalence. A WAM may benefit from future-conditioned representations without using a human-readable video as a faithful rationale.

## Questions and experiments

### 1. Is action causally dependent on imagined content?

Compare a model’s future-conditioned path with an appropriate first-frame/action-only path. Use the same checkpoint where available, observation, language, action noise seed, stats, and action schedule. A mode comparison tests the whole pathway; it does not isolate content from extra compute or architecture unless matched controls are added.

Candidate controls:
- future processing with future tokens replaced by noise or a fixed learned baseline;
- futures/KV from a matched but different scene;
- resampled or shuffled future representations;
- equal-compute control that preserves processing depth while removing scene-specific information.

These conditions answer different questions. A one-pass transform of noise is not automatically a “filler token” control: it may produce learned, input-conditioned hidden states.

### 2. Where and when does action read the future?

Record action-to-video attention or K/V interactions by layer and action denoising step. Test necessity with layer/step-specific ablation and sufficiency with activation patching from matched clean runs. Track action outputs after every denoising update, not only final actions.

### 3. Which content is used?

Make selective, validated edits to the imagined future (e.g. object displacement, contact timing, gripper pose) while holding unrelated scene content fixed. Verify the edit in decoded frames and, where possible, in latent/feature space. Test whether actions follow the edit. Avoid interpreting image artifacts as semantics.

### 4. Is the effect useful for robustness?

Evaluate matched in-distribution and distribution-shifted tasks. Separate model action error/success, intervention-following, OOD robustness, and latency. Adaptive imagination is a later objective: first establish a mechanism and a validated signal for when imagination helps.

## Generalization strategy

Start with a model-neutral protocol, then use FastWAM Optional IDM as one tractable implementation with idm and first_frame modes. Add at least one architecturally distinct WAM before making general claims. FastWAM/Faster-WAM comparisons are useful case studies but share a narrow design lineage; two variants alone cannot establish what generic WAMs do.

## Hypothesis ledger

Record for each claim: hypothesis, model/checkpoint, condition, control, outcome, predicted result, observed result, and alternatives. Keep “future representation contains scene-specific information” separate from “action causally uses that information” and from “using it improves OOD success.”

