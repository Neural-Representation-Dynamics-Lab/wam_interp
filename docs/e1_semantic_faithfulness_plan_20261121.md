# E1 next experiments: visual content, language, and selective use

**Historical plan:** the execution status below records the 21 November codec
pilot and predates the later simulator, fixed-motion and relocation runs. Use
the [22 November plan](e1_next_steps_20261122.md) for current results and TODOs.
The earlier plan is retained so its original hypotheses and limits remain visible.

Research review and proposed plan, 21 November 2026. This is a development plan, not a preregistration or an established novelty claim.

Execution update (21 November): Phase 1 completed as a bounded 42-branch A100 feasibility pilot. All six baseline codec gates and exact identity/historical replay checks passed. The equal-state mean raw versus reconstructed donor contrast was -2.601 versus -2.730 cm in signed world-y, and 13.799 versus 13.643 cm in 3D endpoint separation. The effect survived the tested floating-RGB bottleneck; this does not erase donor-language information or isolate task-relevant semantics. See [the run report](../results/e1_video_bottleneck_20261121/README.md) and [reusable codec workflow](e1_video_bottleneck.md). The run used prospectively fixed engineering stop limits (1 cm endpoint, 5 mm mean path), not separately calibrated statistical-equivalence margins. Human review remains pending. The simulator-pair builder and Phases 2–4 have not been implemented or executed in this pilot; the remaining sections retain the proposed design.

## Decision

Stop using larger counter-instruction donor searches as the main route to cleaner E1 evidence. Keep the completed whole-future swaps as sensitivity baselines. Run a visual-bottleneck feasibility test first, then controlled simulator-video interventions with task-relevant and task-irrelevant factors. Separate this from subsequent language-path and denoising-stage experiments.

The central question is: which properties of a possible future selectively influence actions, through which representation boundary, and does that influence survive passage through the visible video?

## What the literature already establishes

| Source | Relevant evidence | Consequence for our study |
|---|---|---|
| [Bryce Grant, What world action models do with their imagined futures](https://bryceag11.github.io/blog/wam-imagination/) | Reports masking and instruction-generated future K/V swaps, including changes in first object reached. Discusses distribution shift from masking and controls for current scene and robot motion. | Do not sell future dependence, object switching, or simple future probes as the novelty. Distinguish K/V swaps from our earlier clean-latent swap. |
| [Fast-WAM](https://arxiv.org/html/2603.16666v2) | Compares training with video supervision against explicit inference-time imagination; IDM generates video latents before actions. | Keep Optional IDM as our controlled instrument, but do not equate its IDM path with ordinary current-only Fast-WAM. |
| [Zhou, Ni et al., What Makes World Action Models Generalize? / Simple-WAM](https://arxiv.org/html/2609.34981v2) | A fixed-checkpoint sweep freezes video features after different numbers of passes; much of the benefit appears on the first pass over noisy future slots. The proposed method also changes the training noise schedule. | A readable decoded future is not obviously necessary. Distinguish the same-weight sweep from comparisons of separately trained systems. |
| [Faster-WAM, Zhao et al.](https://arxiv.org/html/2608.04404v1) | Computes reusable future context with one pass and sparse interaction. Appendix compares one to ten video steps with the same checkpoint and fixed action-step count. | Later replication should inspect its future K/V pathway, not assume it exposes the same clean-video boundary as Optional IDM. This is the hustvl architecture, not the other similarly named Faster-WAM paper. |
| [Making Foresight Actionable / AGRA](https://arxiv.org/html/2606.12217v1) | Diagnoses background versus interaction-region action sensitivity; evaluates zero, mean, shuffle, and token-swap interventions and a sensitivity ratio. | Task-relevant versus background intervention is also existing work. Our proposed contribution needs physically controlled edits, visual-bottleneck tests, multiple properties, and explicit language-path separation. |
| [Steering Robustness into WAMs](https://arxiv.org/html/2607.14943v1) | Studies architecture-dependent feature separability and activation steering for robustness. | A decodable or steerable feature is not itself evidence of faithful natural use; architecture-specific negative findings matter. |
| [What-If World](https://arxiv.org/html/2605.27589v1) | Evaluates paired video interventions separately for adherence, physics, scene preservation, and outcomes. | Adopt separate validity checks rather than treating plausible individual videos as a valid contrast. Its focus is video-model counterfactual behavior, not downstream policy mediation. |
| [Twin Rollouts](https://arxiv.org/html/2608.08982v1) | Proposes shared-prefix/shared-noise counterfactual branches and simulator-based locality checks. Its main scaled experiments are forthcoming in this version. | Useful methodological motivation for paired randomness and locality; not empirical confirmation of WAM manipulation behavior. |

This is a focused review of relevant methods/results and the local implementation, not a claim to have exhausted all prior work. A defensible candidate contribution is a property-specific causal audit that distinguishes visual-bottleneck-sufficient effects from richer representation effects and reports both selective invariance and its failures.

## What our hook really changes

Current path in `fastwam_idm.py`:

```
donor observation + donor language + noise -> clean donor latent z_A
recipient observation + recipient language B -> clean recipient latent z_B
replace future slots of z_B with z_A; keep observed anchor
video prefill(z_A, B) -> recipient video K/V
action denoising(recipient K/V, B, fixed action noise) -> actions
```

The hook is after video denoising and before video prefill. Both video prefill and action denoising receive the recipient context. We do not import the donor K/V or donor text embeddings. Nevertheless, the clean donor latent can carry instruction-dependent content, including content not obvious in its decoded video. Mannes's ambiguity remains, but it is not identical to directly transplanting a language-conditioned K/V cache.

The policy normally consumes latent/features, not displayed RGB. Disabling the VAE visualization step cannot answer whether it uses visual semantics. Likewise, visual goals and language are not mutually exclusive explanations: an instruction legitimately specifies the depicted goal. We need operational tests of information pathways, not a claim that a representation is entirely free of language influence.

## Phase 1: does the effect survive a visual bottleneck?

Use saved native donors; no additional semantic donor search. Define `R(z) = E(D(z))`, with the model's own video decoder D and image/video encoder E. E does not take text. Encode the complete clip with the correct temporal codec, then restore the exact recipient observed anchor before insertion.

For each fixed state and action seed, run these seven arms:

1. Native recipient inference.
2. Exact identity insertion.
3. Round-tripped native recipient future.
4. Original raw same-target donor.
5. Original raw alternative-target donor.
6. Round-tripped same-target donor.
7. Round-tripped alternative-target donor.

Initial bounded feasibility: existing development states 3/4/5, action seeds 0/1, 42 short branches. Reuse the earlier original/alternative donors whose behavior effect was measured; do not use the latest unclear cream/butter bank as if it were semantically validated. Preserve the old donor-label uncertainty.

Technical requirements:

- Same checkpoint, bf16, exact observation/proprioception, recipient prompt, noise tensors, sampler settings, simulator/controller restoration, and execution window.
- Use full-precision decoded RGB tensors for the first codec check. The existing convenience decoder converts to uint8; avoid silently introducing JPEG/uint8 artifacts. A later uint8 or mild low-pass bottleneck is a separate, labelled robustness arm.
- Encode all nine frames jointly; never encode frames independently and pretend the temporal VAE is unchanged. Audit clipping, normalization, layout, temporal indexing, dtype and scale.
- Test the native anchor before overwriting it, preserve the actual observed anchor exactly, and inspect whether the reconstructed future still joins it coherently.
- Compare reconstruction of robot/objects and their motion, not only global pixel similarity. Do not interpolate latents to match norms: our earlier interpolation changed visual meaning.
- In addition to the round-trip native control, compare the effect contrast of raw donors against the effect contrast of round-tripped donors. The codec changes both donors; report each arm separately.

Interpretation:

| Outcome | Supported interpretation | Unsupported interpretation |
|---|---|---|
| Identity exact; round-trip baseline stable; donor contrast survives | Information sufficient for this action effect survives the tested RGB bottleneck | All useful information is human-readable, all donor-language information was removed, or the robot needs RGB decoding |
| Round-trip baseline disrupts behavior | Codec intervention is not yet a clean instrument | Nonvisual information was essential |
| Baseline stable; donor effect drops and decoded donor meaning also degrades | Codec may have weakened the intervention | Residual language caused the raw effect |
| Baseline and semantic content stable; effect drops selectively | Motivation to investigate information not preserved by this bottleneck | Conclusive identification of hidden language leakage |

Define practical tolerances on a separate calibration set before testing effects. Use exact repeatability for identity; behavioral equivalence for lossy reconstruction requires a predeclared meaningful margin, not merely a nonsignificant difference. The three-state phase is feasibility only.

## Phase 2: controlled visual futures without a donor prompt

The stronger route bypasses language-driven donor generation. Fork the simulator from an identical state/controller and record physically valid alternative action continuations. Render both cameras, pass only their video through the VAE, then build all policy-conditioning caches under one unchanged recipient instruction. Donor controller commands are never supplied to the recipient policy.

Start with the official instruction to put both soup and tomato in the basket. Soup-first and tomato-first can both satisfy it. This avoids defining fidelity as following a future that contradicts the current task. Choose pre-contact states and explicitly verify that both continuations are reachable. If they require a longer horizon than the released nine-frame/32-control interface, select a suitable earlier decision state; do not quietly change model horizon.

First use one relevant factor and one irrelevant factor:

| Video | First approach | Appearance |
|---|---|---|
| V00 | soup region | original |
| V01 | same exact soup trajectory | changed task-irrelevant background material |
| V10 | tomato region | original |
| V11 | same exact tomato trajectory | same background-material change |

Here first approach is a motion/target-choice composite, not isolated object-identity understanding. The background variants must share exact robot/object states, contacts, camera poses, lighting and physics within each trajectory. Select texture patches outside robot, objects, destination and possible collision paths. Verify changes in both cameras with segmentation/depth masks and pixel differences outside the intended region. Keep masks and rejected candidates.

Important validity issue: a future-only texture change, while the observed anchor stays original, creates an appearance discontinuity. Therefore the initial version is explicitly an appearance-mismatch stress test. It cannot alone establish indiscriminate semantic use. Add matched edit-onset/sham controls and a separate observation/future appearance-consistency factorial (original/original, original/edited, edited/original, edited/edited). The latter changes the observation in some arms, so analyze its interaction separately; do not label all four arms a future-only intervention. Global lighting and camera edits are poor first negative controls because they can change visibility or action coordinates.

Core comparisons are relevant motion differences at each appearance and appearance differences at each fixed trajectory. Use signed motion toward the reference target region, first approach/contact identity including no-contact/unclear outcomes, EEF path, gripper orientation/opening, and any collisions. A large motion response alone is not task success.

Add native, exact identity and matched rendering/encoding sham controls. Match edit-support and calibrated strength where possible, report realized latent and pixel distances, and compare multiple mild texture levels. A norm ratio alone does not remove confounding. No arbitrary scaling that blends different future meanings.

Ground-truth 3D trajectories belong to the rendered reference video. They are not inferred 3D coordinates of a generated future. Validate that VAE reconstruction preserves the visible event before using reference state as its label. Simulator provenance reduces dependence on manual labels, but does not eliminate codec validity checks.

## Phase 3: semantic property map and language interaction

After the first two phases pass feasibility, expand along a declared hierarchy:

| Property | Predicted response if used faithfully | Main confound to address |
|---|---|---|
| Approach direction/target region | Corresponding reach adjustment | Copying gripper motion can explain it |
| Contact onset/object lift | Timing/gripper changes where appropriate | Robot motion and object response co-vary |
| Object motion versus robot motion | Distinguishable effects of each component | Object-only/gripper-only composites can be physically inconsistent |
| Event order, with both orders task-valid | Order-consistent behavior | Altered speed, duration and final arrangement |
| Task-irrelevant appearance/distractor motion | Behavior equivalent within a practical margin | Salience, occlusion, anomaly and collision relevance |

Use constrained simulator continuations for valid positives; treat spatial composites, shuffled time, and zeroed tokens as diagnostic corruptions with explicit validity limits. Do not call target swapping at fixed gripper trajectory a clean semantic intervention if it makes the scene physically impossible. A stronger eventual test changes whether the same property is relevant through the task instruction, then tests the property-by-instruction interaction.

For language, distinguish three contexts: generation language Lg, video-prefill language Lv, and action language La. The existing implementation supplies the same recipient context to the last two; splitting them requires an explicit, tested opt-in API.

First fix a simulator-derived visual future V exactly and cross it with two recipient instructions. This is a 2x2 future-content x recipient-language experiment. It measures agreement/conflict and interactions, not a contest in which always following video counts as success. Then, if needed, fix V and La while changing Lv; separately fix V and Lv while changing La. Hold the proprioceptive context tokens identical while swapping only text tokens. Cache provenance must include source context, layer, time, mask and boundary.

This pathway experiment can show how language modulates visual evidence, but cannot retroactively reveal all instruction information present in a previously generated latent. A probe predicting donor instruction also does not establish a harmful extra language channel: the depicted goal predicts the instruction naturally.

## Phase 4: future representations before readable video

Extend to Faster-WAM only after its checkpoint and intervention boundary pass identity tests. Its native one-pass path is a suitable comparison because it consumes future-aware features without an iterative clean-video rollout. Repeat the property/negative-control logic at its supported feature boundary; do not transplant Optional IDM latents or arbitrary noisy frames as if they were equivalent.

For a denoising-stage sweep, keep action steps and weights fixed, and compare video stages at native timestamps. Optional IDM's action conditioning normally assumes a clean video at time zero. Inserting early noisy latents while labelling them clean introduces an interface shift; failure there is not evidence that readable video is necessary. Use a native joint/noisy-context path when available, or explicitly label the sweep an out-of-distribution stress test. Never decode an internal K/V tensor through a VAE and call it an imagined video.

Reported evidence should separate useful early representations, decoded event fidelity, and action effects. Similar benchmark scores between differently trained architectures do not identify the mechanism of either one.

## Sampling, negative results, and trajectory alignment

- Development: states 3/4/5 remain development data. They cannot become held-out confirmation because the design has repeatedly adapted to them.
- First feasibility: 42 branches for Phase 1. Stop if identity or codec validity fails; repair the instrument rather than expand donor search.
- Follow-up: use a small frozen set of physically valid video pairs and two recipient seeds. Enumerate all exclusions before viewing policy outcomes, including no valid pair.
- Confirmation: budget for at least two suitable tasks and roughly 20 new states per task as an initial allocation, then determine the actual sample requirement using held-out calibration variance and an explicit equivalence margin. Forty states is not a claim of adequate statistical power.
- Average seeds within state. Bootstrap paired differences at the initial-state level, retaining task strata. Multiple chunks from one episode are dependent. Declare one primary readout per property; keep exploratory outcomes labelled.
- For task-irrelevant edits, lack of statistical significance is inconclusive. Report a confidence interval relative to a prespecified practical-equivalence band, along with the contrast against relevant edits. If the interval is too wide, the invariance question remains unanswered.
- Log inference failures, missing semantic contrasts, off-manifold edits, codec distortion, inconclusive labels and meaningful nuisance sensitivity as separate negative findings.
- At each replan k, pair that newly generated forecast with the following execution window. Also retain initial-forecast-versus-entire-episode mismatch as a separately named measure if desired. Never pool the two.
- Object trajectory analysis must condition on visibility and compare against stationary and robot-motion baselines. Accurate gripper prediction does not establish object dynamics understanding.

A later task-success study should use paired full episodes of native IDM, current-frame mode where supported by the same Optional IDM checkpoint, and validated intervention conditions. Both modes need matched task/seed sets. Reference futures sourced from simulator success are oracle inputs; success with them is diagnostic, not deployable performance. Keep one-shot intervention with subsequent native replanning distinct from intervention at every chunk.

## Modular delivery

Retain the existing opt-in future hook, replay and annotation utilities. Add narrowly scoped modules for:

- boundary descriptors and a backend capability check;
- raw tensor video codec round trips with anchor and temporal-layout assertions;
- simulator branching, re-rendering, segmentation and intervention-validity records;
- separate generation/prefill/action text contexts;
- paired effect, equivalence and property-level aggregation;
- chunk-aligned trajectory records and visibility masks.

Each run should export JSON with protocol/version/source hashes, backend/checkpoint/statistics, exact input-state and image hashes, all language contexts, noise provenance, temporal indices, intervention masks, codec method, edit magnitudes, eligibility/rejection reasons, executed trajectories, primary/secondary outcomes, latency and review status. Keep raw arrays beside it. No checkpoints, environments, simulator assets or cache directories belong in the eventual PR.

## Immediate deliverable

Implement only Phase 1 and the simulator-pair builder first. The first run should answer whether our existing whole-future effect survives a controlled visual bottleneck. The next should answer whether task-relevant motion and task-irrelevant appearance have distinguishable downstream effects under a single recipient instruction. Language-path and stage-specific experiments follow only after those interfaces are trustworthy.

Neither selective invariance nor a successful RGB round trip proves complete semantic understanding. Together with positive property-specific responses, valid controls and explicit failures, they would support a much narrower and more defensible account of semantic faithfulness than whole-future swapping alone.
