# E1 literature and implementation review

Reviewed 22 November 2026. This is a research-design audit, not an exhaustive survey or a novelty guarantee. Paper methods and relevant ablations were checked against the vendored inference paths at base commit `b16b680`. OpenWAM was read from its PDF (especially Sections 3 and 4.2); the other sources below have full HTML. Model checkpoint results below are from the papers, not new reproductions by this PR.

## What changes our next experiment

**Showing that a future representation changes actions is already established.** Our contribution should test *selective semantic use*: when a specified property changes, does behavior change in the appropriate way, and does it remain sufficiently stable when a comparable, task-irrelevant property changes? We must first demonstrate that the native policy can respond to the property in the tested setting. The latest physical-relocation pilot did not establish that prerequisite.

The visual forecast, latent representation, K/V read by actions, and physical execution are four different objects. No result at one of these levels automatically establishes faithfulness at the others. Turning off RGB visualization is not a test of whether visual semantics matter: these policies normally act without displaying their forecasts.

## Closest previous experiments

| Primary source and inspected part | Finding relevant to E1 | Consequence for our design |
|---|---|---|
| [Grant, *What world action models do with their imagined futures*](https://bryceag11.github.io/blog/wam-imagination/), masking, swaps, gripper-confound and OOD sections | Reports future masking, counter-instruction K/V transplantation, re-encoding, and first-object changes across models. Also identifies motion/proximity confounds and distribution shift from masking. | Counter-instruction swaps, object switching and re-encoding are not sufficient novelty. Test fixed-motion edits, physical outcomes, relevance interactions and explicit failures. This is an author report/blog, not a peer-review claim. |
| [*Making Foresight Actionable* / AGRA](https://arxiv.org/html/2606.12217v1), Sections 3.1–3.3 | Diagnoses interaction-region versus background sensitivity by manipulating video features; introduces alignment training to improve action grounding. Its policy uses one high-noise video-feature pass. | Relevant-region versus background sensitivity is also prior work. Our stronger candidate is a controlled, property-specific audit with validated edits, natural-language controls and held-out physical arrangements. |
| [*SteerTrace*](https://arxiv.org/html/2611.25719v1), Sections 3.3, 5.2–5.5; released 19 November | Uses targeted observation attacks and clean/attacked K/V patching, followed by neuron and subspace interventions. Shows pathways can shift between future and current-frame features when predicted futures are constrained. | Cross-model future-token causality alone is no longer a defensible novelty claim. Distinguish our physical semantic manipulations from optimized adversarial directions; include current-frame and history controls. Its mediation fractions concern the attack effect, not ordinary semantic understanding. |
| [*BadWAM*](https://arxiv.org/html/2607.15207v1), Sections 3–4.5 | Perturbs observations to change actions while penalizing future-prediction drift, and evaluates closed-loop task outcomes. | A similar-looking forecast does not certify similar behavior. Measure both physical action effects and forecast changes, with replanning timestamps. |
| [*Attacking the Trusted Imagination*](https://arxiv.org/html/2606.22966v1), Sections 3–5 and limitations | Distinguishes corrupting imagination, precisely steering it, and harming a downstream policy or oracle; these can have different outcomes. | Large latent distance is not a validated semantic edit. A null policy effect after a failed or off-distribution manipulation is not semantic invariance. |

Our candidate distinction is **physically validated semantic counterfactuals, paired positive and negative controls, and selective-use failure modes across native representation interfaces**. This remains a hypothesis about contribution; stronger claims require the experiments and a final related-work check.

## Why clean video is not a universal interface

[Fast-WAM](https://arxiv.org/html/2603.16666v2), Section 3.3, compares training/inference variants. Ordinary FastWAM conditions actions on the current frame. Our Optional IDM checkpoint also exposes an IDM path that denoises future latents before constructing action conditioning. Use the *same Optional IDM weights* for `idm` versus `first_frame`; comparisons with a separately trained unconditioned checkpoint answer a different question. The current E1 insertion is before recipient-conditioned video prefill, not a direct donor K/V transplant.

[Zhou, Ni et al., *What Makes World Action Models Generalize?*](https://arxiv.org/html/2609.34981v2), Section 4.3, freezes future features after a varying number of video passes while keeping weights and action denoising fixed. It separates that intervention from Simple-WAM's changed training recipe. **Design implication:** investigate when information becomes action-relevant, without assuming a clean, human-readable forecast is necessary. Do not turn comparisons of separately trained models into within-model causal evidence.

[HUST Faster-WAM](https://arxiv.org/html/2608.04404v1), method and video-step appendix, uses one video pass over noisy future slots and sparse, fused K/V reuse. The native action-facing cache is the appropriate audit target. This is arXiv **2608.04404**, not TeamFasterWAM's different paper **2608.02365**. At the pure-noise endpoint, forward-noising an edited clean video with identical noise removes its signal entirely; that is not a useful semantic intervention. Start with native same-instruction cache contrasts and report that future-slot features can also encode present context.

[LingBot-VA](https://arxiv.org/html/2601.21998v1), Section 3, motivates autoregressive prediction, action generation and observed-memory updates. The [released implementation](https://github.com/Robbyant/lingbot-va) uses a shared backbone, unlike the paper's separate MoT streams. In our runtime, the final video forward writes predicted K/V before action inference. A hook on returned latents would be too late. Protect observed history and both CFG branches, and test at least three chunks with two real cache updates. The released LIBERO bundle targets `libero_10`.

[Long-WAM](https://arxiv.org/html/2610.10528v1), Section 3.2 and Appendix B.2, conditions action denoising on history and **partially noisy** future latents; the published IDM setting stops at sigma 0.9. Protect all history frames and reproduce the native noise level before any substitution. A diagnostic full denoising or decoded visualization is an additional computation, not necessarily the representation consumed by the deployed policy. IDM and co-denoising bundles are separate regimes.

[OpenWAM](https://arxiv.org/abs/2609.07398), Sections 3.1 and 4.2.2–4.2.3, distinguishes training-time attention topology from inference-time video/action schedules. Its schedule study favors synchronized denoising in that setting. Our released backend runs joint inference and already exposes mask hooks. Content patching needs matched layer, denoising step and both noise levels; an arbitrary switch to an isolated mask is an ablation, not evidence for how that topology would perform if trained separately. This is **OpenWAM-Official/OpenWAM**, not the different Stanford project of the same name.

These findings are compatible: whether extra video denoising helps depends on checkpoint training and the interface. We should test within-model predictions rather than force one universal conclusion about “imagination.”

## Interpretation rules for the next study

- Keep recipient instruction, current observation, proprioception, action noise and history fixed within each intervention pair. Record donor construction separately; a recipient re-encoding does not certify removal of donor-language information.
- Prefer simulator-rendered donors without a donor-language-generation call for the first clean property test. A renderer is language-free, but selecting which scene to render still implements our experimental hypothesis.
- If depicted robot motion and object identity covary, describe the result as motion/approach conditioning. A matching trajectory is not proof of object-identity reasoning.
- A nuisance edit must preserve the target's visibility, contrast and physical affordances. Otherwise a behavior change may be a perceptual consequence, not indiscriminate background sensitivity.
- A nonsignificant difference is not invariance. Set an engineering tolerance on separate development data, then estimate whether the held-out interval is inside that tolerance. Report uncertainty and failed gates.
- Use physical positions, contacts and task predicates for behavior. Normalized action differences, pixels and world meters are separate units. Score each forecast only against the part of that chunk actually executed before replanning.
- Do not feed actual future observations to a policy and label the improvement deployable. Oracle future replacement is a separately labeled upper-bound experiment.

The [execution plan](e1_next_steps_20261122.md) turns this review into ordered gates, owners to assign, and a cross-model rollout schedule.
