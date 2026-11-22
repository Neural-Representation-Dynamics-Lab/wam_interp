# E1: next experiments and meeting checklist

Prepared for the 22 November meeting, approximately 21:00 Malaysia time. Proposed priorities and owners below need team agreement; no teammate has been messaged or assigned work automatically. See the [source review](e1_research_review_20261122.md) for the literature and [model handoff](e1_multimodel.md) for commands.

The [A100 follow-up protocol](e1_a100_study.md) and [completed native results](e1_a100_native_results.md) record the later execution of this planning document.

## Decision in one paragraph

Keep the completed FastWAM experiments as exploratory causal-sensitivity evidence. **First establish native task/target following with replanning, then test one validated property while holding depicted motion fixed, then replicate across architectures.** Continue the forecast/execution measurement as a paired readout, not a substitute for the semantic intervention. Stop expanding counter-instruction donor banks that fail their visual gates. Do not rerun five large suites before knowing whether the measurement and the edit work.

## What is established, and what is not

All E1 GPU experiments so far used **FastWAM Optional IDM, bf16, A100 40GB**. The repository's existing benchmark results for other models are a separate source of evidence.

| Experiment | Observed result | What it supports / leaves open |
|---|---|---|
| Simulator-rendered future approach replication | 448 branches; 16 test states across two tasks; 32/32 primary seed contrasts in the expected direction; mean target-axis shift about 17.53 cm | Rendered visual conditioning can redirect execution without generating donor language. Robot motion and target approach still covary. Seeds are repeats, not 32 independent states. |
| Fixed-motion material edit | 240 branches; four robot states on one arrangement; changes roughly 0.5–1.3 mm | No validated target-identity effect. The edit did not establish a native change in perceived target identity. |
| Complete visual-asset swap | 48 branches; 0/4 development cases passed the native gate | The four held-out layouts were not run. This is a failed positive control, not proof that identity is irrelevant. |
| Physical object relocation | 48 branches plus 12 repeats; 0/8 layout/seed cases passed in the first 32 controls, without replanning | The policy mostly retained its old approach. Not evidence of eventual episode failure or a universal absence of object semantics. Untouched cream was not an independently validated target-following control. |
| Forecast/execution comparison | Three original native episodes succeeded; selected replanned chunks were compared to their new forecasts | Approximate visible 2D agreement; human landmark review remains pending. No paired evidence yet that imagination improves success. |

Numbers are transcribed from the owner's saved reports; raw generated results are intentionally excluded from this PR. Local evidence paths, report hashes and detailed caveats remain in the owner's research archive. The source scripts write JSON/NPZ outputs for reproduction. The physical relocation record checked for this review used `sigma_shift=1.0`, matching published Optional IDM settings; a sampler-shift explanation for that failure is unestablished.

## P0 — deliver tooling and diagnose the native positive control

Proposed owner: Dun Li, with one backend maintainer checking native protocol parity.

- [x] Preserve the original research checkout and create a separate PR branch from current main.
- [x] Package reusable E1 future capture, paired replay, simulator edits, trajectory metrics and local tests; exclude weights, raw results, simulator checkouts, caches and environments.
- [x] Audit papers and native interfaces for FastWAM, Faster-WAM, LingBot-VA, Long-WAM and OpenWAM.
- [x] Add CPU-only capability inspection and native-baseline command planning for all five backends. Only FastWAM Optional IDM has an implemented E1 future transplant in this PR.
- [ ] Review and merge the tooling PR after CI and backend-owner review. PR submission is not GPU validation.
- [ ] Freeze a native-control protocol with exact source/checkpoint/statistics hashes, benchmark revision, camera transforms, precision, sampler settings, warm-up and replan period.
- [ ] Select two official task/phase settings with unambiguous target criteria. For a multi-object instruction, reaching either allowed object may be valid; do not invent a mandatory first-object order. Use a matching BDDL success predicate, not the original soup task's success flag for a rewritten tomato instruction.
- [ ] Verify unchanged official episodes first. Check task text, checkpoint task coverage, reset state, visibility, action decoding and cache updates. Save native frames and physical target/gripper tracks.
- [ ] Run a bounded **32-episode development diagnostic**: two task/phase settings × four initial states × original/small legal relocation × `idm`/`first_frame`, one fixed paired seed. Both modes use the same Optional IDM checkpoint. Choose native-valid placements before policy outcomes, keep all failures, and report success separately from target approach. This budget is proposed, not already executed.
- [ ] Compare each episode's first 10 and 32 controls with its full replanned continuation. Record time to first target approach/contact, target distance, wrong-object contact, and full task success. A 32-step diagnostic is not the native 10-step replan protocol.
- [ ] Treat large soup/tomato exchanges as a separate OOD stress test. If small supported moves work but exchanges fail, study that boundary; do not silently replace the failed test with an easier success.

**Exit decision:** if the native original task fails, investigate implementation/settings before semantics. If original succeeds but relocation fails throughout the episode, retain a valid robustness failure and choose a supported task/property for a semantic-use test. If only the first chunk fails, revise the semantic measurement window prospectively and keep the old result. Do not select held-out cases on transplant effect size.

## P1 — one clean property test, including its negative result

Proposed owner: Dun Li; independent visual/semantic reviewer to be assigned. Exploratory execution can continue without human labels, but claims stay exploratory.

Start with **target location or target–receptacle relation**, whichever passes P0. Return to target identity only after an edit is demonstrably recognized. Then consider contact versus near miss and event order as separate later studies; do not bundle them into one intervention.

1. Use a non-contact approach segment from a simulator reference. Render paired videos with identical robot joint trajectory, camera trajectory, timestamps, lighting and current anchor. Change one declared object property in the future. Check both cameras, masks, geometry and object overlap; verify that the robot path remains identical. The scene change may make the trajectory a wrong-target approach; label that counterfactual conflict explicitly.
2. Add a task-irrelevant edit with comparable image region/visibility, plus a second instruction under which the same changed property becomes relevant. Within each instruction's donor pair, keep recipient language fixed. Include a motion-only pair as a competing explanation, not as the semantic positive control.
3. Run native, exact identity and unedited simulator-video encoding controls. The unedited synthetic donor must be shown to preserve ordinary behavior sufficiently before interpreting an edited donor. Native-VAE round-trip parity alone does not establish simulator-donor validity.
4. Verify the intended content independently of the policy outcome. Save all failed edits and any visibility/distribution-shift warnings. A numerical latent difference or a different prompt is not a semantic manipulation check. A human can review after exploratory runs, but not retroactively turn outcome-selected cases into preregistered evidence.
5. Freeze the contrasts and tolerances on development states, then use new initial states/object arrangements. Proposed confirmation floor: **20 held-out states, at least two task/phase settings, two seeds per state** for one property. This is a starting budget, not a power calculation; adjust once using development variability and record the revision before the held-out run.

**Temporal-consistency gate:** a future-only edit of a static object's identity or position can contradict the unchanged first frame. Correct single-frame rendering and collision geometry do not resolve that contradiction. Prefer a physically possible continuation from the common observed state; otherwise label the manipulation as an inconsistent counterfactual and include a current-scene × future-scene consistency control. Do not call a response to a teleportation, identity morph or anchor discontinuity clean semantic use. If no valid fixed-motion pair can be constructed, change the property instead of forcing the test.

For state `s`, instruction relevance `r`, and a physical behavior metric `Y`, calculate

```
effect(s,r) = Y(edited future, s,r) - Y(unedited matched future, s,r)
selective effect(s) = effect(s,relevant) - effect(s,irrelevant)
```

Use a signed, preregistered direction towards the displaced target where meaningful. Also report the full 3D effect and wrong-target contact, so projection cannot hide unrelated motion. Average seeds within states, then summarize states; bootstrap whole state blocks, stratify task summaries, and avoid generalization claims based on only two tasks. Report the denominator of eligible and ineligible states and reasons.

Negative controls are co-primary. Define practical nuisance tolerances using repeat/codec noise and task geometry on development data. A confidence interval inside the tolerance supports limited invariance; an interval crossing it is inconclusive. A large nuisance response is a substantive failure of selectivity. Do not normalize away the effect with a near-zero or unstable denominator, or equalize latent norms by interpolating away the intended semantics.

## P2 — expand by native interface, not by copying tensors

| Order | Backend / proposed owner | Implementation and first acceptance test |
|---|---|---|
| 1 | FastWAM Optional IDM / Dun Li | Retain the tested clean-latent boundary. Compare IDM/first-frame with the same checkpoint. Establish P0 and one valid P1 property. |
| 2 | Faster-WAM / Dun Li or its maintainer | Add a scoped hook after sparse fused video prefill. Preserve current-frame K/V, `None` layer slots, native RoPE and timestep. Test capture-only and identity on a real observation; then content versus matched current-token controls. Do not feed clean RGB latents into the one-pass noise endpoint. |
| 3 | LingBot-VA / LingBot backend maintainer | Patch only declared predicted-video cache entries before actions consume them, including CFG handling. Restore KV and streaming-VAE state and run three chunks/two observation updates. Keep real history intact. Separate first-chunk behavior from later chunks. |
| 4 | Long-WAM IDM / backend maintainer | Hook partially noisy future latents before action prefill; preserve all clean history. Match native sigma and future noise. Validate capture/identity and history replay before any semantic noising experiment. COD needs its own stage-specific design. |
| 5 | OpenWAM / backend maintainer | Add stage-indexed action-facing K/V capture/patching with exact eager inference. Preserve checkpoint mask and both noise schedules. Existing mask hooks can measure gross dependence but do not isolate content. Use its separate transformers-5 environment. |

The table is a work allocation proposal, not a claim these four new E1 hooks exist. First run each backend's **native** one-episode smoke through `scripts/evaluate_benchmark.py`. Existing all-backend benchmark results are useful baselines; they do not validate the new instrumentation. For each hook require: same-observation unhooked/capture/identity parity, finite tensors, correct protected context, cleanup on failure, and deterministic replay. Run hooks sequentially on a loaded model and record the complete adapter state.

Use `libero_10` for the shared released-checkpoint baseline. LingBot's available LIBERO checkpoint does not justify claiming coverage of all LIBERO suites. Property-specific tests may have different supported task subsets; publish that coverage. Compare effect direction, selective-use contrast, nuisance sensitivity and native success *within* each model. Do not pool normalized actions or raw latent norms across architectures, or claim that architecture alone caused differences between separately trained checkpoints.

## P3 — align forecasts, then assess usefulness

Proposed owner: trajectory-analysis teammate, with Dun Li maintaining timestamp contracts.

- Save a forecast at every studied replan, not just the beginning of the episode. Compare it only with the executed prefix of its own action chunk. Preserve per-backend first-chunk lengths and sampled observation timing.
- Measure gripper and object tracks, missingness/occlusion, ADE/FDE at matched times, contact/lift events and object–receptacle relations. Simulated execution provides world coordinates; imagined-video pixels do not become 3D without a validated reconstruction. Keep units separate and include stationary/constant-velocity baselines.
- For high-noise representations, an additionally denoised diagnostic video is not automatically the action's actual forecast. Label it as diagnostic. Prefer feature-space semantic probes and controlled physical outcomes until a visualization link is validated.
- After selective-use controls pass, run same-weight Optional IDM `idm` versus `first_frame` under native replanning on paired episodes, including a separately defined OOD subset. Report paired successes/failures, latency and memory. This tests usefulness for the sampled distribution; alignment alone does not.

## What to decide at the meeting

1. Agree that the primary contribution is a **property-specific selective-use audit**, with negative controls and failed manipulation checks published, rather than another future-swap demonstration.
2. Assign one native-protocol reviewer and one independent semantic/landmark reviewer. Assign Faster-WAM and LingBot hook owners before expanding to Long-WAM and OpenWAM.
3. Approve the bounded P0 diagnostic and choose the first property only after it. Allocate GPU time after checking availability; no allocation or run is implied here.
4. Review this modular tooling PR now. Numerical JSON publication should be a separately reviewed, compact evidence package; keep raw NPZ/video/checkpoint artifacts outside Git.

**Suggested update:** “We have a reproducible FastWAM intervention pipeline and clear evidence that future visual conditioning redirects actions. The fixed-motion identity controls have not validated semantic selectivity, and physical swaps failed in the first chunk. Next we will check full native replanning, establish a valid property manipulation, then reproduce that selective-use test in Faster-WAM and LingBot. Long-WAM and OpenWAM are mapped out, but their intervention interfaces are different. We also found substantial related work, so our novelty has to be the controlled semantic audit and its negative results.”
