# Instruction sensitivity diagnostic

This is an exploratory secondary analysis of the complete saved later-state
FastWAM Optional IDM bank. It runs on CPU; it does not load weights, generate new
futures, execute actions, or change the earlier E1 donor selection.

## Design

Three previously studied LIBERO-10 task-0 states, after 32 exactly replayed native
controls; two requested targets (cream cheese/butter); two action wordings
(pick up/put in basket); four seeds (18–21). All 48 futures are included without
norm filtering. The previous 27 AI video reviews were known when this analysis
was designed, so this is neither preregistered nor blinded confirmation.

- Target-word comparison: same state, seed and action wording, different target;
  8 pairs per state, 24 total.
- Action-wording comparison: same state, seed and target, different action;
  8 pairs per state, 24 total. These are not synonymous paraphrases.
- Seed comparison: same state and complete instruction, all six unordered seed
  pairs; 24 pairs per state, 72 total. Comparisons share videos and are dependent.

Compute latent RMSE over generated temporal slots only, requiring exact equality
of the observed anchor. Separately compute mean absolute RGB differences on
decoded frames 1–8 in each 224-pixel-wide camera. Pixel values are on the 0–255
scale. These are sensitivity measures, not target correctness measures.

Keep complete explicit AI video labels. Reuse existing review text with its
source hash and inspect all 21 remaining videos in all nine frames and both
cameras. Unclear labels stay in the denominator. Report both-requested-target
counts over all 24 target-word pairs; do not turn unclear labels into failure
labels or claim that zero confirmed switches proves no semantic change.

Average within each state before averaging states equally. Report all three
state-level results; do not compute a p-value using the repeated seeds as
independent states. Human review remains pending but does not block this
explicitly authorized exploratory analysis.

## Run

From the repository root with its existing validation environment:

```powershell
.venv-validation/Scripts/python.exe -m pytest tests/test_e1_instruction_sensitivity.py -q
.venv-validation/Scripts/python.exe scripts/e1_instruction_sensitivity.py prepare --output results/e1_instruction_sensitivity_fresh
```

Preparation refuses to overwrite an output directory, verifies the source bank
against its saved download manifest, freezes the full factorial records and
analysis source hashes, and generates review sheets. Complete missing entries
in that output's `review.json`, retaining uncertainty and honest provenance.

```powershell
.venv-validation/Scripts/python.exe scripts/e1_instruction_sensitivity.py analyze --output results/e1_instruction_sensitivity_fresh --require-review
```

For numerical inspection before review, omit `--require-review`; output is then
explicitly marked incomplete. To reproduce the completed analysis, use
`results/e1_instruction_sensitivity_20261120` as the output. Its frozen source
must match; use the archived analysis sources when working from a later checkout.
On Linux, substitute the appropriate environment's `python` command.

## Interpretation limits

This tests one checkpoint and one task at three selected later observations.
Requested targets differ from the official soup/tomato task instruction, and the
nine-frame horizon may end before target choice becomes clear. Numerical output
sensitivity does not establish correct instruction grounding or causal use of
future semantics by the action policy. The frozen earlier E1 selection remains
unchanged even if this separate complete-bank review provides new observations.

The native IDM implementation explicitly discards `text_cfg_scale` and
`negative_prompt`; increasing the exposed CFG value would not change this path.
It creates fresh video/action generators from the requested seed and encodes the
formatted positive prompt for the video-denoising context. This is a source audit,
not a new GPU text-embedding intervention. Do not claim a stronger pipeline test.
