# Visual-bottleneck feasibility pilot

This is Phase 1 of the 21 November semantic-faithfulness plan. It tests whether
the previously measured whole-future donor effect survives a native video VAE
decode/encode round trip. It does not remove all language influence or isolate
target identity. No model weights are changed.

Reusable codec functions are in `src/experiments/video_bottleneck.py`; the
experiment runner is `scripts/e1_video_bottleneck.py`. The codec imports torch
only when called and never receives text context. Its eager native encode reads
the entire clip, preserving temporal codec behavior. Decode/encode compute stays
at the loaded model dtype (bf16 in this pilot); tensors exported to NPZ are
converted to float32 for storage. There is no additional uint8/JPEG bottleneck.
Native RGB clipping to [-1,1] is preserved and its fraction recorded.

The reconstructed observed anchor is measured before restoration. All inserted
futures receive the original exact observed anchor. Thus the tested operation is
future-slot replacement by a reconstructed clip, not wholesale substitution of
the input observation. Original/reconstructed RGB tensors and rendered review
sheets are retained for all six state/seed cases and all three input futures.

## Bounded workflow

1. `prepare` validates the historical equal-norm experiment's frozen inputs,
   copies small inputs, fingerprints historical baseline camera traces, and
   freezes the current inference source and protocol.
2. `controls` constructs round trips and runs native, identity and round-tripped
   native branches for every state/seed case. Historical baseline future,
   actions and all recorded simulator arrays must match exactly. Identity must
   reproduce actions and all simulator arrays exactly.
3. Every round-tripped baseline must have <=1 cm EEF endpoint difference and
   <=5 mm mean EEF path difference from native. These are engineering stop
   thresholds fixed before new results, not calibrated equivalence bounds or a
   statistical invariance claim. A failed case stops donor execution after all
   baseline controls have completed.
4. When baseline gates pass, review original/reconstructed video sheets for all
   baseline and donor futures; complete `codec_review.json` with actual evidence.
   AI-assisted exploratory review is authorized; human validation stays pending.
5. `donors` executes raw same/alternative and round-tripped same/alternative
   branches only after the gates and explicit codec review pass. Review and codec
   hashes are frozen. Historical raw-donor action outputs must reproduce exactly.

The maximum is 3 states x 2 seeds x 7 arms = 42 branches, each 32 controls.
There is no silent restart, threshold relaxation, or expanded donor search.
Any failure is retained in status JSON and audited before a fresh run is created.

## Running on the prepared Linux A100 environment

```bash
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export LIBERO_CONFIG_PATH="$PWD/results/libero_config"
export DIFFSYNTH_DOWNLOAD_SOURCE=huggingface
.venv-e1/bin/python scripts/e1_video_bottleneck.py prepare --output results/e1_video_bottleneck_20261121
.venv-e1/bin/python -u scripts/e1_video_bottleneck.py controls --output results/e1_video_bottleneck_20261121
# Only after all baseline gates and actual codec review pass:
.venv-e1/bin/python -u scripts/e1_video_bottleneck.py donors --output results/e1_video_bottleneck_20261121
```

The historical experiment folders, released checkpoint, statistics, base assets,
and simulator environment must already exist. They are not downloaded by this
runner. The input paths and settings are declared in
`configs/experiments/e1_video_bottleneck.json`.

## Interpretation

Average the two seeds within each state; report all three development-state
results. The primary comparison is the signed alternative-minus-same world-y
contrast for raw and reconstructed donors. Report the change between those
contrasts, 3D endpoint/path separation, each baseline gate, clipping fractions,
image reconstruction errors, failure counts and semantic-review limits.

If the baseline codec gate fails, do not infer that hidden language was removed,
that nonvisual content is necessary, or that the old transplant effect was
invalid. The correct result is that this codec path has not provided a clean
instrument. If gates pass and effects survive, conclude only that sufficient
information survived this tested RGB bottleneck. Neither outcome alone answers
task success, general semantic faithfulness, or human-readable information use.

Full simulator camera traces stay in `execution.npz`; smaller derived
`kinematics.npz` files allow local analysis. Results are JSON plus NPZ and images.
Checkpoints, environments, simulator assets, and caches must not be committed.

## Completed development run

The 21 November A100 run completed all 42 branches and passed the independent
saved-array/hash audit. All six baseline round trips stayed within the fixed
limits (endpoint drift 1.154–1.965 mm), and all identity replays were exact.
Equal-state mean signed world-y donor contrast was -2.601 cm raw and -2.730 cm
after reconstruction; 3D endpoint separation was 13.799 cm and 13.643 cm.
These are descriptive feasibility results on development states, with human
review pending. See `results/e1_video_bottleneck_20261121/README.md` and
`analysis.json` for the full result, per-state variation and limitations.
