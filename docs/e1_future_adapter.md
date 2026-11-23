# Running one future intervention on five WAMs

`FutureAdapter` wraps an already-loaded benchmark adapter. All five released LIBERO backends use the same calls. Pick the checkpoint's native inference condition and turn off compilation. Model loading, simulator reset, history updates and environment action decoding continue to use the existing benchmark API.

```python
from wam_interp.experiments import FutureAdapter

futures = FutureAdapter(loaded_adapter)
donor_result, donor = futures.capture(donor_request)
prediction = futures.predict_with_future(recipient_request, donor)
commands = loaded_adapter.postprocess_actions(prediction.actions, environment="libero")
```

Build both requests with the backend's native observation preparation and inference defaults. Run `begin_episode` and `observe_step` for backends that use them. For a paired experiment, restore the recipient's simulator, history, time index, prompt and action noise before each arm. Use one wrapper per loaded model and run calls sequentially. Donors belong to the wrapper, site and schedule that created them. Keep the intervention within the same temporal context, including the absolute chunk index for LingBot. The donor contains detached CPU copies of the editable slots. Capturing with `donor=...` also returns the recipient's original features at that site.

| Backend and condition | Default intervention site |
|---|---|
| FastWAM Optional IDM, `idm` | Clean future video latents, excluding frame zero |
| Faster-WAM, `one_pass_future_cache` | Future keys and values in the middle active sparse layer |
| LingBot-VA, `video` | Middle-layer predicted video cache, preserving observed history and the initial anchor |
| Long-WAM IDM, `idm` | Partially denoised future latents before clean history is concatenated |
| OpenWAM, `joint` | Video keys and values consumed by action queries at the middle layer, first denoising call |

The optional `layer` argument selects a supported layer. OpenWAM also accepts a denoising call `stage`. The shared scene experiment uses two actual donor observations, one with a relocated target and one with a relocated distractor. It transplants the model's own features into the original recipient and measures its gripper after ten executed controls. Each donor's representation may also carry information about the current scene. Gross ablation, a native scene transplant and a clip intervention answer different questions.

## Donors from recorded RGB

Check `futures.supports_clip` before requesting this additional operation. The input is an anchor-first uint8 RGB montage after the backend's camera transformations, sampled at the native video interval. Its first frame must match the prepared recipient image. The checked FastWAM and Long-WAM LIBERO releases both take nine horizontally paired 224-pixel camera views at control offsets 0, 4, ..., 32.

```python
# FastWAM Optional IDM
donor = futures.from_clip(request, clip)

# Long-WAM IDM. Reuse one dedicated noise seed for all clip arms in a state.
donor = futures.from_clip(request, clip, noise_seed=20261123)
```

FastWAM encodes the clip with its native VAE and replaces only the clean future slots. Long-WAM prepends the recipient's native sampled history, encodes the resulting sequence, keeps only the future slots, and applies its training noise schedule at the checkpoint's imagination timestep. Forward-noising an externally specified clip is an experimental intervention. The released inference path ordinarily partially denoises its own future. Faster-WAM, LingBot-VA and OpenWAM currently accept captured feature donors through the common API. Clip requests for these backends raise `NotImplementedError`.

## Reproduction

Use the checkpoint paths and isolated environments in [the model handoff](e1_multimodel.md). Take the native `evaluate_benchmark.py` invocation printed by `scripts/e1_model_matrix.py baseline`. For each of the five variants, use its native condition, one LIBERO-Long bowl episode (task 3) and seed zero. Append:

```bash
--e1-feature-audit results/shared/BACKEND/parity.json \
--e1-scene-protocol /path/to/prior/results/e1_native_replan/protocol.json
```

The audit compares native actions, capture-only, zeroing and exact self-replay on three decisions. For FastWAM and Long-WAM the following extra option reads the archived book-task fixtures from the [previous A100 study](e1_a100_study.md). Use a fresh output directory for each run.

```bash
--e1-clip-study /path/to/prior/results
```

The clip study first scores held-out pairs without using policy outputs. A direction template is the mean normalized positive-minus-negative feature difference on development starts. A held-out pair counts as correctly oriented when its paired difference projects positively on that template. This measures direction information in a matched pair. It does not establish single-clip classification or model use. The arm pilot trains on two starts and tests on two. Book and mug each use four development starts and up to ten held-out starts at seed zero. Source failures remain in the denominator. Each accepted behavioral pair also requires the unedited video's executed endpoint to stay within 20 mm of native. The isolated 32-control diagnostic measures signed half-differences in world x. Its FastWAM regression compares full world trajectories against the archived native and encoded arms.
