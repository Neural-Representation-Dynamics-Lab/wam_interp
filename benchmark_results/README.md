# Benchmark results

One JSON per evaluator call, as written by `evaluate_libero()`. Each file holds the full `LiberoEvalConfig`, `run_metadata` (checkpoint, dataset stats, FastWAM task config, repo revision, model dtype, MuJoCo and robosuite versions) and, per trial, the seed, initial-state index, success, termination reason, step counts and per-call inference metadata.

## FastWAM on libero_10

Released checkpoints from `yuanty/fastwam`, run at repo revision `9ce984f`. Settings follow FastWAM's official LIBERO eval (its README and `eval_libero_single.py`): bf16, MuJoCo 3.3.2, 256px render, 30 warm-up steps, replan every 10 steps, action horizon 32, 10 inference steps, 700 max policy steps, seed 42, 50 trials per task (robosuite 1.4.1 here). All conditions share initial states and seeds, so trials with the same index pair up across folders.

| Folder | Checkpoint / condition | Success |
|---|---|---|
| `fastwam/libero_10/uncond_bf16` | uncond, first_frame | 465/500 (93.0%) |
| `fastwam/libero_10/optional_idm_first_frame_bf16` | Optional IDM, first_frame | 476/500 (95.2%) |
| `fastwam/libero_10/optional_idm_idm_bf16` | Optional IDM, idm | 488/500 (97.6%) |
| `fastwam/libero_10/uncond_bf16_published_shift` | uncond, first_frame, `sigma_shift=5.0` | 469/500 (93.8%) |
| `fastwam/libero_10/optional_idm_idm_bf16_published_shift` | Optional IDM, idm, `sigma_shift=1.0` | 490/500 (98.0%) |

Per task (0 to 9):

- uncond: 46, 49, 46, 49, 43, 49, 42, 49, 43, 49
- Optional IDM first_frame: 48, 50, 44, 48, 46, 50, 45, 50, 46, 49
- Optional IDM idm: 50, 49, 48, 50, 50, 50, 47, 50, 44, 50
- uncond, published shift: 47, 49, 49, 47, 41, 49, 43, 48, 47, 49
- Optional IDM idm, published shift: 50, 50, 48, 50, 50, 50, 47, 50, 46, 49

Paired by trial, Optional IDM first_frame succeeds alone on 8 trials and idm alone on 20 (exact McNemar p = 0.036); no single task differs significantly. On LIBERO-Long the Fast-WAM paper reports 95.2% for Fast-WAM (uncond) and 97.6% for Fast-WAM-IDM, a separate checkpoint from Optional IDM.

The first three runs did not apply the sampler shift FastWAM publishes for its released checkpoints (`EVALUATION.sigma_shift`: 5.0 for uncond, 1.0 for Optional IDM; the first_frame run already matched). The `_published_shift` folders re-run the two affected conditions with it, on L40 GPUs with the same seeds and initial states. Paired by trial, neither changes significantly: uncond has 14 old-only and 18 new-only successes (p = 0.60), idm has 3 and 5 (p = 0.73).

Earlier runs in fp16 or with MuJoCo 3.9.0 are not included. With MuJoCo 3.3.2, uncond on task 4 was 20/50 in fp16 and 43/50 in bf16.
