# Benchmark results

One JSON per evaluator call, as written by `evaluate_libero()`. Each file holds the full `LiberoEvalConfig`, `run_metadata` (checkpoint, dataset stats, FastWAM task config, repo revision, model dtype, MuJoCo and robosuite versions) and, per trial, the seed, initial-state index, success, termination reason, step counts and per-call inference metadata.

## FastWAM on libero_10

Released checkpoints from `yuanty/fastwam`, run at repo revision `9ce984f`. Settings follow FastWAM's official LIBERO eval (its README and `eval_libero_single.py`): bf16, MuJoCo 3.3.2, 256px render, 30 warm-up steps, replan every 10 steps, action horizon 32, 10 inference steps, 700 max policy steps, seed 42 (robosuite 1.4.1 here). All conditions share initial states and seeds, so trials with the same index pair up across folders.

| Folder | Checkpoint / condition | Trials per task | Success |
|---|---|---|---|
| `fastwam/libero_10/uncond_bf16` | uncond, first_frame | 50 | 465/500 (93.0%) |
| `fastwam/libero_10/optional_idm_first_frame_bf16` | Optional IDM, first_frame | 10 | 96/100 |
| `fastwam/libero_10/optional_idm_idm_bf16` | Optional IDM, idm | 10 | 100/100 |

Per task (0 to 9): uncond 46, 49, 46, 49, 43, 49, 42, 49, 43, 49; Optional IDM first_frame 10, 10, 8, 10, 10, 10, 9, 10, 9, 10; idm 10 on every task. The Fast-WAM paper reports 95.2% for uncond on LIBERO-Long.

Earlier runs in fp16 or with MuJoCo 3.9.0 are not included. With MuJoCo 3.3.2, uncond on task 4 was 20/50 in fp16 and 43/50 in bf16.
