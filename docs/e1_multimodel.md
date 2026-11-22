# E1 tooling and cross-model handoff

Start with the [current research plan and TODOs](e1_next_steps_20261122.md), then the [literature/code audit](e1_research_review_20261122.md). Earlier `e1_*` documents describe historical protocols; they are not an instruction to rerun every pilot.

## What this PR implements

The [A100 follow-up](e1_a100_study.md) adds native future-feature hooks for Faster-WAM, LingBot-VA, Long-WAM and OpenWAM. FastWAM Optional IDM provides the clean-latent path. `src/experiments/futures.py` captures/replaces future latents before action K/V construction, preserves the observed anchor, validates shape/dtype/device/finiteness, and removes its hook even after errors. Inactive instrumentation does not alter native inference. Do not share one hooked model concurrently between requests.

| Reusable module | Purpose |
|---|---|
| `e1.py`, `fastwam_session.py`, `libero_pilot.py` | Paired branch execution, adapter-driven action decoding, state/RNG replay, provenance and failure records |
| `futures.py`, `video_bottleneck.py` | Scoped native future intervention and codec controls |
| `simulator_video.py`, `simulator_libero.py` | Controlled simulator-video sources and their provenance |
| `simulator_materials.py`, `simulator_identity.py`, `physical_relocation.py` | Appearance edits versus actual visual/collision geometry relocation |
| `trajectories.py`, `annotation.py` | Time-aligned executed-prefix scoring and explicit missing/occluded landmarks |
| `controlled_replication.py`, `matching.py`, `native_pairs.py` | Paired analysis and candidate eligibility without pretending a failed donor search succeeded |
| `model_capabilities.py` | Lightweight native-interface inventory and command planning for all five backends; unsupported E1 transplants fail explicitly |

The `scripts/e1_*` files compose these modules for specific experiments. They save machine-readable reports to ignored `results/`. The source-overlay packager excludes checkpoint/data/result artifacts. After this PR is merged, a normal Git checkout is preferable to an overlay; overlays remain useful for reproducing an uncommitted historical snapshot.

## CPU preparation

```bash
python -m pip install -e '.[research]'
python scripts/e1_model_matrix.py inventory
python scripts/run_e1.py rehearse --output results/e1_synthetic_rehearsal
python -m pytest tests/test_e1_model_capabilities.py -q
```

The rehearsal is labeled synthetic and is **not experimental evidence**. Full E1 tests additionally require torch; simulator tests use mocks and do not download assets. The existing CI runs the backend tests in their own dependency environments. For FastWAM's actual GPU setup and released assets, use [e1_quickstart.md](e1_quickstart.md) and [setup_e1_linux.sh](../scripts/setup_e1_linux.sh). Other backends use their existing guides:

- [Faster-WAM](fasterwam_validation.md)
- [LingBot-VA](../src/models/lingbot_va/README.md)
- [Long-WAM](../src/models/longwam/README.md)
- [OpenWAM](../src/models/openwam/README.md)

Use Linux, matching CUDA drivers and **bf16**. Do not silently fall back to fp16. Preserve each backend's normalization, camera layout and released configuration. OpenWAM requires a separate transformers-5 environment; the others use transformers 4. Checkpoint size alone does not establish A100-40GB fit: measure peak allocation during the first native run and keep only one model loaded at a time. No new GPU run or memory-fit result is claimed by this PR's planning tools.

## Prepare one native run per backend

These commands **print plans and do not execute GPU inference**. Run from the repository root. Paths below follow the existing backend guides; replace them with the actual host paths. The JSON output contains `argv` and a shell-quoted `linux_command`. Inspect the plan, then execute its command in that backend's environment. Add `--checkpoint-revision` with the resolved download revision and `--check-local-assets` to verify local file/directory presence. This path check does not prove weight compatibility; loaders perform that validation.

```bash
python scripts/e1_model_matrix.py baseline --variant fastwam_optional_idm \
  --checkpoint checkpoints/fastwam/libero_optional_idm_2cam224.pt \
  --dataset-stats checkpoints/fastwam/libero_optional_idm_2cam224_dataset_stats.json \
  --benchmark-root third_party/LIBERO --output results/e1_native/fastwam_idm

python scripts/e1_model_matrix.py baseline --variant fastwam_optional_first_frame \
  --checkpoint checkpoints/fastwam/libero_optional_idm_2cam224.pt \
  --dataset-stats checkpoints/fastwam/libero_optional_idm_2cam224_dataset_stats.json \
  --benchmark-root third_party/LIBERO --output results/e1_native/fastwam_first_frame

python scripts/e1_model_matrix.py baseline --variant fasterwam \
  --checkpoint checkpoints/fasterwam_release/libero/step_021700.pt \
  --dataset-stats checkpoints/fasterwam_release/libero/dataset_stats.json \
  --benchmark-root third_party/LIBERO --output results/e1_native/fasterwam

python scripts/e1_model_matrix.py baseline --variant lingbot_va \
  --checkpoint checkpoints/lingbot_va/libero_long \
  --benchmark-root third_party/LIBERO --output results/e1_native/lingbot_va

python scripts/e1_model_matrix.py baseline --variant longwam_idm \
  --checkpoint checkpoints/longwam/libero_idm/model.pt \
  --dataset-stats checkpoints/longwam/libero_idm/dataset_stats.json \
  --benchmark-root third_party/LIBERO --output results/e1_native/longwam_idm

python scripts/e1_model_matrix.py baseline --variant openwam \
  --checkpoint checkpoints/openwam/libero \
  --benchmark-root third_party/LIBERO --output results/e1_native/openwam
```

Before executing, configure and inspect the pinned ordinary LIBERO checkout from [the evaluator guide](libero_evaluation.md). `--benchmark-root` on the ordinary LIBERO CLI records provenance; it does not redirect Python's imported `libero` or its configured asset paths. In the chosen environment, check both explicitly:

```bash
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
export LIBERO_CONFIG_PATH="$PWD/results/libero_config"
python - <<'PY'
from pathlib import Path
import libero
from libero.libero import get_libero_path
root = Path('third_party/LIBERO').resolve()
assert Path(libero.__file__).resolve().is_relative_to(root), libero.__file__
for name in ('bddl_files', 'init_states', 'assets'):
    path = Path(get_libero_path(name)).resolve()
    assert path.is_dir() and path.is_relative_to(root), (name, str(path))
    print(name, path)
PY
git -C third_party/LIBERO rev-parse HEAD
# Expected: 8f1084e3132a39270c3a13ebe37270a43ece2a01
```

The baseline defaults to task 0 of `libero_10`, one episode, seed 0. Expand a declared selection with `--task-ids 0 1 --episodes 4`. The generated command delegates to the existing `evaluate_benchmark.py`, preserving native replan periods, LingBot's full-chunk observation updates and OpenWAM's trained mask. Do not force all backends to execute 32 actions between replans. A successful native run is a smoke check, not a new E1-hook validation or performance estimate.

## Required evidence for each new E1 boundary

1. Record source/upstream/checkpoint/statistics identities; raw and prepared input hashes; prompt; history; RNG; effective dtype; attention topology; video/action noise levels; chunk index and exact temporal/token layout.
2. Compare uninstrumented, capture-only, exact identity and repeated native inference on identical inputs. Capture-only copying/synchronization changes measured runtime, so benchmark native latency separately. Do not infer exact parity from CPU test doubles.
3. Check that only the declared future slots change. For Long-WAM protect *all* history; for LingBot protect observed caches and CFG pairing; for Faster-WAM preserve sparse fused layer structure; for OpenWAM preserve the joint schedule and action path.
4. Restore the complete environment/controller and backend state before every paired arm. Test exceptions, absent hooks and cleanup. For stateful backends test multiple chunks and observed-cache updates, not just episode initialization.
5. Save normalized actions, decoded commands, executed physical states, contacts, success and termination reason separately. Never compare different backends' normalized action coordinates as if they were world positions.

The A100 follow-up implements each backend’s native feature boundary and records capture, identity and intervention checks on real inputs. Consult each run’s per-backend parity report for completed checks and failures.
