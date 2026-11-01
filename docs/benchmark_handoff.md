# LIBERO-Plus and RoboTwin 2.0 handoff

Status: implemented and CPU protocol-tested; **all six real-checkpoint/GPU
episodes below are pending**. CPU fixtures do not establish policy performance,
simulator compatibility on your server, or numerical checkpoint parity. Do not
report a success rate until retaining the JSON from an actual run. A single
failed episode is not itself evidence of an incorrect integration.

Use this checkout on Linux, Python 3.10, an H100 with sufficient free memory,
CUDA 12.8-compatible drivers, Vulkan/EGL support, internet access, and substantial
disk space. Native extension builds also require a matching CUDA toolkit, a C++
compiler, cmake, ninja, git, and unzip. No remote server access is assumed here.
Commands run from the repository root unless explicitly stated otherwise.

## Isolation and revisions

Keep ordinary LIBERO, LIBERO-Plus, and RoboTwin in separate virtual environments.
LIBERO-Plus exports the same `libero` namespace as LIBERO; installing both into
one environment can silently select the wrong benchmark. The evaluator checks
the imported module path, Git HEAD, classification, and exact suite size before
creating environments. Counts are spatial 2,402; object 2,518; goal 2,591;
`libero_10` 2,519 (10,030 total). Task IDs in our API/CLI are **zero-based suite
indices**; classification IDs in the official JSON are one-based labels.

| Source | Required revision |
| --- | --- |
| [LIBERO-Plus](https://github.com/sylvestf/LIBERO-plus) | `4976dc30028e805ff8094b55501d532c48fec182` |
| [RoboTwin](https://github.com/RoboTwin-Platform/RoboTwin) | `c3ddfa8b97d5519efa828b075999bd0006778e5e` |
| [LingBot-VA runtime/config](https://github.com/Robbyant/lingbot-va) | `7c6ffa9bfc4b83582cafc860fab4c82cc7deeeeb` |

Do not substitute latest RoboTwin: this integration intentionally targets the
legacy `script/` task protocol, not the current `scripts/` layout. The evaluator
temporarily changes cwd to the simulator checkout and restores it on exit; run
one simulator per process.

## LIBERO-Plus environment and assets

```bash
export WAM_ROOT="$PWD"
python3.10 -m venv .venv-plus
source .venv-plus/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -e '.[fastwam,fasterwam,lingbot_va,libero_plus]' pytest
git clone https://github.com/sylvestf/LIBERO-plus.git third_party/LIBERO-plus
git -C third_party/LIBERO-plus checkout --detach 4976dc30028e805ff8094b55501d532c48fec182
python -m pip install -e ./third_party/LIBERO-plus --no-deps --config-settings editable_mode=compat
```

Install OS packages with your server administrator if absent:
`libexpat1 libfontconfig1-dev libpython3-stdlib libmagickwand-dev libegl1`.
`wand` and `scikit-image` (the fork's extra Python requirements) are included
in our `libero_plus` extra. Do not install LIBERO's old training requirements.

Download the [official assets](https://huggingface.co/datasets/Sylvest/LIBERO-plus)
and extract them into `third_party/LIBERO-plus/libero/libero/`, so
`assets/new_objects`, `assets/textures`, and the other native asset directories
exist. No demonstration dataset is needed. Record the resolved dataset revision:

```bash
python - <<'PY'
from pathlib import Path
from huggingface_hub import HfApi, hf_hub_download
import json, zipfile
root = Path('third_party/LIBERO-plus/libero/libero').resolve()
revision = HfApi().dataset_info('Sylvest/LIBERO-plus').sha
archive = hf_hub_download('Sylvest/LIBERO-plus', 'assets.zip', repo_type='dataset', revision=revision)
with zipfile.ZipFile(archive) as stream:
    for member in stream.infolist():
        # entries sit under the authors' own path (inspire/hdd/.../LIBERO-plus-0/assets/...)
        member.filename = member.filename[member.filename.index('assets/'):]
        stream.extract(member, root)
Path('checkpoints').mkdir(exist_ok=True)
Path('checkpoints/libero_plus_assets.json').write_text(json.dumps({'revision': revision, 'archive': archive}, indent=2))
PY
export LIBERO_CONFIG_PATH="$WAM_ROOT/.libero-plus"
export MUJOCO_GL=egl
python - <<'PY'
import os, yaml
from pathlib import Path
root = Path('third_party/LIBERO-plus/libero/libero').resolve()
folder = Path(os.environ['LIBERO_CONFIG_PATH'])
folder.mkdir(exist_ok=True)
paths = dict(benchmark_root=str(root), bddl_files=str(root/'bddl_files'),
             init_states=str(root/'init_files'), datasets=str(root.parent/'datasets'), assets=str(root/'assets'))
(folder/'config.yaml').write_text(yaml.safe_dump(paths))
from wam_interp.evaluations.LIBERO_PLUS.environment import identify_benchmark
from wam_interp.evaluations.LIBERO_PLUS import LiberoPlusEvalConfig
for suite in ('libero_spatial', 'libero_object', 'libero_goal', 'libero_10'):
    benchmark, _, _, _ = identify_benchmark(LiberoPlusEvalConfig(task_suite_name=suite, benchmark_root=root.parents[1]))
    print(suite, benchmark.n_tasks)
PY
```

This configuration uses the checkout's fixed initial states. Variant state-file
mapping follows upstream language/view/table/light/new-object rules, including
the single-state new-object reshape. Serialized state files are trusted upstream
inputs loaded with `weights_only=False`; do not use untrusted pickle files.

## RoboTwin environment and assets

Deactivate the Plus env first. Follow the official host/Vulkan/CUDA prerequisites
from [RoboTwin setup](https://robotwin-platform.github.io/doc/usage/robotwin-install.html),
but use the pinned checkout and commands below, not the latest install script.
open3d also needs a system `libGL.so.1` (`libgl1` on Ubuntu).

```bash
deactivate
python3.10 -m venv .venv-robotwin
source .venv-robotwin/bin/activate
python -m pip install --upgrade pip setuptools wheel ninja cmake
python -m pip install -e . numpy==1.26.4
git clone https://github.com/RoboTwin-Platform/RoboTwin.git third_party/RoboTwin
git -C third_party/RoboTwin checkout --detach c3ddfa8b97d5519efa828b075999bd0006778e5e
python scripts/prepare_robotwin.py --root third_party/RoboTwin
python scripts/prepare_robotwin.py --root third_party/RoboTwin --install
python -m pip install -e '.[fastwam,fasterwam,lingbot_va,robotwin]' pytest
(cd third_party/RoboTwin && bash script/_download_assets.sh)
```

NumPy goes in with the base install because `prepare_robotwin.py` imports
`wam_interp.evaluations.ROBOTWIN`, whose runner needs it.

`prepare_robotwin.py` is read-only unless `--install` is provided. Installation
uses PyTorch 2.7.1/cu128 instead of the simulator's 2.4.1 pin, Hub 0.36.2 instead
of 0.25.0, NumPy 1.26.4, and Zarr 2.18.3. It omits the deprecated `azure`
metapackage; local pre-generated instructions do not need a cloud service.
It runs the pinned upstream SAPIEN/MPLib patches and PyTorch3D/CuRobo build
steps with those constraints active. CuRobo stays on upstream v0.7.8.
This adjusted dependency combination still requires validation on the target
Linux/H100 machine; installer success alone is not an episode validation.
If extension compilation fails, retain the log and resolve the host/toolchain
problem rather than silently changing revisions.

The asset script downloads/extracts backgrounds, embodiments and objects, then
rewrites embodiment asset paths. Verify those paths and an actual task import:

```bash
python - <<'PY'
from pathlib import Path
from wam_interp.evaluations.ROBOTWIN import RoboTwinEvalConfig
from wam_interp.evaluations.ROBOTWIN.environment import open_environment
cfg = RoboTwinEvalConfig(task_name='beat_block_hammer', robotwin_root=Path('third_party/RoboTwin').resolve())
with open_environment(cfg) as frontend:
    print(type(frontend.env).__name__, frontend.args['left_robot_file'], frontend.args['right_robot_file'])
PY
```

Task creation verifies source/API/configuration and robot configuration files;
the first accepted expert rollout additionally exercises meshes, planners and
rendering. Setup/screening failures are recorded with seed and exception. The
default start seed is `100000 * (1 + seed)`; each rejected seed increments it.
`max_seed_attempts` is finite. One accepted episode is the smoke default.
Instructions come verbatim from the native seen/unseen pool (100 descriptions),
after expert-valid seed screening and a fresh reset with the same seed.

### RoboTwin and LIBERO on a Slurm cluster

Tested on GT Futurama L40/L40S nodes, without root or containers:

1. `scripts/cluster/robotwin_probe.sbatch` checks that a node can render with SAPIEN (Vulkan, ray tracing).
2. `scripts/cluster/curobo_build.sbatch` builds the CuRobo wheel once into `$HOME/wheels`. RoboTwin needs CuRobo even for evaluation, because seed screening plans with it.
3. `scripts/cluster/robotwin_eval.sbatch --backend <name> --task-names ... --episodes N` builds the environment with the pins the legacy stack needs, downloads RoboTwin assets and the checkpoint at pinned revisions, and runs `scripts/evaluate_benchmark.py`.
4. `scripts/cluster/libero_eval.sbatch --backend <name> --suite libero_10 ...` does the same for LIBERO.

Each job writes its report after every episode. A RoboTwin run needs about 30 GB of GPU memory.

## Download matching checkpoints

Run in either prepared environment; both use `checkpoints/` in this checkout.
These commands pin model revisions, not floating branches. The FastWAM-family
loader downloads missing Wan VAE/text/tokenizer base assets on first load;
the training backbone initialization is skipped for released checkpoints.
The Wan text encoder and VAE are fetched from
`DiffSynth-Studio/Wan-Series-Converted-Safetensors`, which is published on
ModelScope; the Hugging Face Hub has no public repo of that name.
LingBot snapshots include their transformer/VAE/text encoder/tokenizer. Expect
large downloads and long first loads. Preserve HF caches for reproducibility.

```bash
export DIFFSYNTH_DOWNLOAD_SOURCE=modelscope
export DIFFSYNTH_MODEL_BASE_PATH="$PWD/checkpoints/wan_assets"
python - <<'PY'
from huggingface_hub import snapshot_download
snapshot_download('yuanty/fastwam', revision='8eaceeb24c3cc92ff2a9c9a9d266a4941b836705', local_dir='checkpoints/fastwam',
  allow_patterns=['libero_uncond_2cam224.pt', 'libero_uncond_2cam224_dataset_stats.json',
                  'robotwin_uncond_3cam_384.pt', 'robotwin_uncond_3cam_384_dataset_stats.json'])
snapshot_download('hustvl/FasterWAM', revision='6bf9471ced6919a15ab8fded89f7772f5060c44b', local_dir='checkpoints/fasterwam',
  allow_patterns=['libero/step_021700.pt', 'libero/dataset_stats.json', 'robotwin/step_029355.pt', 'robotwin/dataset_stats.json'])
snapshot_download('robbyant/lingbot-va-posttrain-libero-long', revision='0e89d1e753019988aba484e8da2dc0810e264d9f', local_dir='checkpoints/lingbot-libero')
snapshot_download('robbyant/lingbot-va-posttrain-robotwin', revision='8c9dea8abbc5c91cc9e18bc3264b8915083bbe70', local_dir='checkpoints/lingbot-robotwin')
PY
```

LIBERO-Plus uses the released LIBERO checkpoints/statistics unchanged. Do not
use LIBERO statistics with RoboTwin. LingBot normalization quantiles and channel
maps are embedded in the pinned config (not the family statistics JSON files).

## Six H100 smoke commands (pending execution)

Each command loads just one backend. bf16 is the default; unsupported CUDA bf16
fails explicitly with no precision fallback. `--dtype float16` is an explicit
experiment, not evidence that the released bf16 model is numerically compatible.
Activate `.venv-plus` and export its `LIBERO_CONFIG_PATH`/`MUJOCO_GL` first:

```bash
python scripts/evaluate_benchmark.py --benchmark libero_plus --backend fastwam --benchmark-root third_party/LIBERO-plus --checkpoint checkpoints/fastwam/libero_uncond_2cam224.pt --dataset-stats checkpoints/fastwam/libero_uncond_2cam224_dataset_stats.json --checkpoint-revision 8eaceeb24c3cc92ff2a9c9a9d266a4941b836705 --output results/plus-fastwam
python scripts/evaluate_benchmark.py --benchmark libero_plus --backend fasterwam --benchmark-root third_party/LIBERO-plus --checkpoint checkpoints/fasterwam/libero/step_021700.pt --dataset-stats checkpoints/fasterwam/libero/dataset_stats.json --checkpoint-revision 6bf9471ced6919a15ab8fded89f7772f5060c44b --output results/plus-fasterwam
python scripts/evaluate_benchmark.py --benchmark libero_plus --backend lingbot_va --benchmark-root third_party/LIBERO-plus --checkpoint checkpoints/lingbot-libero --checkpoint-revision 0e89d1e753019988aba484e8da2dc0810e264d9f --output results/plus-lingbot
```

These select variant index 0 of `libero_10`, one trial. For category filtering
use `--all-tasks --categories 'Language Instructions' --difficulty-levels 1 2`
with the exact category spelling from the official classification JSON. The API
also supports multiple explicit `task_ids`. Instructions are never rewritten.
Partial selections are labeled; category and overall counts include denominators.

Activate `.venv-robotwin` for the other three (start from the repository root):

```bash
python scripts/evaluate_benchmark.py --benchmark robotwin --backend fastwam --benchmark-root third_party/RoboTwin --task-name beat_block_hammer --checkpoint checkpoints/fastwam/robotwin_uncond_3cam_384.pt --dataset-stats checkpoints/fastwam/robotwin_uncond_3cam_384_dataset_stats.json --checkpoint-revision 8eaceeb24c3cc92ff2a9c9a9d266a4941b836705 --output results/robotwin-fastwam
python scripts/evaluate_benchmark.py --benchmark robotwin --backend fasterwam --benchmark-root third_party/RoboTwin --task-name beat_block_hammer --checkpoint checkpoints/fasterwam/robotwin/step_029355.pt --dataset-stats checkpoints/fasterwam/robotwin/dataset_stats.json --checkpoint-revision 6bf9471ced6919a15ab8fded89f7772f5060c44b --output results/robotwin-fasterwam
python scripts/evaluate_benchmark.py --benchmark robotwin --backend lingbot_va --benchmark-root third_party/RoboTwin --task-name beat_block_hammer --checkpoint checkpoints/lingbot-robotwin --checkpoint-revision 8c9dea8abbc5c91cc9e18bc3264b8915083bbe70 --output results/robotwin-lingbot
```

Defaults are `demo_clean`, one accepted episode and at most 1,000 seed attempts.
Use `--task-config demo_randomized` for domain randomization. All reports retain
the exact task config, chosen instruction, accepted/rejected seeds and reasons.

## Adapter contracts and fixture checks

`evaluate_libero_plus(adapter, LiberoPlusEvalConfig(...))` and
`evaluate_robotwin(adapter, RoboTwinEvalConfig(...))` consume **already-loaded**
adapters. Model loading stays in `scripts/evaluate_benchmark.py` (or your caller).
The common LIBERO loop owns reset/warmup/step/replan/termination and optional
`begin_episode` / `observe_step` hooks. RoboTwin additionally requires
`environment_action_type('robotwin')`; ordinary LIBERO does not.

| Backend | RoboTwin transform / dispatch |
| --- | --- |
| FastWAM | Head 320x256 above wrists 160x128 each; 14-D joint state, matching normalized stats, 14-D denormalized qpos; `first_frame` |
| Faster-WAM | Same joint transforms; native `one_pass_future_cache` and RoboTwin Hydra config |
| LingBot-VA | Head 256x320 and half-size wrists encoded separately; wrist latents above head, 30 channels mapped to 16 EE values; initial-XYZ addition and initial-rotation times relative-rotation, normalized XYZW quaternions |

Fast-family RoboTwin runs predict 32 actions and execute 28 (explicit plan
default; older FastWAM scripts may use a different prefix). LIBERO-Plus keeps
the existing LIBERO transforms and native dispatch. LingBot must execute its
complete RoboTwin chunks: skip the first 16-action latent frame, execute 16
commands initially, then 32; observe every executed step and sample every
fourth camera frame for cache updates. Do not apply observation-skipping or a
28-command prefix to LingBot. Separate full/half-resolution streaming VAE caches
share frozen VAE weights; no temporal feature cache is shared across resolutions.

Local fixed fixtures check camera ordering/layout, native z-score clipping,
denormalization, EE quaternion composition against SciPy, first-chunk handling,
cache sampling/order, task selection, seed bounds, and cleanup:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m pytest tests/test_benchmarks.py tests/test_fasterwam.py -q
```

Before claiming GPU parity, capture raw camera/state and normalized predicted
actions from the pinned upstream evaluator for each backend/benchmark pair.
Save those arrays as fixtures and run them through the corresponding adapter's
`prepare_observation` and `postprocess_actions`. Compare identical camera sizes,
dtype, statistics, initial pose, and chunk index; save shape/dtype/finite checks
and maximum/mean absolute differences (image interpolation can differ slightly
between libraries). Do not substitute synthetic images for checkpoint parity.
Run identical observation/prompt/seed/inference settings through upstream and
ported model inference as well; use the existing Faster-WAM parity procedure in
`fasterwam_validation.md` for its LIBERO input. Real captured fixture comparisons
for the new simulator combinations are **pending**, not claimed by CPU tests.

Result JSON includes benchmark and repository revisions, package versions,
effective model dtype/device, resolved selection/config, checkpoint identity,
statistics identity, inference metadata and synchronized per-call latency,
termination reasons and observed successes. CLI file checkpoints/statistics get
SHA-256 hashes; LingBot directory identity is its supplied pinned HF revision
and path. Library callers should provide checkpoint download revision/hashes in
`run_metadata` if their loader does not already attach them. Keep setup logs,
`pip freeze`, asset revision receipt, smoke JSONs and any failures as the handoff.
Full suites, research interventions, automatic pushes and PR creation are not
part of these smoke commands.
