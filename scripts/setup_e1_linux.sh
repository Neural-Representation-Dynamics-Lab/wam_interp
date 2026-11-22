#!/usr/bin/env bash
# Run from the transported repository root on the Linux GPU machine.
set -euo pipefail
cd "$(dirname "$0")/.."
PYTHON_BIN="${E1_PYTHON:-python3.10}"
"$PYTHON_BIN" -c 'import sys; assert sys.version_info[:2] == (3, 10), "Use Python 3.10 for the pinned LIBERO environment"'
"$PYTHON_BIN" -m venv .venv-e1
source .venv-e1/bin/activate
python -m pip install --upgrade pip
python -m pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -e '.[fastwam,libero,research]'
mkdir -p third_party checkpoints
if [ ! -d third_party/LIBERO ]; then
    git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git third_party/LIBERO
fi
if [ -n "$(git -C third_party/LIBERO status --porcelain)" ]; then
    printf '%s\n' 'Existing LIBERO checkout has edits; preserve them before setup.' >&2
    exit 1
fi
git -C third_party/LIBERO fetch origin 8f1084e3132a39270c3a13ebe37270a43ece2a01
git -C third_party/LIBERO checkout --detach 8f1084e3132a39270c3a13ebe37270a43ece2a01
python -m pip install -e third_party/LIBERO --no-deps --config-settings editable_mode=compat
export LIBERO_CONFIG_PATH="$PWD/results/libero_config"
mkdir -p "$LIBERO_CONFIG_PATH"
python - <<'PY'
import os
from pathlib import Path
import yaml
root = Path('third_party/LIBERO/libero/libero').resolve()
config = dict(benchmark_root=str(root), bddl_files=str(root/'bddl_files'),
              init_states=str(root/'init_files'), datasets=str(root/'datasets'), assets=str(root/'assets'))
Path(os.environ['LIBERO_CONFIG_PATH'], 'config.yaml').write_text(yaml.safe_dump(config))
from huggingface_hub import snapshot_download
snapshot_download('yuanty/fastwam', revision='8eaceeb24c3cc92ff2a9c9a9d266a4941b836705',
    local_dir='checkpoints/fastwam', allow_patterns=['libero_optional_idm_2cam224.pt',
    'libero_optional_idm_2cam224_dataset_stats.json'])
# Official assets, pinned independently of the optional DiffSynth mirror.
snapshot_download('Wan-AI/Wan2.2-TI2V-5B', revision='921dbaf3f1674a56f47e83fb80a34bac8a8f203e',
    local_dir='checkpoints/Wan-AI/Wan2.2-TI2V-5B',
    allow_patterns=['Wan2.2_VAE.pth', 'models_t5_umt5-xxl-enc-bf16.pth'])
snapshot_download('Wan-AI/Wan2.1-T2V-1.3B', revision='37ec512624d61f7aa208f7ea8140a131f93afc9a',
    local_dir='checkpoints/Wan-AI/Wan2.1-T2V-1.3B', allow_patterns=['google/umt5-xxl/*'])
PY
python -m pytest tests/test_e1.py tests/test_e1_libero.py -q
python -m pip freeze > results/e1_environment_freeze.txt
printf '%s\n' 'Setup complete. Export MUJOCO_GL=egl, PYOPENGL_PLATFORM=egl and LIBERO_CONFIG_PATH before running.'
