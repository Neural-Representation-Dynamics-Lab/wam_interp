#!/bin/bash
# LIBERO evaluation on one Slurm GPU node. Results go to $HOME/kv_results/<output-name>-<jobid>.
#   sbatch scripts/cluster/libero_eval.sbatch --backend fastwam --suite libero_10 --all-tasks --episodes 50 --output-name lb10
#SBATCH -p general
#SBATCH --gres=gpu:l40:1
#SBATCH -c 16
#SBATCH --mem=160G
#SBATCH -t 4:00:00
#SBATCH -J libero

set -euo pipefail
REPO="${SLURM_SUBMIT_DIR:-$PWD}"
[ -d "$REPO/src" ] || REPO="${REPO/#\/nethome\//\/nethome-instruction\/}"
HOME_DIR="$HOME"; ls "$HOME_DIR" >/dev/null 2>&1 || HOME_DIR="${HOME/#\/nethome\//\/nethome-instruction\/}"

BACKEND=fastwam; OUTPUT_NAME=libero; FASTWAM_TASK=libero_uncond_2cam224_1e-4; ARGS=()
while [ $# -gt 0 ]; do
  case "$1" in
    --backend) BACKEND="$2"; shift 2 ;;
    --output-name) OUTPUT_NAME="$2"; shift 2 ;;
    --fastwam-task) FASTWAM_TASK="$2"; ARGS+=("$1" "$2"); shift 2 ;;
    *) ARGS+=("$1"); shift ;;
  esac
done
RESULTS="$HOME_DIR/kv_results/${OUTPUT_NAME}-${SLURM_JOB_ID:-local}"; mkdir -p "$RESULTS"
SCRATCH="/dev/shm/${USER}-lb-${SLURM_JOB_ID:-local}"; mkdir -p "$SCRATCH"
cleanup() { rm -rf "$SCRATCH"; }; trap cleanup EXIT; trap 'cleanup; exit 143' TERM INT

export UV_CACHE_DIR="$SCRATCH/uv" UV_PYTHON_INSTALL_DIR="$SCRATCH/py" HF_HOME="$SCRATCH/hf" XDG_CACHE_HOME="$SCRATCH/xdg"
export MODELSCOPE_CACHE="$SCRATCH/modelscope" TORCH_EXTENSIONS_DIR="$SCRATCH/torch_ext"
export TORCHINDUCTOR_CACHE_DIR="$SCRATCH/inductor" TRITON_CACHE_DIR="$SCRATCH/triton"
export DIFFSYNTH_DOWNLOAD_SOURCE=modelscope DIFFSYNTH_MODEL_BASE_PATH="$SCRATCH/wan_assets"
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl LIBERO_CONFIG_PATH="$SCRATCH/libero_config"
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
UV="$HOME_DIR/.local/bin/uv"
echo "host=$(hostname) job=${SLURM_JOB_ID:-local} backend=$BACKEND results=$RESULTS"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv | tee "$RESULTS/nvidia-smi.csv"

cp -r "$REPO" "$SCRATCH/wam_interp" && cd "$SCRATCH/wam_interp"
{ git rev-parse HEAD; git status --porcelain; } > "$RESULTS/source_state.txt" 2>&1 || true
git diff HEAD > "$RESULTS/source_diff.patch" 2>/dev/null || true

T0=$(date +%s)
case "$BACKEND" in
  lingbot_va) EXTRA=lingbot_va ;;
  longwam) EXTRA=longwam; export DIFFSYNTH_DOWNLOAD_SOURCE=huggingface ;;
  openwam) EXTRA=openwam ;;
  fastwam|fasterwam) EXTRA=fasterwam ;;
  *) echo "unknown backend $BACKEND"; exit 2 ;;
esac
"$UV" venv --python 3.10 "$SCRATCH/venv" -q && . "$SCRATCH/venv/bin/activate"
"$UV" pip install -q torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu128
"$UV" pip install -q -e ".[${EXTRA},libero]" pytest

git clone -q --depth 1 --no-checkout --branch master https://github.com/Lifelong-Robot-Learning/LIBERO.git "$SCRATCH/LIBERO"
git -C "$SCRATCH/LIBERO" fetch -q --depth 1 origin 8f1084e3132a39270c3a13ebe37270a43ece2a01
git -C "$SCRATCH/LIBERO" checkout -q --detach 8f1084e3132a39270c3a13ebe37270a43ece2a01
"$UV" pip install -q -e "$SCRATCH/LIBERO" --no-deps --config-settings editable_mode=compat
mkdir -p "$LIBERO_CONFIG_PATH" "$SCRATCH/libero_datasets"
LR="$SCRATCH/LIBERO/libero/libero"
cat > "$LIBERO_CONFIG_PATH/config.yaml" <<EOF
benchmark_root: $LR
bddl_files: $LR/bddl_files
init_states: $LR/init_files
datasets: $SCRATCH/libero_datasets
assets: $LR/assets
EOF
"$UV" pip freeze > "$RESULTS/pip_freeze.txt"
python -c "import mujoco, robosuite, libero; print('mujoco', mujoco.__version__, 'robosuite', robosuite.__version__)"
echo "env ready in $(( $(date +%s) - T0 ))s"

CK="$SCRATCH/checkpoints"
case "$BACKEND" in
  fastwam)
    REV=8eaceeb24c3cc92ff2a9c9a9d266a4941b836705
    case "$FASTWAM_TASK" in
      libero_uncond_2cam224_1e-4) NAME=libero_uncond_2cam224 ;;
      libero_optional_idm_2cam224_1e-4) NAME=libero_optional_idm_2cam224 ;;
      *) echo "unknown --fastwam-task $FASTWAM_TASK"; exit 2 ;;
    esac
    python -c "from huggingface_hub import snapshot_download as s; s('yuanty/fastwam', revision='$REV', local_dir='$CK/fastwam', allow_patterns=['$NAME.pt', '${NAME}_dataset_stats.json'])"
    MODEL_ARGS=(--checkpoint "$CK/fastwam/$NAME.pt" --dataset-stats "$CK/fastwam/${NAME}_dataset_stats.json" --checkpoint-revision "$REV") ;;
  fasterwam)
    REV=6bf9471ced6919a15ab8fded89f7772f5060c44b
    python -c "from huggingface_hub import snapshot_download as s; s('hustvl/FasterWAM', revision='$REV', local_dir='$CK/fasterwam', allow_patterns=['libero/step_021700.pt', 'libero/dataset_stats.json'])"
    MODEL_ARGS=(--checkpoint "$CK/fasterwam/libero/step_021700.pt" --dataset-stats "$CK/fasterwam/libero/dataset_stats.json" --checkpoint-revision "$REV") ;;
  lingbot_va)
    REV=0e89d1e753019988aba484e8da2dc0810e264d9f
    python -c "from huggingface_hub import snapshot_download as s; s('robbyant/lingbot-va-posttrain-libero-long', revision='$REV', local_dir='$CK/lingbot', allow_patterns=['transformer/*', 'vae/*', 'text_encoder/*', 'tokenizer/*'])"
    MODEL_ARGS=(--checkpoint "$CK/lingbot" --checkpoint-revision "$REV") ;;
  longwam)
    # Bundle files checked against their pinned SHA-256; Wan base assets staged at pinned revisions.
    python - "$CK/longwam" "$DIFFSYNTH_MODEL_BASE_PATH" <<'PY'
import hashlib, sys
from pathlib import Path
from huggingface_hub import hf_hub_download
from wam_interp.models.longwam.loader import BASE_ASSETS, RELEASED_BUNDLES
spec, bundle, models = RELEASED_BUNDLES["libero_idm"], Path(sys.argv[1]), Path(sys.argv[2])
for name in ("config.yaml", "dataset_stats.json", "model.pt"):
    path = Path(hf_hub_download(spec["repo_id"], name, revision=spec["revision"], local_dir=bundle))
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    if digest.hexdigest() != spec["sha256"][name]:
        raise SystemExit(f"{name}: SHA-256 {digest.hexdigest()} != pinned {spec['sha256'][name]}")
for repo_id, asset in BASE_ASSETS.items():
    for name in asset["files"]:
        hf_hub_download(repo_id, name, revision=asset["revision"], local_dir=models / repo_id)
print("Long-WAM bundle verified and base assets staged")
PY
    export DIFFSYNTH_SKIP_DOWNLOAD=true
    REV=$(python -c "from wam_interp.models.longwam.loader import RELEASED_BUNDLES as b; print(b['libero_idm']['revision'])")
    MODEL_ARGS=(--checkpoint "$CK/longwam/model.pt" --dataset-stats "$CK/longwam/dataset_stats.json" --checkpoint-revision "$REV") ;;
  openwam)
    REV=110b07e4d5581c81520a77419cbb0e9371df231b
    python -c "from huggingface_hub import snapshot_download as s; s('OpenWAM/OpenWAM-Alpha-Sim-LIBERO', revision='$REV', local_dir='$CK/openwam')"
    MODEL_ARGS=(--checkpoint "$CK/openwam" --checkpoint-revision "$REV") ;;
esac

echo "starting evaluation at $(( $(date +%s) - T0 ))s"
python scripts/evaluate_benchmark.py --benchmark libero --backend "$BACKEND" --benchmark-root "$SCRATCH/LIBERO" \
  "${MODEL_ARGS[@]}" --output "$RESULTS" "${ARGS[@]}"
echo "done in $(( $(date +%s) - T0 ))s; results in $RESULTS"
