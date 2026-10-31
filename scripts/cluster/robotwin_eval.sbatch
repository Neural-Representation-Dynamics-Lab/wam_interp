#!/bin/bash
# RoboTwin 2.0 evaluation on one Slurm GPU node; needs the CuRobo wheel from curobo_build.sbatch.
#   sbatch scripts/cluster/robotwin_eval.sbatch --backend fasterwam --task-names adjust_bottle --episodes 10 --output-name rt
#SBATCH -p general
#SBATCH --gres=gpu:l40:1
#SBATCH -c 16
#SBATCH --mem=160G
#SBATCH -t 3:00:00
#SBATCH -J robotwin

set -euo pipefail
REPO="${SLURM_SUBMIT_DIR:-$PWD}"
[ -d "$REPO/src" ] || REPO="${REPO/#\/nethome\//\/nethome-instruction\/}"
HOME_DIR="$HOME"; ls "$HOME_DIR" >/dev/null 2>&1 || HOME_DIR="${HOME/#\/nethome\//\/nethome-instruction\/}"

BACKEND=fasterwam; OUTPUT_NAME=robotwin; TASK_CONFIG=demo_clean; CUROBO_VARIANT=a; ARGS=()
while [ $# -gt 0 ]; do
  case "$1" in
    --backend) BACKEND="$2"; shift 2 ;;
    --curobo-variant) CUROBO_VARIANT="$2"; shift 2 ;;
    --output-name) OUTPUT_NAME="$2"; shift 2 ;;
    --task-config) TASK_CONFIG="$2"; ARGS+=("$1" "$2"); shift 2 ;;
    *) ARGS+=("$1"); shift ;;
  esac
done
RESULTS="$HOME_DIR/kv_results/${OUTPUT_NAME}-${SLURM_JOB_ID:-local}"; mkdir -p "$RESULTS"
SCRATCH="/dev/shm/${USER}-rt-${SLURM_JOB_ID:-local}"; mkdir -p "$SCRATCH"
cleanup() { rm -rf "$SCRATCH"; }; trap cleanup EXIT; trap 'cleanup; exit 143' TERM INT

export UV_CACHE_DIR="$SCRATCH/uv" UV_PYTHON_INSTALL_DIR="$SCRATCH/py" HF_HOME="$SCRATCH/hf" XDG_CACHE_HOME="$SCRATCH/xdg"
export WARP_CACHE_PATH="$SCRATCH/warp" TORCH_EXTENSIONS_DIR="$SCRATCH/torch_ext" MODELSCOPE_CACHE="$SCRATCH/modelscope"
export DIFFSYNTH_DOWNLOAD_SOURCE=modelscope DIFFSYNTH_MODEL_BASE_PATH="$SCRATCH/wan_assets"
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false LANG=C.UTF-8 LC_ALL=C.UTF-8
UV="$HOME_DIR/.local/bin/uv"
echo "host=$(hostname) job=${SLURM_JOB_ID:-local} backend=$BACKEND results=$RESULTS"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv | tee "$RESULTS/nvidia-smi.csv"

cp -r "$REPO" "$SCRATCH/wam_interp" && cd "$SCRATCH/wam_interp"
{ git rev-parse HEAD; git status --porcelain; } > "$RESULTS/source_state.txt" 2>&1 || true
git diff HEAD > "$RESULTS/source_diff.patch" 2>/dev/null || true

T0=$(date +%s)
"$UV" venv --python 3.10 "$SCRATCH/venv" -q && . "$SCRATCH/venv/bin/activate"
# Pins the legacy RoboTwin stack needs: sapien imports pkg_resources, mplib needs numpy<2, warp 1.13 removed warp.torch.
if [ "$CUROBO_VARIANT" = b ]; then TORCH=2.9.1; TORCHVISION=0.24.1; TORCH_INDEX=cu130; else TORCH=2.7.1; TORCHVISION=0.22.1; TORCH_INDEX=cu128; fi
cat > "$SCRATCH/constraints.txt" <<EOF
torch==$TORCH
torchvision==$TORCHVISION
numpy==1.26.4
setuptools==69.5.1
opencv-python==4.10.0.84
opencv-python-headless==4.10.0.84
warp-lang==1.12.0
zarr==2.18.3
numcodecs==0.13.1
EOF
# OpenWAM needs huggingface-hub>=1.5; the other backends use the repo's 0.36.2 pin.
[ "$BACKEND" = openwam ] || echo "huggingface-hub==0.36.2" >> "$SCRATCH/constraints.txt"
export UV_CONSTRAINT="$SCRATCH/constraints.txt" PIP_CONSTRAINT="$SCRATCH/constraints.txt"
# The PyTorch index has no setuptools 69.5.1, so install torch without the constraints.
UV_CONSTRAINT= PIP_CONSTRAINT= "$UV" pip install -q "torch==$TORCH" "torchvision==$TORCHVISION" --index-url "https://download.pytorch.org/whl/$TORCH_INDEX"
"$UV" pip install -q pip setuptools==69.5.1 wheel setuptools_scm ninja
case "$BACKEND" in
  lingbot_va) EXTRA=lingbot_va ;;
  longwam) EXTRA=longwam; export DIFFSYNTH_DOWNLOAD_SOURCE=huggingface ;;   # Long-WAM base assets come from HF
  openwam) EXTRA=openwam ;;
  *) EXTRA=fasterwam ;;
esac
"$UV" pip install -q -e ".[${EXTRA},robotwin]" pytest

RT="$SCRATCH/RoboTwin"
git clone -q --filter=blob:none https://github.com/RoboTwin-Platform/RoboTwin.git "$RT"
git -C "$RT" checkout -q --detach c3ddfa8b97d5519efa828b075999bd0006778e5e
python - "$RT" "$SCRATCH/requirements.txt" <<'PY'
import sys
from pathlib import Path
sys.path.insert(0, "scripts")
from prepare_robotwin import filter_requirements
Path(sys.argv[2]).write_text(filter_requirements((Path(sys.argv[1]) / "script/requirements.txt").read_text()))
PY
[ "$BACKEND" = openwam ] && sed -i '/^huggingface-hub==/d' "$SCRATCH/requirements.txt"
"$UV" pip install -q -r "$SCRATCH/requirements.txt"
# The two patches from upstream script/_install.sh.
SITE=$(python -c "import site; print(site.getsitepackages()[0])")
sed -i -E 's/("r")(\))( as)/\1, encoding="utf-8") as/g' "$SITE/sapien/wrapper/urdf_loader.py"
sed -i -E 's/(if np.linalg.norm\(delta_twist\) < 1e-4 )(or collide )(or not within_joint_limit:)/\1\3/g' "$SITE/mplib/planner.py"
(grep -c 'encoding="utf-8"' "$SITE/sapien/wrapper/urdf_loader.py" || true) | sed 's/^/sapien patch lines: /'
(grep -c "or collide or not within_joint_limit" "$SITE/mplib/planner.py" || true) | sed 's/^/mplib unpatched lines (want 0): /'

WHEELS="$HOME_DIR/wheels/curobo-v0.7.8-$CUROBO_VARIANT"
ls "$WHEELS"/*.whl >/dev/null 2>&1 || { echo "No CuRobo wheel in $WHEELS; run curobo_build.sbatch --variant $CUROBO_VARIANT first"; exit 3; }
"$UV" pip install -q "$WHEELS"/*.whl   # with its runtime deps; constraints keep torch/numpy/warp
"$UV" pip install -q warp-lang==1.12.0 setuptools==69.5.1
"$UV" pip freeze > "$RESULTS/pip_freeze.txt"
python -c "import curobo, sapien, mplib, warp; print('imports ok', curobo.__version__ if hasattr(curobo, '__version__') else '')"
echo "env ready in $(( $(date +%s) - T0 ))s"

# Assets at the HF revision matching the pinned code.
ZIPS='["embodiments.zip", "objects.zip"]'
[ "$TASK_CONFIG" = demo_randomized ] && ZIPS='["embodiments.zip", "objects.zip", "background_texture.zip"]'
python - "$RT/assets" "$ZIPS" <<'PY'
import json, sys, zipfile, time
from pathlib import Path
from huggingface_hub import snapshot_download
target, zips = Path(sys.argv[1]), json.loads(sys.argv[2])
t = time.time()
local = snapshot_download("TianxingChen/RoboTwin2.0", repo_type="dataset", allow_patterns=zips,
                          revision="9dc9299c163db059931898a9f0852098a61155a1")
for name in zips:
    with zipfile.ZipFile(Path(local) / name) as z:
        z.extractall(target)
    (Path(local) / name).unlink(missing_ok=True)
print(f"assets {zips} ready in {time.time() - t:.0f}s")
PY
(cd "$RT" && python script/update_embodiment_config_path.py < /dev/null | tail -3)

CK="$SCRATCH/checkpoints"
case "$BACKEND" in
  fasterwam)
    python -c "from huggingface_hub import snapshot_download as s; s('hustvl/FasterWAM', revision='6bf9471ced6919a15ab8fded89f7772f5060c44b', local_dir='$CK/fasterwam', allow_patterns=['robotwin/step_029355.pt', 'robotwin/dataset_stats.json'])"
    MODEL_ARGS=(--checkpoint "$CK/fasterwam/robotwin/step_029355.pt" --dataset-stats "$CK/fasterwam/robotwin/dataset_stats.json" --checkpoint-revision 6bf9471ced6919a15ab8fded89f7772f5060c44b) ;;
  fastwam)
    python -c "from huggingface_hub import snapshot_download as s; s('yuanty/fastwam', revision='8eaceeb24c3cc92ff2a9c9a9d266a4941b836705', local_dir='$CK/fastwam', allow_patterns=['robotwin_uncond_3cam_384.pt', 'robotwin_uncond_3cam_384_dataset_stats.json'])"
    MODEL_ARGS=(--checkpoint "$CK/fastwam/robotwin_uncond_3cam_384.pt" --dataset-stats "$CK/fastwam/robotwin_uncond_3cam_384_dataset_stats.json" --checkpoint-revision 8eaceeb24c3cc92ff2a9c9a9d266a4941b836705) ;;
  lingbot_va)
    python -c "from huggingface_hub import snapshot_download as s; s('robbyant/lingbot-va-posttrain-robotwin', revision='8c9dea8abbc5c91cc9e18bc3264b8915083bbe70', local_dir='$CK/lingbot')"
    MODEL_ARGS=(--checkpoint "$CK/lingbot" --checkpoint-revision 8c9dea8abbc5c91cc9e18bc3264b8915083bbe70) ;;
  longwam)
    # Bundle files checked against their pinned SHA-256; Wan base assets staged at pinned revisions.
    python - "$CK/longwam" "$DIFFSYNTH_MODEL_BASE_PATH" <<'PY'
import hashlib, sys
from pathlib import Path
from huggingface_hub import hf_hub_download
from wam_interp.models.longwam.loader import BASE_ASSETS, RELEASED_BUNDLES
spec, bundle, models = RELEASED_BUNDLES["robotwin2_idm"], Path(sys.argv[1]), Path(sys.argv[2])
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
    REV=$(python -c "from wam_interp.models.longwam.loader import RELEASED_BUNDLES as b; print(b['robotwin2_idm']['revision'])")
    MODEL_ARGS=(--checkpoint "$CK/longwam/model.pt" --dataset-stats "$CK/longwam/dataset_stats.json" --checkpoint-revision "$REV") ;;
  openwam)
    REV=04b96af53eeeb111c64631f822efcbb82f4b186e
    python -c "from huggingface_hub import snapshot_download as s; s('OpenWAM/OpenWAM-Alpha-Sim-RoboTwin-Full', revision='$REV', local_dir='$CK/openwam')"
    MODEL_ARGS=(--checkpoint "$CK/openwam" --checkpoint-revision "$REV") ;;
  *) echo "unknown backend $BACKEND"; exit 2 ;;
esac

echo "starting evaluation at $(( $(date +%s) - T0 ))s"
python scripts/evaluate_benchmark.py --benchmark robotwin --backend "$BACKEND" --benchmark-root "$RT" \
  "${MODEL_ARGS[@]}" --output "$RESULTS" "${ARGS[@]}"
echo "done in $(( $(date +%s) - T0 ))s; results in $RESULTS"
