#!/bin/bash
# Builds the CuRobo v0.7.8 wheel with the node's CUDA 13.3 nvcc and keeps it in $HOME/wheels if a planner test passes.
#   sbatch scripts/cluster/curobo_build.sbatch [--variant a|b]   # a: torch 2.7.1+cu128, b: torch 2.9.1+cu130
#SBATCH -p general
#SBATCH --gres=gpu:l40:1
#SBATCH -c 16
#SBATCH --mem=64G
#SBATCH -t 1:30:00
#SBATCH -J curobo-build

set -euo pipefail
VARIANT=a
while [ $# -gt 0 ]; do case "$1" in --variant) VARIANT="$2"; shift 2 ;; *) shift ;; esac; done
HOME_DIR="$HOME"; ls "$HOME_DIR" >/dev/null 2>&1 || HOME_DIR="${HOME/#\/nethome\//\/nethome-instruction\/}"
OUT="$HOME_DIR/kv_results/curobo-build-$VARIANT-${SLURM_JOB_ID:-local}"; mkdir -p "$OUT"
SCRATCH="/dev/shm/${USER}-curobo-${SLURM_JOB_ID:-local}"; mkdir -p "$SCRATCH"
cleanup() { rm -rf "$SCRATCH"; }; trap cleanup EXIT; trap 'cleanup; exit 143' TERM INT
export UV_CACHE_DIR="$SCRATCH/uv" UV_PYTHON_INSTALL_DIR="$SCRATCH/py" XDG_CACHE_HOME="$SCRATCH/xdg" WARP_CACHE_PATH="$SCRATCH/warp"
export PYTHONUNBUFFERED=1 CUDA_HOME=/usr/local/cuda-13.3 TORCH_CUDA_ARCH_LIST="8.9;9.0" MAX_JOBS=12
export PATH="$CUDA_HOME/bin:$PATH"
UV="$HOME_DIR/.local/bin/uv"
echo "host=$(hostname) variant=$VARIANT"; nvcc --version | tail -2; gcc --version | head -1

"$UV" venv --python 3.10 "$SCRATCH/venv" -q && . "$SCRATCH/venv/bin/activate"
if [ "$VARIANT" = a ]; then
  "$UV" pip install -q torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu128
else
  "$UV" pip install -q torch==2.9.1 torchvision==0.24.1 --index-url https://download.pytorch.org/whl/cu130
fi
"$UV" pip install -q pip "setuptools==69.5.1" wheel setuptools_scm ninja "numpy==1.26.4" "warp-lang==1.12.0"
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda, torch.cuda.is_available())"

# nvcc 13 rejects a dependent type in torch's List_inl.h; spell the same type directly (build venv only).
LIST_INL=$(python -c "import os, torch; print(os.path.join(os.path.dirname(torch.__file__), 'include/ATen/core/List_inl.h'))")
sed -i 's/static_cast<typename decltype(impl_->list)::difference_type>(pos)/static_cast<std::ptrdiff_t>(pos)/' "$LIST_INL"
echo "List_inl.h patched lines: $(grep -c 'static_cast<std::ptrdiff_t>(pos)' "$LIST_INL")"
git clone -q --branch v0.7.8 --depth 1 https://github.com/NVlabs/curobo.git "$SCRATCH/curobo"
WHEELS="$SCRATCH/wheels"; mkdir -p "$WHEELS"
T0=$(date +%s)
cd "$SCRATCH/curobo"
python - "$VARIANT" "$WHEELS" <<'PY'
import runpy, sys
variant, wheels = sys.argv[1], sys.argv[2]
if variant == "a":
    import torch.utils.cpp_extension as ce
    ce._check_cuda_version = lambda *args, **kwargs: print("bypassed torch CUDA major-version check")
sys.argv = ["setup.py", "-q", "bdist_wheel", "-d", wheels]
runpy.run_path("setup.py", run_name="__main__")
PY
echo "build finished in $(( $(date +%s) - T0 ))s"; ls -la "$WHEELS"
cd "$SCRATCH"
"$UV" pip install -q "numpy==1.26.4" "warp-lang==1.12.0" "$WHEELS"/*.whl
python - <<'PY' 2>&1 | tee "$OUT/planner_test.txt" | tail -20
import time, torch
from curobo.types.base import TensorDeviceType
from curobo.types.math import Pose
from curobo.types.robot import JointState
from curobo.wrap.reacher.motion_gen import MotionGen, MotionGenConfig, MotionGenPlanConfig
t = time.time()
cfg = MotionGenConfig.load_from_robot_config("franka.yml", "collision_table.yml", interpolation_dt=0.01)
mg = MotionGen(cfg)
mg.warmup()
print(f"warmup ok in {time.time() - t:.1f}s")
# CuRobo's documented getting-started check: goal = FK(retract), start = retract + 0.3 rad.
retract = mg.get_retract_config()
state = mg.rollout_fn.compute_kinematics(JointState.from_position(retract.view(1, -1)))
goal = Pose(state.ee_pos_seq.squeeze(), quaternion=state.ee_quat_seq.squeeze())
start = JointState.from_position(retract.view(1, -1) + 0.3)
for attempt in range(3):
    t = time.time()
    result = mg.plan_single(start, goal, MotionGenPlanConfig(max_attempts=4))
    print("PLAN success:", bool(result.success.item()), "status:", result.status, f"{time.time() - t:.2f}s",
          "steps:", None if result.interpolated_plan is None else tuple(result.interpolated_plan.position.shape))
    if bool(result.success.item()):
        break
PY
if grep -q "PLAN success: True" "$OUT/planner_test.txt"; then
  mkdir -p "$HOME_DIR/wheels/curobo-v0.7.8-$VARIANT" && cp "$WHEELS"/*.whl "$HOME_DIR/wheels/curobo-v0.7.8-$VARIANT/"
  echo "PLANNER OK; wheel saved to $HOME_DIR/wheels/curobo-v0.7.8-$VARIANT"
else
  echo "PLANNER FAILED; wheel not saved"
fi
