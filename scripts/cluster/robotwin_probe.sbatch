#!/bin/bash
# Checks that a GPU node can run RoboTwin: NVIDIA Vulkan userspace, build toolchain, one ray-traced SAPIEN frame.
#   sbatch --gres=gpu:l40s:1 scripts/cluster/robotwin_probe.sbatch
#SBATCH -p general
#SBATCH --gres=gpu:l40s:1
#SBATCH -c 8
#SBATCH --mem=48G
#SBATCH -t 0:45:00
#SBATCH -J robotwin-probe

set -u
HOME_DIR="$HOME"; ls "$HOME_DIR" >/dev/null 2>&1 || HOME_DIR="${HOME/#\/nethome\//\/nethome-instruction\/}"
OUT="$HOME_DIR/kv_results/robotwin-probe-${SLURM_JOB_ID:-local}"; mkdir -p "$OUT"
SCRATCH="/dev/shm/${USER}-rtprobe-${SLURM_JOB_ID:-local}"; mkdir -p "$SCRATCH"
cleanup() { rm -rf "$SCRATCH"; }; trap cleanup EXIT; trap 'cleanup; exit 143' TERM INT
export UV_CACHE_DIR="$SCRATCH/uv" UV_PYTHON_INSTALL_DIR="$SCRATCH/py" XDG_CACHE_HOME="$SCRATCH/xdg" PYTHONUNBUFFERED=1
REPORT="$OUT/report.txt"
section() { echo; echo "## $*"; }
{
section "node"; hostname; cat /etc/os-release 2>/dev/null | head -2; uname -r
section "gpu"; nvidia-smi --query-gpu=name,driver_version,compute_mode,pci.bus_id --format=csv
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"; ls -l /dev/nvidia* 2>&1 | head
section "driver graphics userspace (ldconfig)"
ldconfig -p | grep -E 'libGLX_nvidia|libEGL_nvidia|libnvidia-(glcore|glvkspirv|gpucomp|eglcore|glsi|rtcore|cbl)|libGLdispatch|libvulkan\.so|libEGL\.so|libGLX\.so' || echo "none found"
section "driver graphics userspace (file search)"
for d in /usr/lib64 /usr/lib/x86_64-linux-gnu /usr/lib /lib64; do ls "$d"/libGLX_nvidia.so* "$d"/libEGL_nvidia.so* "$d"/libnvidia-glvkspirv.so* "$d"/libnvidia-gpucomp.so* "$d"/libnvidia-rtcore.so* 2>/dev/null; done
section "vulkan / glvnd config"
for d in /usr/share/vulkan/icd.d /etc/vulkan/icd.d /usr/share/vulkan/implicit_layer.d /etc/vulkan/implicit_layer.d /usr/share/glvnd/egl_vendor.d /etc/glvnd/egl_vendor.d; do echo "$d:"; ls "$d" 2>&1 | sed 's/^/  /'; done
cat /usr/share/vulkan/icd.d/nvidia_icd.json /etc/vulkan/icd.d/nvidia_icd.json 2>/dev/null
section "toolchain"
ldd --version 2>&1 | head -1; gcc --version 2>&1 | head -1; g++ --version 2>&1 | head -1
which nvcc 2>&1; ls -d /usr/local/cuda* /opt/cuda* /usr/lib/cuda* /opt/nvidia/hpc_sdk 2>/dev/null
for init in /etc/profile.d/lmod.sh /etc/profile.d/modules.sh /usr/share/lmod/lmod/init/bash; do [ -f "$init" ] && . "$init" 2>/dev/null && echo "sourced $init"; done
type module >/dev/null 2>&1 && { module avail cuda 2>&1 | head -20; module avail 2>&1 | grep -i -E "cuda|nvhpc|gcc" | head -20; } || echo "no environment modules"
} > "$REPORT" 2>&1

UV="$HOME_DIR/.local/bin/uv"
"$UV" venv --python 3.10 "$SCRATCH/venv" -q && . "$SCRATCH/venv/bin/activate"
"$UV" pip install -q "sapien==3.0.0b1" "numpy==1.26.4" "setuptools<82" "opencv-python-headless==4.10.0.84" pillow >> "$REPORT" 2>&1
SITE=$(python -c "import site; print(site.getsitepackages()[0])")
if ! ls /usr/share/glvnd/egl_vendor.d/*nvidia* /etc/glvnd/egl_vendor.d/*nvidia* >/dev/null 2>&1; then
  export __EGL_VENDOR_LIBRARY_FILENAMES="$SITE/sapien/vulkan_library/10_nvidia.json"   # SAPIEN 3.0.0b1 import crash workaround
fi

cat > "$SCRATCH/render.py" <<'PY'
import sys, time, traceback
import numpy as np
from PIL import Image
shader, out = sys.argv[1], sys.argv[2]
try:
    import sapien.core as sapien
    from sapien.render import set_global_config
    try:
        print("device summary:", sapien.render.get_device_summary())
    except Exception as exc:
        print("device summary unavailable:", exc)
    t0 = time.time()
    engine = sapien.Engine()
    set_global_config(max_num_materials=50000, max_num_textures=50000)
    renderer = sapien.SapienRenderer()
    engine.set_renderer(renderer)
    if shader == "rt":   # exactly RoboTwin's setup_scene (envs/_base_task.py L206-217 at c3ddfa8)
        sapien.render.set_camera_shader_dir("rt")
        sapien.render.set_ray_tracing_samples_per_pixel(32)
        sapien.render.set_ray_tracing_path_depth(8)
        sapien.render.set_ray_tracing_denoiser("oidn")
    scene = engine.create_scene(sapien.SceneConfig())
    scene.set_timestep(1 / 250)
    scene.add_ground(0)
    scene.set_ambient_light([0.5, 0.5, 0.5])
    scene.add_directional_light([0, 0.5, -1], [0.5, 0.5, 0.5], shadow=True)
    builder = scene.create_actor_builder()
    builder.add_box_collision(half_size=[0.05] * 3)
    builder.add_box_visual(half_size=[0.05] * 3, material=[1.0, 0.0, 0.0])
    box = builder.build(name="box")
    box.set_pose(sapien.Pose([0, 0, 0.05]))
    camera = scene.add_camera(name="head", width=320, height=240, fovy=np.deg2rad(37), near=0.1, far=100)
    camera.entity.set_pose(sapien.Pose([-0.6, 0, 0.3], [0.9659, 0, 0.2588, 0]))
    setup = time.time() - t0
    times = []
    for _ in range(5):
        t = time.time(); scene.step(); scene.update_render(); camera.take_picture()
        rgba = camera.get_picture("Color"); times.append(time.time() - t)
    image = (np.clip(rgba[..., :3], 0, 1) * 255).astype(np.uint8)
    Image.fromarray(image).save(out)
    print(f"OK shader={shader} setup={setup:.2f}s frame_times={[round(x, 3) for x in times]} mean_rgb={image.reshape(-1, 3).mean(0).round(1).tolist()}")
except Exception:
    print(f"FAIL shader={shader}"); traceback.print_exc()
PY
{
section "sapien render: raster (default shader)"
VK_LOADER_DEBUG=error timeout 300 python "$SCRATCH/render.py" default "$OUT/render_default.png" 2>&1 | tail -25
section "sapien render: ray tracing (RoboTwin's settings)"
VK_LOADER_DEBUG=error timeout 300 python "$SCRATCH/render.py" rt "$OUT/render_rt.png" 2>&1 | tail -25
section "sapien render: ray tracing with an explicit EGL-based ICD (fallback)"
printf '{"file_format_version":"1.0.0","ICD":{"library_path":"libEGL_nvidia.so.0","api_version":"1.3.0"}}\n' > "$SCRATCH/icd.json"
VK_ICD_FILENAMES="$SCRATCH/icd.json" VK_LOADER_DEBUG=error timeout 300 python "$SCRATCH/render.py" rt "$OUT/render_rt_eglicd.png" 2>&1 | tail -12
} >> "$REPORT" 2>&1
echo "done; report at $REPORT"
