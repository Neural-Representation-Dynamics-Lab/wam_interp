# Loading FastWAM in Colab

This guide takes a fresh Colab runtime through installing `wam_interp`, downloading the released LIBERO checkpoint, and loading it through the shared model adapter. It makes one model call possible; it does not launch LIBERO or run an evaluation episode.

## 1. Clone and install the repository

The FastWAM backend is on the `fastwam-model` branch while that work is being developed. If it has since been merged into the default branch, omit `--branch fastwam-model`.

```python
%cd /content
!git clone --branch fastwam-model https://github.com/Neural-Representation-Dynamics-Lab/wam_interp.git
%cd /content/wam_interp
%pip install -e ".[all]"
```

Restart the runtime after installation, then continue in a new cell. This avoids Python keeping an older installed copy of the package in memory.

## 2. Download the released checkpoint and dataset statistics

The checkpoint is hosted publicly on Hugging Face, so downloading these files does not require a token or `login()`. Keep the weights outside the repository checkout:

```python
from pathlib import Path
from huggingface_hub import hf_hub_download

CHECKPOINTS_PATH = Path("/content/checkpoints/fastwam_release")
CHECKPOINTS_PATH.mkdir(parents=True, exist_ok=True)

for filename in (
    "libero_uncond_2cam224.pt",
    "libero_uncond_2cam224_dataset_stats.json",
):
    hf_hub_download(
        repo_id="yuanty/fastwam",
        filename=filename,
        local_dir=CHECKPOINTS_PATH,
    )
```

The `.pt` file is the model checkpoint used below. The dataset-statistics file is required by `FastWAMLiberoAdapter` to normalize state input and denormalize predicted actions during live LIBERO evaluation. The Hugging Face repository also contains a README, but it is not needed to load the checkpoint.

## 3. Load FastWAM through the adapter

```python
import os
from wam_interp.models import load_model_adapter

# Optional: store downloaded base-model assets here instead of ./checkpoints/
os.environ["DIFFSYNTH_MODEL_BASE_PATH"] = "/content/checkpoints"

adapter = load_model_adapter(
    "fastwam",
    task="libero_uncond_2cam224_1e-4",
    checkpoint=CHECKPOINTS_PATH / "libero_uncond_2cam224.pt",
    device="cuda",
    dtype="float16",
)
```

The loader finds this checkout's `configs/fastwam/` directory automatically, so the call does not need a `repository_root` argument. It composes `sim_libero.yaml`, which sets `skip_dit_load_from_pretrain=true` and clears `action_dit_pretrained_path`. The model is therefore constructed without the separate ActionDiT initialization file; the released checkpoint is loaded into it before inference. The Wan 2.2 and text tokenizer/encoder assets are still needed, and the model code downloads any missing assets. By default, those files are cached under `./checkpoints/` relative to the process working directory. The optional environment setting in the example puts them under `/content/checkpoints` instead.

The released FastWAM checkpoint is separate from these base assets: the code first constructs the configured model, then loads the `.pt` checkpoint into it. This section demonstrates a single model load, not an evaluation episode. For live LIBERO evaluation, construct `FastWAMLiberoAdapter` with the raw loaded model, resolved data config, and statistics file, then pass it to `evaluate_libero()`. See [`the LIBERO evaluation guide`](libero_evaluation.md).

For the meaning of `libero_uncond_2cam224_1e-4`, the model variants, and the local implementation layout, see [the FastWAM backend README](../src/models/fastwam/README.md). The upstream [FastWAM repository](https://github.com/yuantianyuan01/FastWAM) and its [released checkpoints](https://huggingface.co/yuanty/fastwam) are useful references for the original project and model assets.
