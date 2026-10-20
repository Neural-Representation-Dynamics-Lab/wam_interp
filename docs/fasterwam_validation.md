# Faster-WAM H100 validation handoff

Local CPU tests cover config resolution, native adapter dispatch, registration,
missing checkpoints, trace cleanup, and LIBERO transform formulas. They do not
establish released-checkpoint correctness. Run the following on Linux with an
H100, outbound access to Hugging Face, and storage for the policy and Wan assets.
No full training dataset is required. Record the commands and retain result
files, including failures.

## Environment

From the `wam_interp` checkout containing this change:

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -e ".[fasterwam,libero]" pytest rich
python -m pytest tests/test_fasterwam.py -q
export DIFFSYNTH_DOWNLOAD_SOURCE=huggingface
export DIFFSYNTH_MODEL_BASE_PATH="$(pwd)/checkpoints/wan_assets"
```

The pinned native comparison checkout is only needed for validation; normal
backend use needs no external FasterWAM installation:

```bash
git clone https://github.com/hustvl/FasterWAM.git third_party/FasterWAM
git -C third_party/FasterWAM checkout --detach 83667817df0d4f823f39d90700e61ea2f432ac45
python scripts/validate_fasterwam.py --mode download
```

Download mode resolves the Hugging Face revision to a commit and records it
with checkpoint/statistics SHA-256 hashes in `results/fasterwam/download.json`.
For repeat downloads pass `--revision` with that recorded commit. Base assets
download on the first model load and are shared by both comparison runs.

## One-observation numerical comparison

Use an existing correctly prepared NPZ when available. Otherwise generate a
fixed synthetic input for numerical parity; synthetic input is not evidence of
robot task performance:

```bash
python scripts/validate_fasterwam.py --mode fixture
python scripts/validate_fasterwam.py --mode port --observation results/fasterwam/observation.npz --prompt "pick up the object"
python scripts/validate_fasterwam.py --mode upstream --upstream-root third_party/FasterWAM --observation results/fasterwam/observation.npz --prompt "pick up the object"
python scripts/validate_fasterwam.py --mode compare
```

The upstream worker executes the original `create_fasterwam` factory function
extracted from its source, avoiding its module-level training imports. It then
calls the original model inference method. Both workers use the same resolved
config, checkpoint, prepared input, prompt, seed, bf16 dtype, and steps. They run
in separate processes, so only one model occupies GPU memory at a time.

Each worker warms up once, then saves `[32,7]` actions, finite-value checks,
dtype, synchronized latency, and peak allocated CUDA memory. The comparison
checks matching input hashes/settings, reports maximum/mean absolute action
error, and exits unsuccessfully if actions differ beyond `rtol=1e-4, atol=1e-5`.
Tolerances are explicit and configurable; do not relax them without reporting
the observed numerical discrepancy. Repeat with `--condition joint_rollout`
and a separate `--output results/fasterwam_joint` for both workers and compare.

## One LIBERO task/trial

Follow `docs/libero_evaluation.md` to install the pinned official LIBERO source
and configure `LIBERO_CONFIG_PATH` and `MUJOCO_GL=egl`. Then run:

```bash
python scripts/validate_fasterwam.py --mode libero --suite libero_spatial --task-id 0
```

This uses the existing shared evaluator, 30 no-op warm-up steps, 32 predicted
actions per call, a 10-action execution prefix, nine video frames, and the
released normalization statistics. It saves one full-length trial's outcome
under `results/fasterwam/libero/`. A failed task is an observed outcome; it is
not automatically a broken integration. Retain its logs for review.

Successful handoff evidence consists of `download.json`, `port.json`,
`upstream.json`, both action tensors, `comparison.json`, and the LIBERO result
JSON. No H100 results are claimed by the local CPU checks.
