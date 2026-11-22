"""Audited E1 interfaces, distinct from ordinary benchmark support.

This module imports no model, tensor library, or simulator. Entries describe
the code at the linked boundary; they do not advertise unimplemented hooks.
"""

from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class ModelCapability:
    variant: str
    backend: str
    condition: str
    representation: str
    boundary: str
    protected_context: str
    noise_contract: str
    e1_status: str
    source_files: tuple[str, ...]
    paper: str
    checkpoint_kind: str = "file"
    needs_statistics: bool = True
    stateful: bool = False


CAPABILITIES = (
    ModelCapability(
        "fastwam_optional_idm", "fastwam", "idm", "clean_video_latents",
        "fastwam.idm.clean_video_before_action_prefill",
        "Keep latent time index 0 bitwise unchanged.",
        "Clean future after native video denoising; recipient context builds new K/V.",
        "exploratory_gpu_evidence",
        ("src/models/fastwam/model_code/fastwam_idm.py", "src/experiments/futures.py"),
        "https://arxiv.org/abs/2603.16666v2",
    ),
    ModelCapability(
        "fastwam_optional_first_frame", "fastwam", "first_frame", "current_frame_features",
        "none", "Only the observed frame is available to the action path.",
        "Same Optional IDM weights as the IDM arm; no future substitution exists in this mode.",
        "native_comparator_only",
        ("src/models/fastwam/model_code/fastwam_idm.py", "src/models/fastwam/adapter.py"),
        "https://arxiv.org/abs/2603.16666v2",
    ),
    ModelCapability(
        "fasterwam", "fasterwam", "one_pass_future_cache", "fused_future_kv",
        "planned: after SparseMoT.prefill_video_cache, before action denoising",
        "Keep current-frame tokens, sparse None slots and layer/fusion layout unchanged.",
        "One video pass on Gaussian future slots; match timestep, seed and post-RoPE K/V layout.",
        "boundary_audited_hook_pending",
        ("src/models/fasterwam/model_code/jointwam.py", "src/models/fasterwam/model_code/sparse_mot.py"),
        "https://arxiv.org/abs/2608.04404v1",
    ),
    ModelCapability(
        "lingbot_va", "lingbot_va", "video", "predicted_video_kv",
        "planned: final video cache write in LingBotVARuntime.infer_chunk",
        "Keep real history, action history, cache IDs/masks and the first-chunk anchor unchanged.",
        "Match the final video forward's timestep, both CFG branches and predicted cache slots; "
        "editing returned latents after the cache write would not change the actions.",
        "boundary_audited_hook_pending",
        ("src/models/lingbot_va/runtime.py", "src/models/lingbot_va/libero_adapter.py"),
        "https://arxiv.org/abs/2601.21998v1",
        checkpoint_kind="directory", needs_statistics=False, stateful=True,
    ),
    ModelCapability(
        "longwam_idm", "longwam", "idm", "partially_noisy_future_latents",
        "planned: infer_joint_ar, after _imagine_future_latents and before video prefill",
        "Protect all model.num_clean_frames history latents, not just the current frame.",
        "Match model.imagine_timestep / scheduler training steps (published default sigma=0.9). "
        "A clean RGB encoding is not a valid direct substitute.",
        "boundary_audited_hook_pending",
        ("src/models/longwam/model_code/models/wan22/longwam_base.py", "src/models/longwam/libero_adapter.py"),
        "https://arxiv.org/abs/2610.10528v1", stateful=True,
    ),
    ModelCapability(
        "openwam", "openwam", "joint", "joint_denoising_features",
        "planned: action-facing video K/V at a declared joint layer and denoising step",
        "Protect current-frame tokens, action states, training mask and unselected layers/steps.",
        "Match both video and action noise levels; preserve the checkpoint's native mask. "
        "Public mask hooks exist, but a mask ablation is not a content transplant.",
        "boundary_audited_hook_pending",
        ("src/models/openwam/adapter.py", "src/models/openwam/model_code/model/architectures/dual_system/joint_self_attn.py"),
        "https://arxiv.org/abs/2609.07398",
        checkpoint_kind="directory", needs_statistics=False,
    ),
)


def capability(variant: str) -> ModelCapability:
    for entry in CAPABILITIES:
        if entry.variant == variant:
            return entry
    raise ValueError(f"Unknown E1 variant {variant!r}; choose {[c.variant for c in CAPABILITIES]}")


def capability_report() -> dict:
    return {
        "audit_date": "2026-11-22",
        "scope": "LIBERO; native evaluator support does not imply an E1 intervention implementation",
        "models": [asdict(entry) for entry in CAPABILITIES],
    }


def require_future_transplant(variant: str) -> ModelCapability:
    entry = capability(variant)
    if entry.e1_status != "exploratory_gpu_evidence":
        raise NotImplementedError(
            f"{variant}: {entry.e1_status}. {entry.boundary}. "
            "Do not reuse the FastWAM clean-latent hook for this model."
        )
    return entry


def baseline_argv(variant, *, python, checkpoint, benchmark_root, output,
                  dataset_stats=None, task_ids=(0,), episodes=1, seed=0,
                  checkpoint_revision=None):
    """Build an argv list for the existing model-neutral native evaluator.

    Paths may be Linux paths when planning on Windows. No path probing, model
    import, download or evaluation happens here. Defaults are a one-episode
    runtime smoke check, not a semantic positive control or performance study.
    """
    entry = capability(variant)
    if type(episodes) is not int or episodes < 1 or type(seed) is not int or seed < 0:
        raise ValueError("episodes must be positive and seed must be nonnegative integers")
    ids = tuple(task_ids)
    if not ids or any(type(i) is not int or not 0 <= i < 10 for i in ids) or len(set(ids)) != len(ids):
        raise ValueError("Choose distinct libero_10 task IDs in [0,9]")
    if entry.needs_statistics and not dataset_stats:
        raise ValueError(f"{entry.backend} requires its matching dataset statistics")
    for name, value in (("python", python), ("checkpoint", checkpoint),
                        ("benchmark_root", benchmark_root), ("output", output)):
        if not str(value).strip() or value is None:
            raise ValueError(f"Missing {name}")
    argv = [str(python), "scripts/evaluate_benchmark.py", "--benchmark", "libero",
            "--backend", entry.backend, "--condition", entry.condition,
            "--checkpoint", str(checkpoint), "--benchmark-root", str(benchmark_root),
            "--suite", "libero_10", "--task-ids", *(str(i) for i in ids),
            "--episodes", str(episodes), "--seed", str(seed), "--dtype", "bfloat16",
            "--output", str(output)]
    if dataset_stats:
        argv += ["--dataset-stats", str(dataset_stats)]
    if checkpoint_revision:
        argv += ["--checkpoint-revision", str(checkpoint_revision)]
    if entry.backend == "fastwam":
        argv += ["--fastwam-task", "libero_optional_idm_2cam224_1e-4"]
    # Long-WAM resolves the native config.yaml next to the checkpoint.
    # OpenWAM's default exact eager path preserves the trained attention mask.
    return argv


def validate_local_assets(variant, *, checkpoint, dataset_stats=None):
    """Optional host-side path check, separate from portable command planning."""
    entry = capability(variant)
    path = Path(checkpoint).expanduser()
    present = path.is_file() if entry.checkpoint_kind == "file" else path.is_dir()
    if not present:
        raise FileNotFoundError(f"Expected checkpoint {entry.checkpoint_kind}: {path}")
    if entry.needs_statistics and (not dataset_stats or not Path(dataset_stats).is_file()):
        raise FileNotFoundError("Missing matching statistics JSON")
    if entry.backend == "longwam" and not (path.parent / "config.yaml").is_file():
        raise FileNotFoundError("Long-WAM requires config.yaml next to the checkpoint")
