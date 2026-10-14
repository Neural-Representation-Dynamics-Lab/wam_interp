"""Single-observation orchestration and small CLI."""

import argparse
import json
from pathlib import Path

from wam_interp.config import RunConfig
from wam_interp.data.npz import load_observation_npz
from wam_interp.models import get_model_adapter, load_model
from wam_interp.tracing.recorder import TraceRecorder
from wam_interp.types import InferenceRequest


def run(config: RunConfig):
    config.validate()
    root = Path(__file__).resolve().parents[3]
    model = load_model(
        config.backend, task=config.task, repository_root=root,
        checkpoint=config.checkpoint, device=config.device, dtype=config.dtype,
    )
    observation = load_observation_npz(config.observation, config.prompt)
    request = InferenceRequest(
        observation=observation,
        condition=config.condition,
        action_horizon=config.action_horizon,
        num_video_frames=config.num_video_frames,
        num_inference_steps=config.num_inference_steps,
        seed=config.seed,
        compile_action_infer=False,
        trace_modules=config.trace_modules,
    )
    recorder = TraceRecorder() if config.trace_modules else None
    adapter = get_model_adapter(config.backend, model=model)
    result = adapter.predict_action(request, recorder=recorder)
    payload = {
        "metadata": result.metadata,
        "normalized_actions": result.actions.detach().cpu().tolist(),
    }
    config.output.parent.mkdir(parents=True, exist_ok=True)
    config.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    if recorder is not None:
        recorder.save(config.output.with_suffix(".trace.pt"))
    return result


def _parser():
    parser = argparse.ArgumentParser(description="Run one model-ready WAM observation.")
    parser.add_argument("--backend", default="fastwam")
    parser.add_argument("--task", default="libero_optional_idm_2cam224_1e-4")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--observation", type=Path, required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--condition", default="first_frame",
                        help="Inference condition understood by the selected model adapter")
    parser.add_argument("--action-horizon", type=int, default=32)
    parser.add_argument("--num-video-frames", type=int, default=33)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=("float32", "float16", "bfloat16"), default="bfloat16")
    parser.add_argument("--output", type=Path, default=Path("wam_interp_result.json"))
    parser.add_argument("--trace-module", action="append", default=[])
    return parser


def main(argv=None):
    args = _parser().parse_args(argv)
    config = RunConfig(
        backend=args.backend, task=args.task, checkpoint=args.checkpoint,
        observation=args.observation, prompt=args.prompt, condition=args.condition,
        action_horizon=args.action_horizon, num_video_frames=args.num_video_frames,
        num_inference_steps=args.steps, seed=args.seed, device=args.device,
        dtype=args.dtype, output=args.output, trace_modules=tuple(args.trace_module),
    )
    run(config)


if __name__ == "__main__":
    main()

