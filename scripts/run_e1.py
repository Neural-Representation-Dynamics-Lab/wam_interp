"""CPU rehearsal and E1 execution through an explicitly supplied session factory."""

import argparse
import importlib
import json
from pathlib import Path
import subprocess

import numpy as np

from wam_interp.experiments.e1 import Branch, context_id, run_e1
from wam_interp.experiments.trajectories import trajectory_metrics


class SyntheticSession:
    """Deliberately simple fixture. Its effects are not scientific evidence."""
    def __init__(self):
        self.position = 0.0

    def snapshot(self):
        return self.position

    def restore(self, state):
        self.position = state

    def run(self, donor):
        future = np.array([0., 1., 2.]) if donor is None else donor.copy()
        commands = np.diff(future)[:, None]
        self.position += float(commands.sum())
        return Branch(future, commands.copy(), commands, commands.copy(), self.position,
                      {"synthetic": True, "termination": "chunk_complete"})


def rehearsal(output):
    context = dict.fromkeys(("checkpoint", "statistics", "resolved_config", "observation_sha256",
                             "state_sha256"), "SYNTHETIC")
    context.update(repo_revision=subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parents[1], text=True).strip(),
        backend="synthetic", prompt="fixture", action_seed=0, precision="float64",
        behavior_measure="synthetic endpoint x", state_id="synthetic_0")
    manifest = {"context": context, "property": "direction", "recipient_label": "right",
                "expected_sign": -1, "synthetic": True, "donors": {}}
    for name, label in (("same_property", "right"), ("opposite_property", "left")):
        manifest["donors"][name] = {"context_id": context_id(context), "label": label,
            "reviewed_before_execution": True, "nuisance_notes": "Synthetic construction only"}
    donors = {"same_property": np.array([0., 1.1, 2.2]),
              "opposite_property": np.array([0., -1., -2.])}
    result = run_e1(SyntheticSession(), donors=donors, manifest=manifest, output=output)
    # The later trajectory analysis is a distinct stage, using saved coordinates.
    times = np.array([0., .2, .4])
    imagined = np.array([[0., 0.], [.1, .1], [.2, .2]])
    executed = np.array([[0., 0.], [.1, .09], [.2, .18]])
    np.savez_compressed(output / "trajectory_fixture.npz", times=times,
                        imagined=imagined, executed=executed, valid=np.ones(3, dtype=bool))
    (output / "trajectory_fixture.json").write_text(json.dumps({
        "synthetic": True, "landmark": "gripper", "coordinate_frame": "fixed_external_camera",
        "units": "normalized_image_xy", "executed_until": .4,
        "condition": "baseline", "state_id": "synthetic_0"}, indent=2), encoding="utf-8")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subs = parser.add_subparsers(dest="command", required=True)
    demo = subs.add_parser("rehearse", help="No torch, checkpoint, GPU, or simulator")
    demo.add_argument("--output", type=Path, required=True)
    run = subs.add_parser("run", help="Requires a session with verified simulator/model restoration")
    run.add_argument("--factory", required=True, help="module:function(manifest) -> Session")
    run.add_argument("--manifest", type=Path, required=True)
    run.add_argument("--donors", type=Path, required=True, help="NPZ: same_property, opposite_property")
    run.add_argument("--output", type=Path, required=True)
    analyze = subs.add_parser("trajectory", help="Follow-up analysis of aligned coordinate tracks")
    analyze.add_argument("--tracks", type=Path, required=True)
    analyze.add_argument("--metadata", type=Path, required=True)
    analyze.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "rehearse":
        result = rehearsal(args.output)
    elif args.command == "run":
        manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
        module, function = args.factory.split(":", 1)
        session = getattr(importlib.import_module(module), function)(manifest)
        with np.load(args.donors, allow_pickle=False) as arrays:
            result = run_e1(session, donors=dict(arrays), manifest=manifest, output=args.output)
    else:
        metadata = json.loads(args.metadata.read_text(encoding="utf-8"))
        required = {"synthetic", "landmark", "coordinate_frame", "units", "executed_until", "condition", "state_id"}
        if not required <= metadata.keys():
            parser.error(f"Missing trajectory metadata: {sorted(required - metadata.keys())}")
        with np.load(args.tracks, allow_pickle=False) as arrays:
            result = {"metadata": metadata, "metrics": trajectory_metrics(
                arrays["times"], arrays["imagined"], arrays["executed"],
                valid=arrays.get("valid"), executed_until=metadata["executed_until"])}
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8") as stream:
            json.dump(result, stream, indent=2, allow_nan=False)
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
