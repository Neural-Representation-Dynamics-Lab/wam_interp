"""Prepare/review/run E1, then export aligned frames for trajectory annotation."""
import argparse
import json
from pathlib import Path

import numpy as np

from wam_interp.experiments.e1 import run_e1, context_id
from wam_interp.experiments.fastwam_session import _array
from wam_interp.experiments.libero_pilot import load_pilot, provenance, save_frames, digest_file, validate_config


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")


def candidate_requests(config, recipient_prompt):
    prompts = config.get("donor_prompts")
    if prompts:
        return [(f"{name}_seed_{seed}", seed, prompt)
                for name, prompt in prompts.items() for seed in config["candidate_seeds"]]
    return [(f"candidate_{seed}", seed, recipient_prompt) for seed in config["candidate_seeds"]]


def prepare(pilot, output):
    output.mkdir(parents=True, exist_ok=False)
    write(output / "status.json", {"status": "preparing", "synthetic": False})
    pilot.verify_replay()
    context = provenance(pilot)
    write(output / "context.json", context)
    # Baseline native call vs observer vs identity: fail before donor selection.
    pilot.restore(pilot.saved_rng)
    native = pilot.adapter.predict_action(pilot.request)
    observed, capture, latency = pilot.infer()
    identity, _, _ = pilot.infer(donor=capture["used"])
    errors = {"observer_max_action_error": float(np.max(np.abs(_array(native.actions) - _array(observed.actions)))),
              "identity_max_action_error": float(np.max(np.abs(_array(native.actions) - _array(identity.actions))))}
    if not all(np.isfinite(v) and v <= pilot.config["identity_atol"] for v in errors.values()):
        write(output / "preflight.json", errors)
        raise RuntimeError("Native/observer/identity preflight failed")
    import torch
    errors.update(latency_seconds_with_capture=latency,
                  peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated() if torch.cuda.is_initialized() else 0)
    write(output / "preflight.json", errors)
    np.savez_compressed(output / "baseline.npz", future=_array(capture["used"]))
    frame_count = save_frames(pilot.adapter.model, capture["used"], output / "baseline_frames")
    if frame_count != len(pilot.config["frame_steps"]):
        raise ValueError("Baseline frame count differs from configured mapping")
    write(output / "baseline_identity.json", {"sha256": digest_file(output / "baseline.npz"), "frames": frame_count})
    candidates = []
    for name, seed, prompt in candidate_requests(pilot.config, context["prompt"]):
        # Donor actions are discarded and never executed/used for selection.
        _, captured, _ = pilot.infer(seed=seed, prompt=prompt)
        path = output / f"{name}.npz"
        np.savez_compressed(path, future=_array(captured["used"]))
        save_frames(pilot.adapter.model, captured["used"], output / f"{name}_frames")
        candidates.append({"id": name, "seed": seed, "sha256": digest_file(path),
                           "generation_request": {"prompt": prompt, "seed": seed,
                               "recipient_prompt": context["prompt"],
                               "protocol": "counterfactual_instruction" if pilot.config.get("donor_prompts") else "same_instruction_seed_variation"},
                           "label": None, "nuisance_notes": "", "acceptable": False})
    write(output / "review.json", {"reviewed_before_execution": False,
        "context_id": context_id(context), "property": pilot.config["property"],
        "recipient_label": None, "expected_sign": None,
        "same_property": None, "opposite_property": None,
        "frame_mapping_verified": False, "frame_steps": pilot.config["frame_steps"],
        "candidates": candidates})
    # Keep a separate immutable index for integrity checks after manual review.
    write(output / "candidate_index.json", candidates)
    write(output / "status.json", {"status": "awaiting_semantic_review", "synthetic": False})


def reviewed_manifest(prepared, current_context):
    status = json.loads((prepared / "status.json").read_text())
    if status.get("status") != "awaiting_semantic_review":
        raise ValueError("Candidate preparation is not complete")
    context = json.loads((prepared / "context.json").read_text())
    if context != current_context:
        raise ValueError("Prepared and current context differ; regenerate candidates")
    review = json.loads((prepared / "review.json").read_text())
    if review["reviewed_before_execution"] is not True or review["frame_mapping_verified"] is not True:
        raise ValueError("Review semantics and frame mapping before running E1")
    if review["context_id"] != context_id(context):
        raise ValueError("Review context mismatch")
    if not review.get("recipient_label") or review.get("expected_sign") not in (-1, 1):
        raise ValueError("Review requires a recipient label and expected direction")
    baseline_identity = json.loads((prepared / "baseline_identity.json").read_text())
    if digest_file(prepared / "baseline.npz") != baseline_identity["sha256"]:
        raise ValueError("Baseline changed after preparation")
    index = {c["id"]: c for c in json.loads((prepared / "candidate_index.json").read_text())}
    labels = {c["id"]: c for c in review["candidates"]}
    manifest = {"context": context, "synthetic": False, "property": review["property"],
        "recipient_label": review["recipient_label"], "expected_sign": review["expected_sign"], "donors": {}}
    donors = {}
    conditions = ("same_property", "opposite_property") + (("matched_same_property",) if review.get("matched_same_property") else ())
    for condition in conditions:
        selected = review[condition]
        if selected not in index or selected not in labels:
            raise ValueError(f"Select an existing {condition} candidate")
        path = prepared / f"{selected}.npz"
        if digest_file(path) != index[selected]["sha256"]:
            raise ValueError("Candidate file changed after generation")
        if labels[selected]["acceptable"] is not True:
            raise ValueError("Selected candidate was not accepted")
        if labels[selected].get("generation_request") != index[selected].get("generation_request"):
            raise ValueError("Donor instruction provenance changed during review")
        same = labels[selected].get("label") == review["recipient_label"]
        if not labels[selected].get("label") or same != (condition != "opposite_property") or not labels[selected].get("nuisance_notes"):
            raise ValueError("Donor label or nuisance review does not match its condition")
        with np.load(path, allow_pickle=False) as data:
            donors[condition] = data["future"].copy()
        manifest["donors"][condition] = {**labels[selected], "context_id": context_id(context),
                                        "reviewed_before_execution": True}
    steps = review["frame_steps"]
    if len(steps) != baseline_identity["frames"] or len(steps) < 2 or steps[0] != 0 or any(type(x) is not int for x in steps) or any(b <= a for a, b in zip(steps, steps[1:])):
        raise ValueError("Invalid reviewed video-to-control-step mapping")
    return manifest, donors, steps


def run(pilot, prepared, output):
    pilot.verify_replay()
    manifest, donors, frame_steps = reviewed_manifest(prepared, provenance(pilot))
    with np.load(prepared / "baseline.npz", allow_pickle=False) as data:
        prepared_future = data["future"].copy()
    underlying = pilot.session
    conditions = ("baseline", "identity", "same_property", "opposite_property") + (("matched_same_property",) if "matched_same_property" in donors else ())
    from PIL import Image
    from wam_interp.types import EnvironmentObservation
    class RecordingSession:
        call = 0
        def snapshot(self): return underlying.snapshot()
        def restore(self, state): underlying.restore(state)
        def run(self, donor):
            name = conditions[self.call]
            self.call += 1
            branch = underlying.run(donor)
            if name == "baseline":
                if branch.future.shape != prepared_future.shape or not np.allclose(branch.future, prepared_future, rtol=0, atol=pilot.config["identity_atol"]):
                    raise RuntimeError("Baseline future differs from the reviewed future; E1 stopped")
                self.baseline_actions = branch.actions.copy()
                self.baseline_execution = {k: v.copy() for k, v in pilot.last_rollout.items()}
            if name == "identity" and not np.allclose(branch.actions, self.baseline_actions, rtol=0, atol=pilot.config["identity_atol"]):
                raise RuntimeError("Identity control failed; E1 stopped")
            if name == "identity" and any(not np.array_equal(v, pilot.last_rollout[k]) for k, v in self.baseline_execution.items()):
                raise RuntimeError("Identity execution differs from baseline; E1 stopped")
            frames = output / f"{name}_imagined"
            count = save_frames(pilot.adapter.model, branch.future, frames)
            if count != len(frame_steps):
                raise ValueError("Decoded frame count differs from reviewed mapping")
            rollout = pilot.last_rollout
            np.savez_compressed(output / f"{name}_execution.npz", **rollout)
            actual = output / f"{name}_executed"
            actual.mkdir()
            pairs = []
            for frame, step in enumerate(frame_steps):
                if step >= len(rollout["control_steps"]): break
                # Reuse native rotation/crop/resize so both videos share pixels.
                obs = EnvironmentObservation(cameras={"agentview": rollout["agentview"][step],
                    "wrist": rollout["wrist"][step]}, state=pilot.reference.state, instruction=pilot.task.language)
                pixels = _array(pilot.adapter.prepare_observation(obs).image)[0]
                pixels = np.clip((pixels.transpose(1, 2, 0) + 1) * 127.5, 0, 255).astype(np.uint8)
                Image.fromarray(pixels).save(actual / f"{frame:03d}.png")
                pairs.append({"step": step, "imagined": f"{name}_imagined/{frame:03d}.png",
                              "executed": f"{name}_executed/{frame:03d}.png"})
            write(output / f"{name}_frame_pairs.json", {"state_id": manifest["context"]["state_id"],
                "context_id": context_id(manifest["context"]),
                "condition": name, "synthetic": False, "pairs": pairs,
                "coordinate_frame": "agentview_rotated_cropped", "units": "normalized_image_xy"})
            return branch
    result = run_e1(RecordingSession(), donors=donors, manifest=manifest, output=output)
    from wam_interp.experiments.annotation import make_annotation_page
    for pairs in output.glob("*_frame_pairs.json"):
        make_annotation_page(pairs)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage", choices=("check", "prepare", "run"))
    p.add_argument("--config", type=Path, default=Path("configs/experiments/e1_libero.json"))
    p.add_argument("--prepared", type=Path)
    p.add_argument("--output", type=Path)
    args = p.parse_args()
    config = json.loads(args.config.read_text())
    validate_config(config)
    if args.stage == "check":
        print(json.dumps({"config_valid": True, "checkpoint_present": Path(config["checkpoint"]).is_file(),
            "statistics_present": Path(config["statistics"]).is_file(), "gpu_tested": False}, indent=2))
        return
    if args.output is None or (args.stage == "run" and args.prepared is None):
        p.error("--output required; run also requires --prepared")
    if args.output.exists():
        p.error("Output exists; choose a fresh directory")
    if args.stage == "run":
        # Reject incomplete review before spending GPU time on model loading.
        reviewed_manifest(args.prepared, json.loads((args.prepared / "context.json").read_text()))
    pilot = None
    try:
        pilot = load_pilot(config)
        if args.stage == "prepare": prepare(pilot, args.output)
        else: print(json.dumps(run(pilot, args.prepared, args.output), indent=2))
    except Exception as exc:
        if args.output.exists(): write(args.output / "status.json", {"status": "failed", "error": str(exc)})
        raise
    finally:
        if pilot is not None: pilot.close()


if __name__ == "__main__": main()
