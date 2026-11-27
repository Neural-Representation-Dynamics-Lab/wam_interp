"""Roadmap E7 content arm: swap Fast-WAM's imagined future for a scripted approach.

For each start, run_reference from e1_simulator_pairs.py renders two scripted pre-contact approaches,
one to each object of the task's pair. Each is encoded as Fast-WAM's future latent with the observed
first latent frame kept, and one 32-control chunk is executed under the native future and under each
scripted future. ID starts are the original libero_10 tasks and OOD starts the E7 LIBERO-Plus variants,
both run in the LIBERO-Plus install.

Run from the repo root with wam_interp resolving to this checkout:

    python scripts/e7_scripted_future.py --arm checks --output-dir results/e7_scripted
    python scripts/e7_scripted_future.py --arm id --output-dir results/e7_scripted
"""

import argparse
import json
import subprocess
from argparse import Namespace
from contextlib import contextmanager
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

import e7_future_ablation as e7
from e1_simulator_pairs import run_reference, sheet
from wam_interp.evaluations.LIBERO import environment as libero_env
from wam_interp.evaluations.LIBERO import runner as libero_runner
from wam_interp.evaluations.LIBERO.environment import create_libero_environment, reset_to_initial_state, step_libero
from wam_interp.evaluations.LIBERO_PLUS.environment import initial_states_for_variant
from wam_interp.evaluations.provenance import package_versions
from wam_interp.experiments.fastwam_session import _array
from wam_interp.experiments.libero_pilot import LiberoPilot, validate_config
from wam_interp.experiments.simulator_video import encode_reference_video, paired_camera_clip, valid_approach
from wam_interp.models.fastwam.libero_adapter import FastWAMLiberoAdapter
from wam_interp.models.fastwam.loader import compose_fastwam_config, load_fastwam

EXECUTED = 10  # steps Fast-WAM's LIBERO eval runs before replanning


class _OneTask:
    def __init__(self, task):
        self.task = task

    def get_task(self, task_id):
        return self.task


class _NoAppearance:
    """Stands in for run_reference's table-tint capture, which E7 does not use."""

    enabled = False
    latest = {}
    geom_ids = np.zeros(0, dtype=int)
    material_id = -1
    convention = 0

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


@contextmanager
def fixed_task(task, states):
    """run_reference loads its task through _get_suite, which in the LIBERO-Plus install lists variants."""
    saved = libero_runner._get_suite, libero_env.initial_states_for_task
    libero_runner._get_suite = lambda name: _OneTask(task)
    libero_env.initial_states_for_task = lambda suite, task_id: states
    try:
        yield
    finally:
        libero_runner._get_suite, libero_env.initial_states_for_task = saved


def scripted_clip(config, start, label):
    with fixed_task(start["task"], start["states"]):
        arrays, meta = run_reference(config, start["index"], label, capture=False,
                                     capture_factory=lambda env: _NoAppearance())
    steps = config["frame_steps"]
    return paired_camera_clip(arrays["agentview"][steps], arrays["wrist"][steps]), arrays, meta


def id_starts(states_per_task):
    for task_id, (task, states) in enumerate(e7.original_tasks()):
        for index in range(states_per_task):
            yield {"unit": f"task{task_id}_state{index}", "base_task": task_id, "task_id": task_id,
                   "task": task, "states": states, "index": index}


def ood_starts(ood):
    cells = e7.sample_variants(Namespace(levels=tuple(ood["levels"]), variants_per_cell=ood["variants_per_cell"],
                                         sample_seed=ood["sample_seed"]))
    suite = libero_runner._get_suite("libero_10")
    for cell, variants in cells.items():
        category, level = cell.rsplit(":", 1)
        for variant in variants:
            task = suite.get_task(variant)
            base = max((i for i, name in enumerate(e7.ORIGINAL_LIBERO_10) if task.name.startswith(name)),
                       key=lambda i: len(e7.ORIGINAL_LIBERO_10[i]))
            yield {"unit": f"variant{variant}", "base_task": base, "task_id": variant, "category": category,
                   "level": int(level), "task": task, "states": initial_states_for_variant(suite, variant), "index": 0}


def load_adapter(args, protocol):
    # E1 loads the original Wan .pth files; this uses the converted copies the E7 Faster-WAM runs already use.
    cfg = compose_fastwam_config("libero_optional_idm_2cam224_1e-4", protocol["pilot"]["repository_root"])
    model = load_fastwam(cfg, str(args.checkpoint), device="cuda", dtype="bfloat16")
    return FastWAMLiberoAdapter(model, data_config=cfg.data.train, dataset_stats=str(args.dataset_stats))


def branch(pilot, donor, anchor):
    """One chunk from the restored start; the observed latent frame and any donor must arrive unchanged."""
    pilot.restore(pilot.saved_rng)
    result = pilot.session.run(donor)
    if not np.array_equal(result.future[:, :, :1], anchor[:, :, :1]):
        raise RuntimeError("Observed latent frame changed")
    if donor is not None:
        expected = torch.as_tensor(donor).to(torch.bfloat16).float().numpy()
        if not np.array_equal(result.future[:, :, 1:], expected[:, :, 1:]):
            raise RuntimeError("Scripted future was not inserted as given")
    return result, pilot.last_rollout["eef_position"].copy()


def compare(runs, labels, objects, axis):
    """Endpoint shift toward the second object, endpoint distances and action changes for one seed."""
    first, second = labels
    shift = {arm: float(np.dot(eef[-1, :2] - eef[0, :2], axis)) for arm, (_, eef) in runs.items()}
    out = {"axis_shift_m": shift, "toward_shown_m": shift[second] - shift[first],
           "endpoint_separation_m": float(np.linalg.norm(runs[second][1][-1] - runs[first][1][-1]))}
    out["closer_to_shown"] = {
        label: bool(np.linalg.norm(runs[label][1][-1, :2] - objects[label][:2])
                    < np.linalg.norm(runs[label][1][-1, :2] - objects[other][:2]))
        for label, other in ((first, second), (second, first))}
    actions = {arm: result.actions.astype(np.float64) for arm, (result, _) in runs.items()}
    out["action_abs_diff_executed"] = {
        f"{a}-{b}": float(np.abs(actions[a][:EXECUTED] - actions[b][:EXECUTED]).mean())
        for a, b in ((first, "native"), (second, "native"), (second, first))}
    out["termination"] = {arm: result.metadata["rollout"]["termination"] for arm, (result, _) in runs.items()}
    return out


def run_start(adapter, protocol, args, start, folder, repeat_native=False):
    pair = protocol["pairs"][str(start["base_task"])]
    labels = list(pair)
    row = {key: start[key] for key in ("unit", "base_task", "task_id", "category", "level") if key in start}
    row.update(task=start["task"].name, instruction=start["task"].language, labels=labels, approach={})
    reference_config = {**protocol["pilot"], **protocol["reference"], "task_id": start["task_id"], "targets": pair}
    references = {}
    for label in labels:
        try:
            references[label] = scripted_clip(reference_config, start, label)
        except (RuntimeError, ValueError) as exc:
            row["rejected"] = f"{label}: {exc}"
            return row
        row["approach"][label] = references[label][2]["approach"]
    failed = [label for label in labels
              if not valid_approach(row["approach"][label], protocol["reference"]["approach_limits"])]
    if failed:
        row["rejected"] = "approach limits: " + ", ".join(failed)
        return row

    folder.mkdir(parents=True, exist_ok=True)
    for label in labels:
        sheet(references[label][0], folder / f"{label}_sheet.jpg")
    config = {**protocol["pilot"], "checkpoint": str(args.checkpoint), "statistics": str(args.dataset_stats),
              "task_id": start["task_id"], "initial_state_index": start["index"],
              "action_seed": protocol["action_seeds"][0]}
    validate_config(config)
    pilot = LiberoPilot(config, adapter, task=start["task"], initial_state=start["states"][start["index"]],
                        create_env=create_libero_environment, reset_env=reset_to_initial_state, step_env=step_libero)
    try:
        pilot.verify_replay()
        for label in labels:
            for camera in ("agentview", "wrist"):
                if not np.array_equal(references[label][1][camera][0], pilot.reference.cameras[camera]):
                    raise RuntimeError(f"Scripted run for {label} starts from a different {camera} image")
        observed = references[labels[0]][0][0]
        image = torch.as_tensor(observed.copy(), device=adapter.model.device).permute(2, 0, 1).unsqueeze(0)
        if not torch.equal(image.to(adapter.model.torch_dtype) * (2.0 / 255.0) - 1.0, pilot.request.observation.image):
            raise RuntimeError("Scripted frame zero differs from the policy's observation")
        meta = references[labels[0]][2]
        start_objects = references[labels[0]][1]["object_position"][0]
        objects = {labels[0]: start_objects[meta["target_index"]], labels[1]: start_objects[meta["other_index"]]}
        axis = (objects[labels[1]] - objects[labels[0]])[:2]
        axis = axis / np.linalg.norm(axis)

        native_request, row["seeds"] = pilot.request, {}
        for seed in protocol["action_seeds"]:
            pilot.request = pilot.session.request = replace(native_request, seed=seed)
            pilot.config["action_seed"] = seed
            _, captured, _ = pilot.infer()
            anchor = _array(captured["used"])
            runs = {"native": branch(pilot, None, anchor)}
            if not np.array_equal(runs["native"][0].future, anchor):
                raise RuntimeError("Native branch differs from the native inference")
            if repeat_native:
                again = branch(pilot, None, anchor)
                if not (np.array_equal(again[0].actions, runs["native"][0].actions)
                        and np.array_equal(again[1], runs["native"][1])):
                    raise RuntimeError("Native branch is not reproducible")
            for label in labels:
                encoded, codec = encode_reference_video(adapter.model, references[label][0], observed, anchor)
                runs[label] = branch(pilot, encoded["future"], anchor)
                row.setdefault("anchor_error", {})[f"{seed}/{label}"] = codec["anchor_error_before_restore"]
            row["seeds"][str(seed)] = compare(runs, labels, objects, axis)
            np.savez_compressed(folder / f"seed_{seed}.npz", **{f"{arm}_eef": eef for arm, (_, eef) in runs.items()},
                                **{f"{arm}_actions": result.actions for arm, (result, _) in runs.items()})
    finally:
        pilot.close()
    return row


def source_revision():
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], text=True).strip())
    return {"revision": head, "dirty": dirty}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--arm", choices=("checks", "id", "ood"), required=True)
    parser.add_argument("--protocol", type=Path, default=Path("configs/experiments/e7_scripted_future.json"))
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset-stats", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    protocol = json.loads(args.protocol.read_text())

    adapter = load_adapter(args, protocol)
    root = args.output_dir / args.arm
    if args.arm == "checks":
        start = next(id_starts(1))
        row = run_start(adapter, protocol, args, start, root / start["unit"], repeat_native=True)
        root.mkdir(parents=True, exist_ok=True)
        (root / "checks.json").write_text(json.dumps(row, indent=2))
        print("CHECKS", json.dumps(row), flush=True)
        return

    root.mkdir(parents=True, exist_ok=True)
    starts = id_starts(protocol["id_states"]) if args.arm == "id" else ood_starts(protocol["ood"])
    report = {"experiment": "E7 scripted future", "arm": args.arm, "protocol": protocol,
              "checkpoint": str(args.checkpoint), "source": source_revision(), "package_versions": package_versions(),
              "started": datetime.now(timezone.utc).isoformat(), "completed": False, "rows": []}
    path = root / "rows.json"
    began = perf_counter()
    for start in starts:
        tic = perf_counter()
        row = run_start(adapter, protocol, args, start, root / start["unit"])
        row["seconds"] = perf_counter() - tic
        report["rows"].append(row)
        report["duration_seconds"] = perf_counter() - began
        path.write_text(json.dumps(report, indent=1))
        print(json.dumps({k: row[k] for k in ("unit", "rejected") if k in row}), flush=True)
    report["completed"] = True
    path.write_text(json.dumps(report, indent=1))
    used = [row for row in report["rows"] if "rejected" not in row]
    print(f"RESULT {args.arm} {len(used)}/{len(report['rows'])} starts used {path}", flush=True)


if __name__ == "__main__":
    main()
