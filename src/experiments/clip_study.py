"""Paired open-loop LIBERO test using saved book-task stimuli and a loaded FutureAdapter."""

import hashlib
import json
import random
from contextlib import contextmanager, nullcontext
from pathlib import Path

import numpy as np
from wam_interp.evaluations._libero_rollout import (
    LIBERO_DUMMY_ACTION,
    _request_for_call,
)
from wam_interp.evaluations.LIBERO.config import LiberoEvalConfig
from wam_interp.evaluations.LIBERO.environment import (
    create_libero_environment,
    initial_states_for_task,
    reset_to_initial_state,
    step_libero,
)
from wam_interp.evaluations.LIBERO.runner import _get_suite


def save(path, value):
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf8")
    temp.replace(path)


@contextmanager
def start(
    futures,
    task,
    state,
    index,
    *,
    task_id=5,
    seed=0,
    xml_processor=None,
    warmup_context=None,
):
    random.seed(0)
    np.random.seed(0)
    env = create_libero_environment(task, 256, 0)
    meta = {
        "instruction": task.language,
        "metadata": {"task_id": task_id, "initial_state_index": index},
    }
    try:
        if xml_processor is not None:
            original = env.env._xml_processor
            env.env.set_xml_processor(
                (lambda xml: xml_processor(original(xml)))
                if original
                else xml_processor
            )
        obs = reset_to_initial_state(env, state, **meta)
        with warmup_context(env) if warmup_context else nullcontext():
            for _ in range(30):
                step = step_libero(env, LIBERO_DUMMY_ACTION, **meta)
                if step.done or step.success:
                    raise RuntimeError("Task ended during reset")
                obs = step.observation
        if futures is None:
            yield env, meta, obs, None
            return
        adapter = futures.adapter
        begin = getattr(adapter, "begin_episode", None)
        if begin:
            begin(obs)
        config = LiberoEvalConfig(inference_options={"condition": "idm"})
        request = _request_for_call(
            config, adapter, adapter.prepare_observation(obs), seed, 0
        )
        yield env, meta, obs, request
    finally:
        env.close()


def execute(futures, task, state, index, donor, anchor):
    import torch

    with start(futures, task, state, index) as (env, meta, obs, request):
        if not torch.equal(request.observation.image, anchor.image) or not torch.equal(
            request.observation.proprio, anchor.proprio
        ):
            raise ValueError("Recipient reset differs from the source")
        if donor is None:
            prediction = futures.adapter.predict_action(request)
        else:
            prediction = futures.predict_with_future(request, donor)
        cmds = futures.adapter.postprocess_actions(prediction.actions, environment="libero")
        cmds = torch.as_tensor(cmds).detach().cpu().float().numpy().reshape(-1, 7)
        if cmds.shape != (32, 7) or not np.isfinite(cmds).all():
            raise ValueError("Expected 32 finite LIBERO commands")
        path = [obs.state["eef_position"].copy()]
        for cmd in cmds:
            step = step_libero(env, cmd, **meta)
            path.append(step.observation.state["eef_position"].copy())
            if step.done or step.success:
                break
        if len(path) != 33:
            raise ValueError("Unmatched open-loop execution window")
        return np.stack(path)


def source_cases(root):
    for i in range(4):
        yield "motion", i, root / f"e1_motion_control/state_{i}"
    for i in (*range(4), *range(24, 34)):
        split = "development" if i < 4 else "held_out"
        yield "objects", i, root / f"e1_prescribed_slide/{split}/task_5/state_{i}/seed_0"


def score_directions(records):
    """Decode the orientation of a paired difference by a frozen positive-direction template."""
    results = {}
    for prop in ("arm", "book", "mug"):
        dev = [
            v["directions"][prop]
            for v in records
            if v["split"] == "development" and prop in v["directions"]
        ]
        test = [v for v in records if v["split"] == "held_out" and prop in v["directions"]]
        scores = []
        if dev:
            template = np.stack(dev).mean(axis=0)
            norm = np.linalg.norm(template)
            scores = [float(template @ v["directions"][prop] / norm) if norm else 0.0 for v in test]
        floor = 2 if prop == "arm" else 4
        results[prop] = {
            "training_states": len(dev),
            "tested_states": [r["state"] for r in test],
            "cosines": scores,
            "correct": sum(x > 0 for x in scores),
            "passed": len(scores) >= floor
            and sum(x > 0 for x in scores) / len(scores) >= (1.0 if prop == "arm" else 0.75),
        }
    return results


def run_clip_study(futures, source_root, output):
    source_root, output = Path(source_root), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if not futures.supports_clip:
        raise NotImplementedError(f"No clip preparation for {futures.backend}")
    suite = _get_suite("libero_10")
    task = suite.get_task(5)
    if "book" not in task.language:
        raise ValueError("Wrong LIBERO task")
    states = initial_states_for_task(suite, 5)
    specs = {
        "arm": ("-1", "1"),
        "book": ("black_book_1_-1", "black_book_1_1"),
        "mug": ("white_yellow_mug_1_-1", "white_yellow_mug_1_1"),
    }
    protocol = {
        "backend": futures.backend,
        "task": task.language,
        "horizon": 32,
        "action_seed": 0,
        "sampling_controls": list(range(0, 33, 4)),
        "warmup": 30,
        "max_sham_drift_m": 0.02,
        "capture_site": futures.capability.boundary,
        "match": "identical byte-level source archive across models",
        "content_check": "Paired latent differences, native VAE features. Arm train 0-1, test 2-3, both positive. Objects train 0-3, test 24-33 with source pass, at least 4 test states and 75% positive per property. All before viewing actions.",
        "noise": "Long-WAM training forward-noise at model.imagine_timestep, same noise across arms per state. External intervention, distinct from its native partial denoising.",
    }
    save(output / "protocol.json", protocol)
    rows, records = [], []
    for kind, idx, directory in source_cases(source_root):
        source = json.loads((directory / "case.json").read_text())
        split = "development" if idx < (2 if kind == "motion" else 4) else "held_out"
        row = {"kind": kind, "state": idx, "split": split, "source_pass": source["source_pass"]}
        rows.append(row)
        if not row["source_pass"]:
            continue
        clip_path = directory / "source_clips.npz"
        row["clip_noise_seed"] = 20261123 + idx if futures.backend == "longwam" else None
        row["source_sha256"] = hashlib.sha256(clip_path.read_bytes()).hexdigest()
        with (
            np.load(clip_path, allow_pickle=False) as clips,
            start(futures, task, states[idx], idx) as (_, _, _, request),
        ):
            options = {"noise_seed": row["clip_noise_seed"]} if futures.backend == "longwam" else {}
            donors = {key: futures.from_clip(request, clips[key], **options) for key in clips.files}
        directions = {}
        for prop in ("arm",) if kind == "motion" else ("book", "mug"):
            minus, plus = specs[prop]
            diff = (donors[plus].parts[0].float() - donors[minus].parts[0].float()).numpy().reshape(-1)
            norm = np.linalg.norm(diff)
            directions[prop] = diff / norm if norm else diff
        records.append(
            {
                "row": row,
                "state": idx,
                "split": split,
                "directory": directory,
                "donors": donors,
                "directions": directions,
                "anchor": request.observation,
            }
        )
    gates = score_directions(records)
    save(output / "content_check.json", gates)
    for case in records:
        idx, row = case["state"], case["row"]
        paths = {}
        for key in ("native", "sham"):
            paths[key] = execute(
                futures,
                task,
                states[idx],
                idx,
                None if key == "native" else case["donors"][key],
                case["anchor"],
            )
        row["sham_drift_m"] = float(np.linalg.norm(paths["sham"][-1] - paths["native"][-1]))
        row["sham_pass"] = row["sham_drift_m"] <= 0.02
        row["effects_m"] = {}
        if row["sham_pass"]:
            for prop in case["directions"]:
                regression_only = futures.backend == "fastwam" and row["kind"] == "motion"
                if not gates[prop]["passed"] and not regression_only:
                    continue
                minus, plus = specs[prop]
                for key in (minus, plus):
                    paths[key] = execute(
                        futures, task, states[idx], idx, case["donors"][key], case["anchor"]
                    )
                if gates[prop]["passed"]:
                    row["effects_m"][prop] = float((paths[plus][-1, 0] - paths[minus][-1, 0]) / 2)
        if futures.backend == "fastwam" and (case["directory"] / "execution.npz").is_file():
            with np.load(case["directory"] / "execution.npz", allow_pickle=False) as golden:
                if not {"native", "sham"}.issubset(golden.files):
                    raise ValueError("Archived comparison needs both native and sham trajectories")
                row["previous_trace_max_abs_m"] = {
                    k: float(np.abs(v - golden[k]).max()) for k, v in paths.items() if k in golden
                }
                if any(v > 1e-7 for v in row["previous_trace_max_abs_m"].values()):
                    save(output / "regression_failure.json", row)
                    raise RuntimeError("Shared FastWAM trajectory differs from the archived original")
        np.savez_compressed(output / f"{row['kind']}_{idx}.npz", **paths)
        save(output / "cases.json", rows)
        print(
            "Shared clip",
            futures.backend,
            row["kind"],
            idx,
            row["sham_pass"],
            row["effects_m"],
            flush=True,
        )
    save(output / "cases.json", rows)
