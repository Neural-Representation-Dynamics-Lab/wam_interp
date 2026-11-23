"""Run model stages of the paired motion and appearance experiment on LIBERO."""

import argparse
import json
from pathlib import Path

import numpy as np
from wam_interp.evaluations.LIBERO.environment import initial_states_for_task
from wam_interp.evaluations.LIBERO.runner import _get_suite
from wam_interp.experiments.clip_study import start
from wam_interp.experiments.future_adapter import FutureAdapter


def load(backend):
    if backend == "fastwam":
        from wam_interp.experiments.libero_pilot import load_pilot

        pilot = load_pilot(json.loads(Path("configs/experiments/e1_libero.json").read_text()))
        adapter = pilot.adapter
        pilot.close()
    else:
        from wam_interp.models.longwam.libero_adapter import LongWAMLiberoAdapter
        from wam_interp.models.longwam.loader import load_longwam

        root = Path("checkpoints/longwam/libero_idm")
        model = load_longwam(root / "config.yaml", root / "model.pt", device="cuda", dtype="bfloat16")
        adapter = LongWAMLiberoAdapter(model, dataset_stats=root / "dataset_stats.json")
    return FutureAdapter(adapter)


def commands(api, prediction):
    import torch

    result = api.adapter.postprocess_actions(prediction.actions, environment="libero")
    result = torch.as_tensor(result).detach().cpu().float().numpy().reshape(-1, 7)
    if result.shape != (32, 7) or not np.isfinite(result).all():
        raise ValueError("Expected 32 finite LIBERO commands")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("references", "features", "behavior", "commands", "calibrate"))
    parser.add_argument("--backend", choices=("fastwam", "longwam"), required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--wait-sources", action="store_true")
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    suite = _get_suite("libero_10")
    api = load(args.backend)
    for task_id in manifest["tasks"]:
        task = suite.get_task(task_id)
        states = initial_states_for_task(suite, task_id)
        for idx in manifest["states"][str(task_id)]:
            for seed in manifest["seeds"]:
                folder = args.out / f"task_{task_id}/state_{idx}/seed_{seed}"
                folder.mkdir(parents=True, exist_ok=True)
                if args.stage == "references":
                    dest = folder / f"{args.backend}_reference.npy"
                    if dest.exists():
                        continue
                    with start(api, task, states[idx], idx, task_id=task_id, seed=seed) as (
                        _,
                        _,
                        _,
                        request,
                    ):
                        values = commands(api, api.adapter.predict_action(request))
                    temp = dest.with_suffix(".tmp.npy")
                    np.save(temp, values)
                    temp.replace(dest)
                else:
                    if args.wait_sources:
                        import time

                        until = time.monotonic() + 1800
                        while not (folder / "fastwam_source.json").exists():
                            if time.monotonic() > until:
                                raise TimeoutError(f"No source at {folder}")
                            time.sleep(2)
                    from wam_interp.experiments.confirmation import run_case

                    if args.stage == "calibrate":
                        for stage in ("features", "behavior"):
                            run_case(
                                api,
                                stage,
                                args.backend,
                                task,
                                task_id,
                                states[idx],
                                idx,
                                seed,
                                folder,
                                manifest,
                            )
                        if (
                            args.backend == "longwam"
                            and idx == manifest["states"][str(task_id)][0]
                            and seed == manifest["seeds"][0]
                        ):
                            from wam_interp.experiments.confirm_sources import (
                                make_sources,
                            )

                            with start(api, task, states[idx], idx, task_id=task_id, seed=seed) as (
                                _,
                                _,
                                _,
                                req,
                            ):
                                np.save(
                                    folder / "longwam_reference.npy",
                                    commands(api, api.adapter.predict_action(req)),
                                )
                            make_sources(task, task_id, states[idx], idx, seed, folder, "longwam", True)
                            for stage in ("features", "behavior"):
                                run_case(
                                    api,
                                    stage,
                                    args.backend,
                                    task,
                                    task_id,
                                    states[idx],
                                    idx,
                                    seed,
                                    folder,
                                    manifest,
                                    source="longwam",
                                )
                    else:
                        run_case(
                            api,
                            args.stage,
                            args.backend,
                            task,
                            task_id,
                            states[idx],
                            idx,
                            seed,
                            folder,
                            manifest,
                        )
                print(args.stage, args.backend, task_id, idx, seed, flush=True)


if __name__ == "__main__":
    main()
