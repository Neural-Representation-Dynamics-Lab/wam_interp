"""Open-loop action change against the share of a frame masked, on a few ID start states.

Used to pick the closed-loop partial-mask levels. Run from the repository root on
feat/e7-partial-mask (c1fb12e) in the LIBERO-Plus environment.
"""
import json
import sys
from argparse import Namespace
from pathlib import Path


sys.path.insert(0, "scripts")
import e7_future_ablation as e7  # noqa: E402
import wam_interp.interventions.future_conditioning as fc  # noqa: E402
from wam_interp.evaluations._libero_rollout import _request_for_call  # noqa: E402
from wam_interp.evaluations.LIBERO import LiberoEvalConfig  # noqa: E402
from wam_interp.evaluations.LIBERO.environment import (  # noqa: E402
    LIBERO_DUMMY_ACTION, create_libero_environment, reset_to_initial_state, step_libero,
)

OUT = Path(sys.argv[1])
PERCENTS = list(range(5, 100, 5)) + [96, 98]
fc.PARTIAL_MASKS = tuple(f"{name}_{p}" for name in ("current_mask", "future1_mask") for p in PERCENTS)

args = Namespace(
    task_config="libero_fasterwam_2cam224_1e-4",
    checkpoint=Path(sys.argv[2]), dataset_stats=Path(sys.argv[3]))
adapter, latent_frames = e7.load_adapter(args)
ablation = fc.FasterWAMFutureAblation(adapter.model, latent_frames)
config = LiberoEvalConfig(task_suite_name="libero_10", **e7.PROTOCOL)
tasks = e7.original_tasks()

rows = []
for task_id in range(0, 10, 2):
    task, states = tasks[task_id]
    env = create_libero_environment(task, e7.PROTOCOL["camera_resolution"], seed=0)
    obs = reset_to_initial_state(env, states[0], instruction=task.language, metadata={})
    for _ in range(e7.PROTOCOL["num_steps_wait"]):
        obs = step_libero(env, LIBERO_DUMMY_ACTION, instruction=task.language, metadata={}).observation
    env.close()
    request = _request_for_call(config, adapter, adapter.prepare_observation(obs), episode_seed=7, inference_call=0)
    ablation.condition = "normal"
    reference = adapter.predict_action(request).actions
    for name in ("future1_mask", "current_mask"):
        for p in [*PERCENTS, 100]:
            ablation.condition = name if p == 100 else f"{name}_{p}"
            delta = (adapter.predict_action(request).actions - reference).abs().float()
            rows.append({"task_id": task_id, "mask": name, "percent": p,
                         "executed": float(delta[:10].mean()), "later": float(delta[10:].mean())})
    print("task", task_id, "done", flush=True)

OUT.write_text(json.dumps(rows, indent=1))
for name in ("future1_mask", "current_mask"):
    for p in [*PERCENTS, 100]:
        sel = [r for r in rows if r["mask"] == name and r["percent"] == p]
        print(name, p, round(sum(r["executed"] for r in sel) / len(sel), 4),
              round(sum(r["later"] for r in sel) / len(sel), 4))
