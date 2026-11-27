"""Open-loop action change when masking one position of future frame 1, or all of it but one.

Same five ID start states as sweep_open_loop.py. Run from the repository root on
feat/e7-partial-mask (c1fb12e) in the LIBERO-Plus environment.
"""
import json
import sys
from argparse import Namespace
from pathlib import Path

import torch

sys.path.insert(0, "scripts")
import e7_future_ablation as e7  # noqa: E402
import wam_interp.interventions.future_conditioning as fc  # noqa: E402
from wam_interp.evaluations._libero_rollout import _request_for_call  # noqa: E402
from wam_interp.evaluations.LIBERO import LiberoEvalConfig  # noqa: E402
from wam_interp.evaluations.LIBERO.environment import (  # noqa: E402
    LIBERO_DUMMY_ACTION, create_libero_environment, reset_to_initial_state, step_libero,
)

OUT = Path(sys.argv[1])
args = Namespace(task_config="libero_fasterwam_2cam224_1e-4", checkpoint=Path(sys.argv[2]),
                 dataset_stats=Path(sys.argv[3]))
adapter, latent_frames = e7.load_adapter(args)
ablation = fc.FasterWAMFutureAblation(adapter.model, latent_frames)
config = LiberoEvalConfig(task_suite_name="libero_10", **e7.PROTOCOL)
tasks = e7.original_tasks()
selected = {}
ablation.columns = lambda condition, video_seq_len: selected.get(condition)

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
    frame = 98
    for position in range(frame):
        others = torch.tensor([frame + q for q in range(frame) if q != position])
        for mode, columns in (("drop_one", torch.tensor([frame + position])), ("keep_one", others)):
            selected["probe"] = columns
            ablation.condition = "probe"
            delta = (adapter.predict_action(request).actions - reference).abs().float()
            rows.append({"task_id": task_id, "position": position, "mode": mode,
                         "executed": float(delta[:10].mean()), "later": float(delta[10:].mean())})
    print("task", task_id, "done", flush=True)

OUT.write_text(json.dumps(rows, indent=1))
for mode in ("drop_one", "keep_one"):
    means = sorted(((sum(r["executed"] for r in rows if r["mode"] == mode and r["position"] == q) / 5, q)
                    for q in range(98)), reverse=(mode == "drop_one"))
    print(mode, "top 8 by executed change" if mode == "drop_one" else "lowest 8 (most rescue)",
          [(q, round(v, 4)) for v, q in means[:8]])
    print(mode, "median", round(sorted(v for v, _ in means)[49], 4))
