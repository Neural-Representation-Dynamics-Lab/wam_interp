"""Connect a loaded FastWAM adapter to an explicitly checkpointable environment."""

from copy import deepcopy
from time import perf_counter

import numpy as np

from .e1 import Branch
from .futures import predict_with_future


def _array(value):
    if hasattr(value, "detach"):
        return value.detach().float().cpu().numpy().copy()
    return np.array(value, copy=True)


class FastWAMChunkSession:
    """One prepared request, native transforms, and a caller-owned simulator.

    snapshot_fn/restore_fn must cover model, controller, simulator and RNG state.
    rollout_fn(commands) executes a prefix and returns a mapping with behavior,
    executed_commands, metadata and optional trajectories. It must record actual
    observed state, not infer it by integrating normalized model actions.
    This does not construct or download the model/simulator.
    """

    def __init__(self, adapter, request, *, snapshot_fn, restore_fn, rollout_fn,
                 execute_steps, environment="libero"):
        if request.condition != "idm" or request.compile_action_infer or request.seed is None:
            raise ValueError("Use eager IDM with an explicit fixed seed")
        if execute_steps <= 0:
            raise ValueError("execute_steps must be positive")
        self.adapter, self.request = adapter, request
        self.snapshot_fn, self.restore_fn, self.rollout_fn = snapshot_fn, restore_fn, rollout_fn
        self.execute_steps, self.environment = execute_steps, environment

    def snapshot(self):
        return self.snapshot_fn()

    def restore(self, state):
        self.restore_fn(deepcopy(state))

    def run(self, donor):
        import torch
        if torch.cuda.is_initialized():
            torch.cuda.synchronize()
        started = perf_counter()
        result, capture = predict_with_future(self.adapter, self.request, donor)
        if torch.cuda.is_initialized():
            torch.cuda.synchronize()
        latency = perf_counter() - started
        commands = _array(self.adapter.postprocess_actions(result.actions, environment=self.environment))
        if commands.ndim == 3 and commands.shape[0] == 1:
            commands = commands[0]
        if commands.ndim != 2 or not len(commands) or not np.isfinite(commands).all():
            raise ValueError("Invalid decoded command chunk")
        rollout = self.rollout_fn(commands[:self.execute_steps].copy())
        executed = _array(rollout["executed_commands"])
        if len(executed) > min(self.execute_steps, len(commands)):
            raise ValueError("Rollout executed beyond the requested prefix")
        return Branch(future=_array(capture["used"]), actions=_array(result.actions),
            commands=commands, executed_commands=executed, behavior=float(rollout["behavior"]),
            metadata={"inference": result.metadata, "rollout": rollout.get("metadata", {}),
                      "future_storage_dtype": "float32", "execute_steps": self.execute_steps,
                      "inference_with_capture_seconds": latency},
            trajectories=rollout.get("trajectories", {}))
