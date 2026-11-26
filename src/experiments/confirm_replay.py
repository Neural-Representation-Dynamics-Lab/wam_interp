"""Replay open-loop chunks from a fresh LIBERO reset and verify the recipient anchor each time."""

import hashlib
import random
from contextlib import contextmanager

import numpy as np
from wam_interp.evaluations.LIBERO.environment import (
    LIBERO_DUMMY_ACTION,
    create_libero_environment,
    reset_to_initial_state,
    step_libero,
)

from .simulator_libero import RobotContactCapture


def anchor_hash(obs):
    h = hashlib.sha256(obs.instruction.encode("utf8"))
    for group in (obs.cameras, obs.state):
        for k, v in sorted(group.items()):
            arr = np.ascontiguousarray(v)
            h.update(k.encode())
            h.update(str((arr.shape, arr.dtype)).encode())
            h.update(arr.tobytes())
    return h.hexdigest()


@contextmanager
def omit_unused_rgb(env):
    """Unused intermediate RGB does not enter the dynamics, readout or model."""
    native = env.sim.render

    def render(*args, **kw):
        if (
            kw.get("camera_name") in ("agentview", "robot0_eye_in_hand")
            and not kw.get("depth")
            and not kw.get("segmentation")
        ):
            return np.zeros((kw["height"], kw["width"], 3), dtype=np.uint8)
        return native(*args, **kw)

    env.sim.render = render
    try:
        yield
    finally:
        env.sim.render = native


def replay(task, state, commands, expected_anchor):
    random.seed(0)
    np.random.seed(0)
    env = create_libero_environment(task, 256, 0)
    meta = {"instruction": task.language, "metadata": {}}
    from .clip_study import start

    paths = {}
    contacts = {}

    def measure(sim, obs, chunk):
        path = [obs.state["eef_position"].copy()]
        with RobotContactCapture(sim) as contact, omit_unused_rgb(sim):
            contact.scene_geoms = set(range(sim.sim.model.ngeom)) - contact.robot_geoms
            contact.enabled = True
            for action in chunk:
                step = step_libero(sim, action, **meta)
                path.append(step.observation.state["eef_position"].copy())
                if step.done or step.success:
                    break
        return np.stack(path), len(contact.events)

    try:
        # Reuse the XML model but restore the simulator and controller on every arm.
        env.env.hard_reset = False
        native_reset = False

        def native_arm(key, chunk):
            with start(None, task, state, 0) as (fresh, _, original, _):
                if anchor_hash(original) != expected_anchor:
                    raise ValueError("Fresh native reset differs from the recorded recipient anchor")
                paths[key], contacts[key] = measure(fresh, original, chunk)

        for key, chunk in commands.items():
            if native_reset:
                native_arm(key, chunk)
                continue
            random.seed(0)
            np.random.seed(0)
            env.seed(0)
            reset_to_initial_state(env, state, **meta)
            with omit_unused_rgb(env):
                for _ in range(29):
                    step = step_libero(env, LIBERO_DUMMY_ACTION, **meta)
                    if step.done or step.success:
                        raise RuntimeError("Warmup terminated early")
            step = step_libero(env, LIBERO_DUMMY_ACTION, **meta)
            obs = step.observation
            if step.done or step.success or anchor_hash(obs) != expected_anchor:
                # Some rooms rebuild randomized assets at reset. Recreate the native environment.
                native_reset = True
                native_arm(key, chunk)
                continue
            paths[key], contacts[key] = measure(env, obs, chunk)
        return paths, contacts
    finally:
        env.close()
