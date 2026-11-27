"""Render and validate matched obstacle conditions at native LIBERO sensor times."""

from contextlib import contextmanager
from pathlib import Path

import numpy as np
from wam_interp.evaluations.LIBERO.environment import step_libero

from .clip_study import save, start
from .confirm_replay import anchor_hash
from .obstacle import placement
from .simulator_video import paired_camera_clip

PLACEMENTS = ("minus", "plus", "off_minus", "off_plus")
READOUT = 16


@contextmanager
def settle_without_barrier(env):
    """Let the released initial objects fall into place before inserting the solid."""
    model = env.sim.model
    i = model.geom_name2id("e1_barrier")
    masks = int(model.geom_contype[i]), int(model.geom_conaffinity[i])
    model.geom_contype[i] = model.geom_conaffinity[i] = 0
    try:
        yield
    finally:
        model.geom_contype[i], model.geom_conaffinity[i] = masks


def scene(api, task, state, idx, task_id, seed, box):
    return start(
        api,
        task,
        state,
        idx,
        task_id=task_id,
        seed=seed,
        xml_processor=box.xml if box else None,
        warmup_context=settle_without_barrier if box else None,
    )


def rollout(
    env,
    meta,
    obs,
    actions,
    *,
    frames=False,
    barrier=False,
    robot_only=False,
    stride=4,
    on_step=None,
):
    """Log any obstacle contact at physics substeps, including contact with other objects."""
    sim = env.sim
    positions = [obs.state["eef_position"].copy()]
    qpos = [sim.data.qpos.copy()]
    cameras = {k: [v.copy()] for k, v in obs.cameras.items()} if frames else {}
    contacts = []
    original = sim.step
    geom = sim.model.geom_name2id("e1_barrier") if barrier else None
    robots = None
    if robot_only:
        from .simulator_libero import RobotContactCapture

        robots = RobotContactCapture(env).robot_geoms
    step_number = 0

    def record_step(*args, **kwargs):
        result = original(*args, **kwargs)
        for contact in sim.data.contact[: sim.data.ncon]:
            pair = {int(contact.geom1), int(contact.geom2)}
            if geom in pair and (robots is None or robots.intersection(pair)):
                contacts.append(step_number)
                break
        return result

    if barrier:
        sim.step = record_step
    try:
        for step_number, cmd in enumerate(actions, 1):
            result = step_libero(env, cmd, **meta)
            current = result.observation
            if on_step is not None:
                on_step(result, cmd)
            positions.append(current.state["eef_position"].copy())
            qpos.append(sim.data.qpos.copy())
            if frames and step_number % stride == 0:
                for k, images in cameras.items():
                    images.append(current.cameras[k].copy())
            if result.done or result.success:
                break
    finally:
        if barrier:
            sim.step = original
    return (
        np.stack(positions),
        np.stack(qpos),
        {k: np.stack(v) for k, v in cameras.items()},
        sorted(set(contacts)),
    )


def render_sources(task, task_id, state, idx, seed, actions, out, *, station=28):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    info = {
        "task": task_id,
        "instruction": task.language,
        "placement_station": station,
        "state": idx,
        "seed": seed,
        "passed": False,
        "placements": {},
    }

    with start(None, task, state, idx, task_id=task_id) as (env, meta, obs, _):
        info["clear_anchor"] = anchor_hash(obs)
        native_path, qpos, cams, _ = rollout(env, meta, obs, actions, frames=True)
    if len(native_path) != 33:
        info["reason"] = "short reference"
        save(out / "sources.json", info)
        return
    raw = {
        "clear": (
            native_path,
            qpos,
            paired_camera_clip(cams["agentview"], cams["wrist"]),
        )
    }
    first_contact = 33
    for name in PLACEMENTS:
        try:
            box = placement(
                native_path,
                -1 if name.endswith("minus") else 1,
                off_route=name.startswith("off"),
                station=station,
            )
        except ValueError as exc:
            info["reason"] = str(exc)
            break
        with scene(None, task, state, idx, task_id, seed, box) as (env, meta, obs, _):
            anchor = anchor_hash(obs)
            other, physics, cams, contacts = rollout(
                env, meta, obs, actions, frames=True, barrier=True
            )
        with scene(None, task, state, idx, task_id, seed, box) as (env, meta, obs, _):
            _, _, _, robot_contacts = rollout(
                env, meta, obs, actions, barrier=True, robot_only=True
            )
        info["placements"][name] = {
            "geometry": box.as_dict(),
            "anchor": anchor,
            "full_reference_contact": robot_contacts,
            "all_contacts": contacts,
            "passed": False,
        }
        if len(other) != 33:
            info["placements"][name]["reason"] = "short barrier reference"
            continue
        if contacts:
            first_contact = min(first_contact, contacts[0])
        raw[name] = other, physics, paired_camera_clip(cams["agentview"], cams["wrist"])
    # Last real sampled view before any contact, shared across every future.
    # Repeat that view through the final scheduled sample; no post-collision pixels enter the donor.
    prefix = 4 * ((first_contact - 1) // 4)
    info.update(
        depicted_native_prefix_steps=prefix, tail="repeat_last_precontact_sample"
    )
    data = {"commands": actions, "native_path": native_path}
    for name, (path, physics, clip) in raw.items():
        data[name] = hold_after(clip, prefix // 4)
        data[name + "_path"] = hold_after(path, prefix)
        data[name + "_qpos"] = hold_after(physics, prefix)
        if name == "clear":
            continue
        row = info["placements"][name]
        delta = (data[name] != data["clear"]).any(-1).sum((1, 2))
        contacts = row["full_reference_contact"]
        blocks = (
            (not row["all_contacts"])
            if row["geometry"]["off_route"]
            else bool(contacts) and contacts[0] > READOUT
        )
        row.update(
            changed_pixels=delta.tolist(),
            max_qpos_delta=float(
                np.abs(data[name + "_qpos"] - data["clear_qpos"]).max()
            ),
            max_eef_delta_m=float(
                np.abs(data[name + "_path"] - data["clear_path"]).max()
            ),
        )
        row["passed"] = bool(
            blocks
            and prefix >= 24
            and row["max_qpos_delta"] <= 1e-8
            and row["max_eef_delta_m"] <= 1e-8
            and delta.min() >= 100
        )
    info["passed"] = all(
        info["placements"].get(k, {}).get("passed") for k in PLACEMENTS
    )
    np.savez_compressed(out / "sources.npz", **data)
    save(out / "sources.json", info)


def hold_after(array, index):
    if not 0 <= index < len(array):
        raise ValueError("Clip lacks the registered precontact sample")
    held = array.copy()
    held[index + 1 :] = held[index]
    return held


def recipient_clip(source, anchor_clip):
    if (
        source.shape != anchor_clip.shape
        or source.dtype != np.uint8
        or anchor_clip.dtype != np.uint8
    ):
        raise ValueError("Mismatched camera clips")
    clip = source.copy()
    clip[0] = anchor_clip[0]
    return clip
