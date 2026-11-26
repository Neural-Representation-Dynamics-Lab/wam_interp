"""Render paired LIBERO sources from native commands and geometry-ID masks."""

import hashlib
from contextlib import AbstractContextManager
from pathlib import Path

import numpy as np
from wam_interp.evaluations.LIBERO.environment import step_libero

from .clip_study import save, start
from .simulator_libero import RobotContactCapture
from .simulator_video import dilate, paired_camera_clip

CAMERAS = ("agentview", "robot0_eye_in_hand")
MOTION = ("left", "right")
TINTS = ("neutral", "warm", "cool")


def tint_pair(rgb, mask, phase):
    """Opposite approximately isoluminant sRGB-channel changes with symmetric gamut protection."""
    rgb = np.asarray(rgb)
    direction = np.array([21.0, -4.0, -22.0])
    headroom = np.minimum(rgb, 255 - rgb).astype(float)
    scale = np.minimum(1.0, (headroom / np.abs(direction)).min(-1))
    change = np.rint(direction * (scale * mask * min(1.0, phase / 3.0))[..., None]).astype(np.int16)
    return tuple((rgb.astype(np.int16) + sign * change).astype(np.uint8) for sign in (1, -1))


class BackgroundSensors(AbstractContextManager):
    def __init__(self, env, stabilize):
        import mujoco
        from robosuite import macros
        from robosuite.utils.mjcf_utils import IMAGE_CONVENTION_MAPPING

        self.sim = env.sim
        self.original = self.sim.render
        model = self.sim.model
        self.ids = [
            i
            for i in range(model.ngeom)
            if model.geom_bodyid[i] == 0
            and ("wall" in (model.geom_id2name(i) or "").lower() or model.geom_id2name(i) == "floor")
        ]
        # Some LIBERO rooms put their backdrop in an unnamed, noncolliding mesh.
        for i in range(model.ngeom):
            if model.geom_bodyid[i] or model.geom_contype[i] or model.geom_conaffinity[i]:
                continue
            if model.geom_type[i] == int(mujoco.mjtGeom.mjGEOM_MESH):
                name = mujoco.mj_id2name(model._model, mujoco.mjtObj.mjOBJ_MESH, model.geom_dataid[i])
                if name in {"kitchen_background_vis", "living_room_vis"}:
                    self.ids.append(i)
        self.ids = sorted(set(self.ids))
        self.geom = int(mujoco.mjtObj.mjOBJ_GEOM)
        self.names = [
            model.geom_id2name(i)
            or mujoco.mj_id2name(model._model, mujoco.mjtObj.mjOBJ_MESH, model.geom_dataid[i])
            for i in self.ids
        ]
        self.convention = IMAGE_CONVENTION_MAPPING[macros.IMAGE_CONVENTION]
        self.capture = False
        self.stabilize = stabilize
        self.views = {}
        self.camera = model.camera_name2id(CAMERAS[1])
        data = self.sim.data
        self.position = data.cam_xpos[self.camera].copy()
        self.matrix = data.cam_xmat[self.camera].copy()

    def __enter__(self):
        self.sim.render = self.render
        return self

    def __exit__(self, *exc):
        self.sim.render = self.original

    def render(self, *args, **kwargs):
        image = self.original(*args, **kwargs)
        cam = kwargs.get("camera_name")
        if not self.capture or cam not in CAMERAS or kwargs.get("depth") or kwargs.get("segmentation"):
            return image

        def read(rgb):
            seg = self.original(*args, **{**kwargs, "segmentation": True, "depth": False})
            mask = (seg[..., 0] == self.geom) & np.isin(seg[..., 1], self.ids)
            # Exclude geometry edges before resizing the image.
            mask = ~dilate(~mask, 2)
            return rgb[:: self.convention].copy(), mask[:: self.convention].copy()

        self.views[cam] = read(image)
        if self.stabilize and cam == CAMERAS[1]:
            pos, mat = self.sim.data.cam_xpos[self.camera], self.sim.data.cam_xmat[self.camera]
            old_pos, old_mat = pos.copy(), mat.copy()
            try:
                pos[:] = self.position
                mat[:] = self.matrix
                self.views["fixed_wrist"] = read(self.original(*args, **kwargs))
            finally:
                pos[:] = old_pos
                mat[:] = old_mat
        return image


def render_case(task, task_id, state, index, seed, commands, bias, stabilize=False):
    with start(None, task, state, index, task_id=task_id, seed=seed) as (env, meta, obs, _):
        ids = list(env.env.obj_body_id.values())
        poses = [env.sim.data.body_xpos[ids].copy()]
        robot = [obs.state["eef_position"].copy()]
        frames = {CAMERAS[0]: [obs.cameras["agentview"]], CAMERAS[1]: [obs.cameras["wrist"]]}
        if stabilize:
            frames["fixed_wrist"] = [obs.cameras["wrist"]]
        masks = {k: [np.zeros(v[0].shape[:2], bool)] for k, v in frames.items()}
        with BackgroundSensors(env, stabilize) as sensor, RobotContactCapture(env) as contacts:
            contacts.scene_geoms = set(range(env.sim.model.ngeom)) - contacts.robot_geoms
            contacts.enabled = True
            for step, action in enumerate(commands, 1):
                sensor.capture = step % 4 == 0
                cmd = action.copy()
                cmd[0] = np.clip(cmd[0] + bias, -1.0, 1.0)
                result = step_libero(env, cmd, **meta)
                if result.done or result.success:
                    raise ValueError("Early termination inside source window")
                poses.append(env.sim.data.body_xpos[ids].copy())
                robot.append(result.observation.state["eef_position"].copy())
                if sensor.capture:
                    for key, view_frames in frames.items():
                        rgb, mask = sensor.views[key]
                        if key in CAMERAS:
                            field = "agentview" if key == CAMERAS[0] else "wrist"
                            if not np.array_equal(rgb, result.observation.cameras[field]):
                                raise ValueError("Camera capture does not match the observation")
                        view_frames.append(rgb)
                        masks[key].append(mask)

        def pack(views):
            return paired_camera_clip(frames[CAMERAS[0]], views)

        variants = {"neutral": pack(frames[CAMERAS[1]])}
        for tint_index, name in enumerate(("warm", "cool")):
            views = [
                [
                    tint_pair(rgb, mask, t)[tint_index]
                    for t, (rgb, mask) in enumerate(zip(frames[k], masks[k]))
                ]
                for k in CAMERAS
            ]
            variants[name] = paired_camera_clip(*views)
        if stabilize:
            variants["fixed"] = pack(frames["fixed_wrist"])
        support = (
            paired_camera_clip(
                *[
                    [np.repeat(mask[..., None], 3, axis=-1).astype("uint8") * 255 for mask in masks[k]]
                    for k in CAMERAS
                ]
            )[..., 0]
            > 0
        )
        changed = np.any(variants["warm"] != variants["neutral"], axis=-1)
        leak = int(np.count_nonzero(changed & ~np.stack([dilate(s, 2) for s in support])))
        rgb_delta = variants["warm"].astype(float) - variants["neutral"]
        checks = {
            "background_geometries": sensor.names,
            "contact_substeps": len(contacts.events),
            "object_drift_m": float(np.linalg.norm(poses - poses[0], axis=-1).max()),
            "mask_leak_pixels": leak,
            "changed_fraction": float(changed[1:].mean()),
            "pixel_mae": float(np.abs(rgb_delta[1:]).mean()),
            "frame_zero_exact": not bool(changed[0].any()),
        }
        return variants, np.array(robot), checks, support


def make_sources(task, task_id, state, index, seed, folder, backend="fastwam", stabilize=False):
    folder = Path(folder)
    marker = folder / f"{backend}_source.json"
    if marker.exists():
        return
    commands = np.load(folder / f"{backend}_reference.npy", allow_pickle=False)
    base, reference_path, check, _ = render_case(
        task, task_id, state, index, seed, commands, 0.0, stabilize
    )
    report = {"passed": False, "reference": check, "attempts": [], "instruction": task.language}
    for bias in (0.05, 0.09, 0.15):
        data = {"sham": base["neutral"], "source_reference_path": reference_path}
        if stabilize:
            data["fixed_sham"] = base["fixed"]
        successful = check["contact_substeps"] == 0 and check["object_drift_m"] <= 0.001
        attempt = {}
        for name, sign in zip(MOTION, (-1, 1)):
            variants, path, checks, mask = render_case(
                task, task_id, state, index, seed, commands, sign * bias, stabilize
            )
            diff = path[-1] - reference_path[-1]
            checks["endpoint_delta_m"] = diff.tolist()
            ok = (
                0.02 <= sign * diff[0] <= 0.04
                and np.linalg.norm(diff[1:]) <= 0.01
                and checks["object_drift_m"] <= 0.001
                and checks["contact_substeps"] == 0
                and checks["changed_fraction"] >= 0.005
                and checks["pixel_mae"] >= 0.10
                and checks["mask_leak_pixels"] == 0
                and checks["frame_zero_exact"]
                and np.array_equal(variants["neutral"][0], base["neutral"][0])
            )
            successful &= bool(ok)
            attempt[name] = checks
            data.update({f"{name}_{tint}": rgb for tint, rgb in variants.items()})
            data[f"{name}_mask"] = mask
            data[f"{name}_source_path"] = path
        report["attempts"].append({"bias": bias, "arms": attempt})
        if successful:
            temp = folder / f"{backend}_source.tmp.npz"
            dest = folder / f"{backend}_source.npz"
            np.savez_compressed(temp, **data)
            temp.replace(dest)
            report.update(passed=True, sha256=hashlib.sha256(dest.read_bytes()).hexdigest())
            break
    save(marker, report)
