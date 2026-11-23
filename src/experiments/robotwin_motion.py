"""Fixed-context RoboTwin stimuli with Cartesian displacements of the active arm."""

from contextlib import contextmanager

import numpy as np
from scipy.spatial.transform import Rotation

from .confirm_sources import tint_pair
from .simulator_video import dilate

CAMERAS = ("head", "left_wrist", "right_wrist")


def tcp(env):
    return np.array([env.robot.get_left_tcp_pose()[:3], env.robot.get_right_tcp_pose()[:3]], dtype=float)


class CartesianOffset:
    """Invert the active arm's FK for a world translation while preserving the native link orientation."""

    def __init__(self, env, side):
        self.robot = env.robot
        self.side = side
        self.entity = getattr(self.robot, side + "_entity")
        self.pin = self.entity.create_pinocchio_model()
        self.link = self.entity.links.index(getattr(self.robot, side + "_ee").child_link)
        self.indices = [
            self.entity.active_joints.index(j) for j in getattr(self.robot, side + "_arm_joints")
        ]
        self.command_slice = slice(0, 6) if side == "left" else slice(7, 13)
        self.rotation = self.entity.get_root_pose().to_transformation_matrix()[:3, :3]
        self.limits = self.entity.get_qlimits()[self.indices]

    def pose(self, q):
        self.pin.compute_forward_kinematics(q)
        mat = self.pin.get_link_pose(self.link).to_transformation_matrix()
        return mat[:3, 3].astype(float), Rotation.from_matrix(mat[:3, :3])

    def command(self, reference, dx):
        cmd = reference.copy()
        q = self.entity.get_qpos().copy()
        q[self.indices] = cmd[self.command_slice]
        pos, rot = self.pose(q)
        target = pos + self.rotation.T @ np.array([dx, 0, 0])
        for _ in range(12):
            here, orientation = self.pose(q)
            residual = np.r_[target - here, (rot * orientation.inv()).as_rotvec()]
            if np.linalg.norm(residual[:3]) < 0.0003 and np.linalg.norm(residual[3:]) < 0.003:
                break
            cols = []
            for j in self.indices:
                other = q.copy()
                other[j] += 0.001
                p, r = self.pose(other)
                cols.append(np.r_[p - here, (r * orientation.inv()).as_rotvec()] / 0.001)
            jac = np.stack(cols, axis=1)
            change = jac.T @ np.linalg.solve(jac @ jac.T + 1.0e-5 * np.eye(6), residual)
            q[self.indices] = np.clip(q[self.indices] + np.clip(change, -0.15, 0.15), *self.limits.T)
        cmd[self.command_slice] = q[self.indices]
        return cmd


def wall_masks(env):
    cameras = env.cameras
    items = (
        cameras.static_camera_list[cameras.static_camera_name.index("head_camera")],
        cameras.left_camera,
        cameras.right_camera,
    )
    return {
        k: ~dilate(~(cam.get_picture("Segmentation")[..., 1] == env.wall.actor.per_scene_id), 2)
        for k, cam in zip(CAMERAS, items)
    }


@contextmanager
def monitor(env):
    step = env.scene.step
    entities = {l.entity.name for s in ("left", "right") for l in getattr(env.robot, s + "_entity").links}
    props = [env.can, env.pot if hasattr(env, "pot") else env.basket]
    scene = {env.table.name, env.wall.actor.name, *(o.actor.name for o in props)}
    initial = np.stack([o.get_pose().p for o in props])
    log = {"substeps": 0, "contact_substeps": 0, "object_drift_m": 0.0}

    def measured():
        val = step()
        log["substeps"] += 1
        drift = np.linalg.norm(np.stack([o.get_pose().p for o in props]) - initial, axis=1).max()
        log["object_drift_m"] = max(log["object_drift_m"], float(drift))
        for c in env.scene.get_contacts():
            a, b = (body.entity.name for body in c.bodies)
            if (a in entities and b in scene) or (b in entities and a in scene):
                log["contact_substeps"] += 1
                break
        return val

    env.scene.step = measured
    try:
        yield log
    finally:
        env.scene.step = step


def copy_frames(obs):
    return {k: v.copy() for k, v in obs.cameras.items()}


def record(front, obs, commands, side, dx, stride=4):
    offset = CartesianOffset(front.env, side) if dx else None
    frames = [copy_frames(obs)]
    masks = [{k: np.zeros(v.shape[:2], bool) for k, v in frames[0].items()}]
    path = [tcp(front.env)]
    times = [0.0]
    with monitor(front.env) as log:
        for i, cmd in enumerate(commands, 1):
            actual = offset.command(cmd, dx * i / len(commands)) if offset else cmd
            obs, success = front.step(actual, "qpos")
            path.append(tcp(front.env))
            times.append(log["substeps"] * front.env.scene.get_timestep())
            if i % stride == 0:
                frames.append(copy_frames(obs))
                masks.append(wall_masks(front.env))
            if success:
                break
    variants = {key: [] for key in ("neutral", "warm", "cool")}
    changed = energy = pixels = leaks = 0
    for t, (fr, ma) in enumerate(zip(frames, masks)):
        variants["neutral"].append(fr)
        warm, cool = {}, {}
        for k in CAMERAS:
            warm[k], cool[k] = tint_pair(fr[k], ma[k], t)
            diff = warm[k].astype(float) - fr[k]
            hit = (diff != 0).any(-1)
            leaks += int((hit & ~ma[k]).sum())
            if t:
                changed += int(hit.sum())
                pixels += int(hit.size)
                energy += float(np.abs(diff).sum() / 3)
        variants["warm"].append(warm)
        variants["cool"].append(cool)
    log.update(
        controls=len(path) - 1,
        elapsed_seconds=times[-1],
        sample_times=times[::stride],
        changed_fraction=changed / max(1, pixels),
        pixel_mae=energy / max(1, pixels),
        mask_leak_pixels=leaks,
    )
    return variants, np.stack(path), dict(log)
