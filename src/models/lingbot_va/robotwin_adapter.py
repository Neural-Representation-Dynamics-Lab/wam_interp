"""LingBot-VA RoboTwin EE commands and stateful observation cadence.

Ported from evaluation/robotwin/eval_polict_client_openpi.py at 7c6ffa9.
"""

import numpy as np

from wam_interp.types import PreparedObservation
from .adapter import LingBotVAAdapter


def compose_relative_pose(relative, initial):
    """Upstream: translation addition and init_R * relative_R, XYZW quaternions."""
    result = np.asarray(relative, dtype=np.float64).copy()
    initial = np.asarray(initial, dtype=np.float64)
    if result.shape != (16,) or initial.shape != (16,):
        raise ValueError("RoboTwin end-effector poses must have 16 values")
    for offset in (0, 8):
        result[offset:offset+3] += initial[offset:offset+3]
        a = initial[offset+3:offset+7]
        b = result[offset+3:offset+7].copy()
        if np.linalg.norm(a) < 1e-12 or np.linalg.norm(b) < 1e-12:
            raise ValueError("Cannot execute a zero-norm end-effector quaternion")
        a = a / np.linalg.norm(a)
        b = b / np.linalg.norm(b)
        xyz = a[3] * b[:3] + b[3] * a[:3] + np.cross(a[:3], b[:3])
        w = a[3] * b[3] - np.dot(a[:3], b[:3])
        quat = np.concatenate([xyz, [w]])
        result[offset+3:offset+7] = quat / np.linalg.norm(quat)
    return result


class LingBotVARobotWinAdapter(LingBotVAAdapter):
    requires_full_action_chunk = True

    def __init__(self, model):
        super().__init__(model)
        if model.config.env_type != "robotwin_tshape" or model.config.action_per_frame != 16:
            raise ValueError("Use the released LingBot RoboTwin config/checkpoint")
        self._in_episode = False

    def environment_action_type(self, environment):
        if environment != "robotwin":
            raise ValueError("Expected environment='robotwin'")
        return "ee"

    def begin_episode(self, observation):
        self.reset(observation.instruction)
        self._initial_pose = np.concatenate([
            observation.state["left_endpose"], [observation.state["left_gripper"]],
            observation.state["right_endpose"], [observation.state["right_gripper"]],
        ]).astype(np.float64)
        if self._initial_pose.shape != (16,):
            raise ValueError("RoboTwin episode needs two XYZ+XYZW+gripper initial poses")
        self._last_chunk = None
        self._frames = []
        self._observed_steps = 0
        self._expected_steps = 0
        self._in_episode = True

    def prepare_observation(self, observation):
        import torch
        import torch.nn.functional as F
        cfg = self.model.config
        views = []
        for index, name in enumerate(("head", "left_wrist", "right_wrist")):
            image = np.asarray(observation.cameras[name])
            if image.ndim != 3 or image.shape[-1] != 3 or image.dtype != np.uint8:
                raise ValueError(f"RoboTwin camera {name} must contain uint8 RGB")
            image = torch.from_numpy(np.ascontiguousarray(image)).permute(2, 0, 1).float()[None]
            size = (cfg.height, cfg.width) if index == 0 else (cfg.height // 2, cfg.width // 2)
            views.append(F.interpolate(image, size=size, mode="bilinear", align_corners=False)[0] / 255.0 * 2.0 - 1.0)
        return PreparedObservation(tuple(views), None, observation.instruction,
            {**observation.metadata, "camera_order": ["head", "left_wrist", "right_wrist"],
             "camera_layout": "latent_wrists_above_head"})

    def observe_step(self, observation, action):
        self._observed_steps += 1
        stride = self.model.config.action_per_frame // 4
        if self._observed_steps % stride == 0:
            self._frames.append(self.prepare_observation(observation).image)

    def predict_action(self, request, recorder=None):
        if not self._in_episode:
            raise RuntimeError("Call begin_episode() before LingBot RoboTwin inference")
        # Steps observed since the last chunk mean it was executed; none means a repeat prediction.
        if self._last_chunk is not None and self._observed_steps:
            stride = self.model.config.action_per_frame // 4
            if self._observed_steps != self._expected_steps or len(self._frames) != self._expected_steps // stride:
                raise ValueError("LingBot RoboTwin requires the complete chunk and every fourth observed frame")
            self.update(self._frames, self._last_chunk)
            self._frames = []
            self._observed_steps = 0
        return super().predict_action(request, recorder=recorder)

    def postprocess_actions(self, actions, *, environment):
        self.environment_action_type(environment)
        first = self.model.frame_st_id == 0
        self._last_chunk = self.model.postprocess_action(actions)
        if self._last_chunk.shape[0] != 16:
            raise ValueError("LingBot RoboTwin checkpoint must decode 16 EE channels")
        commands = self._last_chunk.transpose(1, 2, 0).reshape(-1, 16)
        if first:
            commands = commands[self.model.config.action_per_frame:]
        self._expected_steps = len(commands)
        return np.stack([compose_relative_pose(command, self._initial_pose) for command in commands])
