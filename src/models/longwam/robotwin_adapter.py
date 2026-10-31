"""Long-WAM RoboTwin 2.0 transforms (14-D qpos actions), following upstream runtime/robotwin.py."""

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from .episode import LongWAMEpisodeAdapter
from .transforms import ROBOTWIN_CAMERAS, robotwin_model_image


class LongWAMRobotWinAdapter(LongWAMEpisodeAdapter):
    benchmark = "robotwin"

    def __init__(self, model: Any, *, dataset_stats: Mapping[str, Any] | str | Path,
                 data_config: Mapping[str, Any] | None = None):
        super().__init__(model, dataset_stats=dataset_stats, data_config=data_config)
        if self.state_dim != 14 or self.action_dim != 14:
            raise ValueError("Long-WAM RoboTwin checkpoints use 14-D joint state and actions")
        if str(self.data_config.get("concat_multi_camera")) != "robotwin" or self.video_size != (384, 320):
            raise ValueError("Long-WAM RoboTwin expects the 384x320 'robotwin' camera mosaic")

    def environment_action_type(self, environment):
        if environment != "robotwin":
            raise ValueError("Expected environment='robotwin'")
        return "qpos"

    def _model_image(self, observation):
        return robotwin_model_image(observation.cameras, self.image_meta,
                                    device=self.model.device, dtype=self.model.torch_dtype)

    def _image_metadata(self):
        return {"camera_order": [camera for _, camera, _ in ROBOTWIN_CAMERAS],
                "camera_layout": "head_above_left_right_wrists", "resize": "two_stage_antialiased_bilinear"}

    def _raw_state(self, observation):
        try:
            joints = observation.state["joint_position"]
        except KeyError as exc:
            raise KeyError("RoboTwin state is missing 'joint_position'") from exc
        return np.asarray(joints, dtype=np.float32)

    def postprocess_actions(self, actions: Any, *, environment: str = "robotwin") -> np.ndarray:
        self.environment_action_type(environment)
        return self._denormalized_actions(actions)
