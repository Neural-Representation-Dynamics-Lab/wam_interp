"""Long-WAM LIBERO transforms, following upstream runtime/libero.py."""

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from .episode import LongWAMEpisodeAdapter
from .transforms import libero_model_image, quat2axisangle


class LongWAMLiberoAdapter(LongWAMEpisodeAdapter):
    benchmark = "libero"

    def __init__(self, model: Any, *, dataset_stats: Mapping[str, Any] | str | Path,
                 data_config: Mapping[str, Any] | None = None, binarize_gripper: bool = True):
        super().__init__(model, dataset_stats=dataset_stats, data_config=data_config)
        self.binarize_gripper = bool(binarize_gripper)
        self.num_cameras = int(self.processor_config.get("num_output_cameras", len(self.image_meta)))
        if self.num_cameras not in (1, 2) or len(self.image_meta) < self.num_cameras:
            raise ValueError("Long-WAM LIBERO supports one or two configured cameras")
        self.concatenation = str(self.data_config.get("concat_multi_camera", "horizontal"))
        if self.state_dim != 8 or self.action_dim != 7:
            raise ValueError("Long-WAM LIBERO checkpoints use 8-D state and 7-D actions")

    def _model_image(self, observation):
        image, _ = libero_model_image(observation.cameras, self.image_meta, self.num_cameras, self.concatenation,
                                      self.video_size, device=self.model.device, dtype=self.model.torch_dtype)
        return image

    def _image_metadata(self):
        return {"camera_order": ["agentview", "wrist"][:self.num_cameras], "rotation": "180",
                "camera_concat": self.concatenation if self.num_cameras == 2 else None}

    def _raw_state(self, observation):
        import torch

        state = observation.state
        try:
            position, quaternion, gripper = (state[key] for key in ("eef_position", "eef_quaternion", "gripper_position"))
        except KeyError as exc:
            raise KeyError(f"LIBERO state is missing {exc.args[0]!r}") from exc
        # benchmarks/libero/adapter.py policy_batch: float64 axis-angle, each part cast to float32.
        return torch.cat([
            torch.as_tensor(np.asarray(position), dtype=torch.float32),
            torch.as_tensor(quat2axisangle(np.array(quaternion, copy=True)), dtype=torch.float32),
            torch.as_tensor(np.asarray(gripper), dtype=torch.float32),
        ])

    def postprocess_actions(self, actions: Any, *, environment: str = "libero") -> np.ndarray:
        if environment != "libero":
            raise ValueError(f"LongWAMLiberoAdapter decodes environment='libero', got {environment!r}")
        action = self._denormalized_actions(actions)
        # The dataloader flips the sign of the gripper action to align with other datasets
        # (0 = close, 1 = open), so flip it back (-1 = open, +1 = close) before executing the action
        action[..., -1] = action[..., -1] * 2 - 1
        action[..., -1] = action[..., -1] * -1.0
        if self.binarize_gripper:
            action[..., -1] = np.sign(action[..., -1])
        return action
