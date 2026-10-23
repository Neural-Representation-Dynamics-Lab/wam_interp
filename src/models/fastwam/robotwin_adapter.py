"""FastWAM-family RoboTwin qpos transforms, matching upstream deploy_policy."""

import numpy as np

from wam_interp.types import PreparedObservation
from .adapter import FastWAMAdapter
from .libero_adapter import (
    _plain_mapping, _load_stats, _normalization_mode, _normalize_field,
    _denormalize_field, _as_numpy, _model_device, _model_dtype,
)


class RobotWinJointTransforms:
    def __init__(self, model, data_config, dataset_stats):
        self.model = model
        self.data = _plain_mapping(data_config)
        self.stats = _load_stats(dataset_stats)
        self.processor = self.data["processor"]
        if self.processor.get("action_state_transforms") not in (None, [], {}):
            raise NotImplementedError("RoboTwin action_state_transforms must be implemented explicitly")
        self.state_key = self.data["shape_meta"]["state"][0]["key"]
        self.action_key = self.data["shape_meta"]["action"][0]["key"]
        if self.processor["proprio_output_dim"] != 14 or self.processor["action_output_dim"] != 14:
            raise ValueError("RoboTwin joint checkpoints require 14 state/action values")

    def prepare(self, observation):
        import torch
        from PIL import Image
        views = []
        for key, size in (("head", (320, 256)), ("left_wrist", (160, 128)), ("right_wrist", (160, 128))):
            image = np.asarray(observation.cameras[key])
            if image.ndim != 3 or image.shape[-1] != 3 or image.dtype != np.uint8:
                raise ValueError(f"RoboTwin camera {key} must contain uint8 RGB pixels")
            views.append(np.asarray(Image.fromarray(image).resize(size, Image.Resampling.BILINEAR)))
        montage = np.concatenate([views[0], np.concatenate(views[1:], axis=1)], axis=0)
        image = torch.from_numpy(montage.copy()).permute(2, 0, 1).unsqueeze(0)
        image = image.to(device=_model_device(self.model), dtype=_model_dtype(self.model)) * (2.0/255.0) - 1.0
        state = np.asarray(observation.state["joint_position"], dtype=np.float32)
        if state.shape != (14,):
            raise ValueError(f"Expected 14-D joint state, got {state.shape}")
        normalized = _normalize_field(state, self.stats["state"][self.state_key],
            mode=_normalization_mode(self.processor, "state", self.state_key), use_stepwise=False,
            field_name=f"state/{self.state_key}")
        proprio = torch.as_tensor(normalized, device=_model_device(self.model), dtype=torch.float32)[None]
        return PreparedObservation(image, proprio, observation.instruction,
            {**observation.metadata, "camera_order": ["head", "left_wrist", "right_wrist"],
             "camera_layout": "head_above_wrists", "image_shape": list(image.shape)})

    def decode(self, actions):
        values = _as_numpy(actions).astype(np.float32)
        if values.ndim == 3 and values.shape[0] == 1:
            values = values[0]
        if values.ndim != 2 or values.shape[1] != 14:
            raise ValueError(f"Expected normalized qpos actions [T,14], got {values.shape}")
        return _denormalize_field(values, self.stats["action"][self.action_key],
            mode=_normalization_mode(self.processor, "action", self.action_key),
            use_stepwise=bool(self.processor.get("use_stepwise_action_norm", False)),
            field_name=f"action/{self.action_key}")


class FastWAMRobotWinAdapter(FastWAMAdapter):
    def __init__(self, model, *, data_config, dataset_stats):
        super().__init__(model)
        self.transforms = RobotWinJointTransforms(model, data_config, dataset_stats)

    def inference_defaults(self):
        return {"condition": "first_frame", "num_video_frames": 9}

    def environment_action_type(self, environment):
        if environment != "robotwin":
            raise ValueError("Expected environment='robotwin'")
        return "qpos"

    def prepare_observation(self, observation):
        return self.transforms.prepare(observation)

    def postprocess_actions(self, actions, *, environment):
        self.environment_action_type(environment)
        return self.transforms.decode(actions)
