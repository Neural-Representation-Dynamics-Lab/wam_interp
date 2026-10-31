"""OpenWAM RoboTwin transforms (EE actions), following upstream benchmarks/robotwin/openwam2robotwin_interface.py."""

import numpy as np

from wam_interp.types import EnvironmentObservation, PreparedObservation

from .adapter import OpenWAMAdapter
from .libero_adapter import composite_to_tensor

ROBOTWIN_EEF20_DIM = 20
_CAMERAS = (("head", "head_camera"), ("left_wrist", "left_wrist_camera"), ("right_wrist", "right_wrist_camera"))


class OpenWAMRobotWinAdapter(OpenWAMAdapter):
    def __init__(self, model):
        super().__init__(model)
        dataloader = model.cfg.dataloader
        if str(dataloader.get("action_mode")) != "eef" or str(dataloader.get("type")) != "robotwin":
            raise ValueError("Use an OpenWAM RoboTwin EEF checkpoint (dataloader.type=robotwin, action_mode=eef)")
        raw_dim = self.denormalize(np.zeros((1, self.architecture.action_dim), np.float32)).shape[-1]
        if raw_dim != ROBOTWIN_EEF20_DIM:
            raise ValueError(f"RoboTwin EE checkpoint must decode {ROBOTWIN_EEF20_DIM}-D actions, got {raw_dim}")
        from .model_code.deploy.obs_preprocess import ObsPreprocessor

        self.observation_preprocessor = ObsPreprocessor.from_cfg(model.cfg, model.engine)

    def environment_action_type(self, environment):
        if environment != "robotwin":
            raise ValueError("Expected environment='robotwin'")
        return "ee"

    def prepare_observation(self, observation: EnvironmentObservation) -> PreparedObservation:
        import torch
        from PIL import Image
        from .model_code.benchmarks.utils.action_conversion import robotwin_endpose_to_eef20d
        from .model_code.dataloader.transforms.multiview import format_prompt_for_inference

        images = {}
        for name, slot in _CAMERAS:
            image = np.asarray(observation.cameras[name])
            if image.ndim != 3 or image.shape[-1] != 3 or image.dtype != np.uint8:
                raise ValueError(f"RoboTwin camera {name!r} must be HxWx3 uint8, got {image.dtype} {image.shape}")
            images[slot] = Image.fromarray(np.ascontiguousarray(image))
        state = observation.state
        eef20 = robotwin_endpose_to_eef20d(state["left_endpose"], state["right_endpose"],
                                           state["left_gripper"], state["right_gripper"])
        payload = self.observation_preprocessor.preprocess({
            "images": images, "prompt": format_prompt_for_inference(observation.instruction), "state": eef20})
        return PreparedObservation(
            image=composite_to_tensor(payload["image"]),
            proprio=torch.from_numpy(self.normalize_state(payload["state"])),
            prompt=payload["prompt"],
            metadata={**observation.metadata, "camera_layout": list(self.observation_preprocessor.camera_layout),
                      "raw_state_eef20": payload["state"].tolist()},
        )

    def postprocess_actions(self, actions, *, environment):
        from .model_code.benchmarks.utils.action_conversion import eef20d_to_ee16d

        self.environment_action_type(environment)
        return np.stack([eef20d_to_ee16d(step) for step in self.denormalize(actions)])
