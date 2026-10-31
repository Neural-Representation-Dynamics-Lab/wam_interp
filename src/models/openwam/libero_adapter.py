"""OpenWAM LIBERO transforms, following upstream benchmarks/libero/openwam2libero_interface.py."""

import numpy as np

from wam_interp.types import EnvironmentObservation, PreparedObservation

from .adapter import OpenWAMAdapter

LIBERO_EEF10_DIM = 10


def native_eef10_to_libero7d(action, *, clip=True):
    """Upstream benchmarks/libero/openwam2libero_interface.py::native_eef10_to_libero7d."""
    from .model_code.benchmarks.utils.action_conversion import libero_open_scale_to_gripper_cmd, rot6d_to_axis_angle

    value = np.asarray(action, dtype=np.float32).reshape(-1)
    if value.shape != (LIBERO_EEF10_DIM,):
        raise ValueError(f"expected a native LIBERO EEF10 action, got {value.shape}")
    if not np.isfinite(value).all():
        raise ValueError("native LIBERO EEF10 action contains NaN or infinity")
    position = value[:3].copy()
    rotation = rot6d_to_axis_angle(value[3:9]).astype(np.float32)
    gripper = np.array([libero_open_scale_to_gripper_cmd(value[9])], dtype=np.float32)
    if clip:
        position = np.clip(position, -1.0, 1.0)
        rotation = np.clip(rotation, -1.0, 1.0)
        gripper = np.clip(gripper, -1.0, 1.0)
    return np.concatenate([position, rotation, gripper]).astype(np.float32)


def _camera(observation, name):
    image = np.asarray(observation.cameras[name])
    if image.ndim != 3 or image.shape[-1] != 3:
        raise ValueError(f"LIBERO camera {name!r} must be HxWx3, got {image.shape}")
    return np.ascontiguousarray(image.astype(np.uint8, copy=False)[::-1, ::-1])  # upstream image_transform=rotate_180


def composite_to_tensor(image):
    import torch

    return torch.from_numpy(np.asarray(image, dtype=np.uint8).copy()).permute(2, 0, 1).float() / 127.5 - 1.0


class OpenWAMLiberoAdapter(OpenWAMAdapter):
    """``wrist='real'`` sends the wrist camera as upstream's client does; ``'black'`` leaves that tile black."""

    environment = "libero"

    def __init__(self, model, *, wrist="real"):
        super().__init__(model)
        if wrist not in ("real", "black"):
            raise ValueError("wrist must be 'real' or 'black'")
        dataloader = model.cfg.dataloader
        if str(dataloader.get("action_mode")) != "eef" or str(dataloader.get("type")) != "libero":
            raise ValueError("Use an OpenWAM LIBERO checkpoint (dataloader.type=libero, action_mode=eef)")
        raw_dim = self.denormalize(np.zeros((1, self.architecture.action_dim), np.float32)).shape[-1]
        if raw_dim != LIBERO_EEF10_DIM:
            raise ValueError(f"LIBERO checkpoint must decode {LIBERO_EEF10_DIM}-D EEF actions, got {raw_dim}")
        from .model_code.deploy.obs_preprocess import ObsPreprocessor

        self.wrist = wrist
        self.observation_preprocessor = ObsPreprocessor.from_cfg(model.cfg, model.engine)

    def prepare_observation(self, observation: EnvironmentObservation) -> PreparedObservation:
        from PIL import Image
        from .model_code.benchmarks.utils.action_conversion import libero_obs_to_eef10
        from .model_code.benchmarks.utils.client import resize_for_lshape_slot

        head = resize_for_lshape_slot(_camera(observation, "agentview"), "head_camera")
        wrist = None
        if self.wrist == "real":
            wrist = Image.fromarray(resize_for_lshape_slot(_camera(observation, "wrist"), "left_wrist_camera"))
        state = libero_obs_to_eef10(observation.state["eef_position"], observation.state["eef_quaternion"],
                                    observation.state["gripper_position"])
        payload = self.observation_preprocessor.preprocess({
            "images": {"head_camera": Image.fromarray(head), "left_wrist_camera": wrist, "right_wrist_camera": None},
            "prompt": observation.instruction, "state": state})
        import torch

        return PreparedObservation(
            image=composite_to_tensor(payload["image"]),
            proprio=torch.from_numpy(self.normalize_state(payload["state"])),
            prompt=payload["prompt"],
            metadata={**observation.metadata, "camera_layout": list(self.observation_preprocessor.camera_layout),
                      "image_transform": "rotate_180", "wrist": self.wrist, "raw_state_eef10": payload["state"].tolist()},
        )

    def postprocess_actions(self, actions, *, environment="libero"):
        if environment != self.environment:
            raise ValueError(f"{type(self).__name__} only supports environment={self.environment!r}")
        return np.stack([native_eef10_to_libero7d(step) for step in self.denormalize(actions)])
