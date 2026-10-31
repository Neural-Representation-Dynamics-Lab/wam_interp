"""LingBot-VA LIBERO transforms and episode protocol, following upstream evaluation/libero/client.py."""

from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from wam_interp.types import EnvironmentObservation, PreparedObservation

from .adapter import LingBotVAAdapter


class LingBotVALiberoAdapter(LingBotVAAdapter):
    camera_order = ("agentview", "wrist")

    def __init__(self, model: Any):
        super().__init__(model)
        # Upstream keeps every (action_per_frame // 4)-th frame; with 4 that is every frame.
        if model.config.action_per_frame != 4:
            raise ValueError("LingBotVALiberoAdapter expects action_per_frame == 4")
        self._in_episode = False
        self._last_chunk = None
        self._frames: list[Any] = []

    def begin_episode(self, observation: EnvironmentObservation) -> None:
        self.reset(observation.instruction)
        self._in_episode = True
        self._last_chunk = None
        self._frames = []

    def observe_step(self, observation: EnvironmentObservation, action: Any) -> None:
        self._frames.append(self.prepare_observation(observation).image)

    def prepare_observation(self, observation: EnvironmentObservation) -> PreparedObservation:
        cfg = self.model.config
        views = []
        for name in self.camera_order:
            image = np.ascontiguousarray(np.asarray(observation.cameras[name])[::-1])
            tensor = torch.from_numpy(image).float().permute(2, 0, 1)[None]
            tensor = F.interpolate(tensor, size=(cfg.height, cfg.width), mode="bilinear",
                                   align_corners=False)[0]
            views.append(tensor / 255.0 * 2.0 - 1.0)
        return PreparedObservation(
            image=torch.stack(views, dim=0),
            proprio=None,
            prompt=observation.instruction,
            metadata={
                **observation.metadata,
                "camera_order": list(self.camera_order),
                "flip": "vertical",
                "lingbot_image_shape": [len(views), 3, cfg.height, cfg.width],
            },
        )

    def predict_action(self, request, recorder=None):
        if not self._in_episode:
            raise RuntimeError("begin_episode() must run before LingBot-VA predicts in LIBERO")
        # New frames since the last chunk mean it was executed; none means a repeat prediction.
        if self._last_chunk is not None and self._frames:
            cfg = self.model.config
            expected = self._last_chunk.shape[1] * self._last_chunk.shape[2]
            if self.model.frame_st_id == 0:
                expected -= cfg.action_per_frame
            if len(self._frames) != expected:
                raise ValueError(
                    f"LingBot-VA needs every frame of the executed chunk ({expected}), got "
                    f"{len(self._frames)}; set replan_steps >= {self.action_horizon}"
                )
            self.update(self._frames, self._last_chunk)
            self._frames = []
        return super().predict_action(request, recorder=recorder)

    def postprocess_actions(self, actions: Any, *, environment: str = "libero"):
        if environment != "libero":
            raise ValueError(f"LingBotVALiberoAdapter only supports LIBERO, got {environment!r}")
        first_chunk = self.model.frame_st_id == 0
        chunk = self.model.postprocess_action(actions)  # [7, frames, actions_per_frame]
        self._last_chunk = chunk
        commands = chunk.transpose(1, 2, 0).reshape(-1, chunk.shape[0])
        if first_chunk:
            commands = commands[chunk.shape[2]:]
        return commands
