"""A common eager capture/replay interface for action-facing future representations."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .futures import BOUNDARY, capture_future
from .model_capabilities import CAPABILITIES
from .native_features import native_features


@dataclass(frozen=True)
class FutureDonor:
    """CPU tensors from one loaded adapter and intervention site; observed slots are excluded."""

    parts: tuple[Any, ...]
    backend: str
    schedule: tuple
    _binding: object = field(repr=False, compare=False)
    frame_start: int | None = None


class FutureAdapter:
    """Wrap an already-loaded benchmark adapter. Sequential use on one model."""

    def __init__(self, adapter, *, layer=None, stage=0):
        self.adapter = adapter
        self.backend = adapter.name
        entries = [
            c
            for c in CAPABILITIES
            if c.backend == self.backend
            and c.e1_status in ("native_feature_hook_implemented", "exploratory_gpu_evidence")
        ]
        if len(entries) != 1:
            raise ValueError(f"No E1 future adapter for {self.backend!r}")
        self.capability = entries[0]
        self.owner = adapter.architecture.mot_driver if self.backend == "openwam" else adapter.model
        if (
            self.backend == "fastwam"
            and getattr(self.owner, "research_future_boundary", None) != BOUNDARY
        ):
            raise ValueError("FastWAM requires an Optional IDM checkpoint")
        if self.backend == "longwam" and adapter.native_condition != "idm":
            raise ValueError("Long-WAM requires an IDM checkpoint")
        if self.backend == "fasterwam":
            layers = sorted(self.owner.mot.condition_layer_set)
        elif self.backend == "lingbot_va":
            layers = list(range(len(self.owner.transformer.blocks)))
        elif self.backend == "openwam":
            layers = list(range(self.owner.num_layers))
        else:
            layers = [None]
        if not layers or (layer is not None and layer not in layers):
            raise ValueError(f"Invalid intervention layer {layer} for {self.backend}")
        if type(stage) is not int or stage < 0 or (stage and self.backend != "openwam"):
            raise ValueError(f"Invalid intervention stage {stage} for {self.backend}")
        self._layer = layers[len(layers) // 2] if layer is None else layer
        self._stage = stage
        self._binding = object()

    @property
    def layer(self):
        return self._layer

    @property
    def stage(self):
        return self._stage

    @property
    def supports_clip(self):
        return self.backend in ("fastwam", "longwam")

    def _schedule(self, request):
        if request.condition != self.capability.condition or request.compile_action_infer:
            raise ValueError(
                f"Use eager {self.backend} inference with condition={self.capability.condition!r}"
            )
        return (
            request.condition,
            request.action_horizon,
            request.num_video_frames,
            request.num_inference_steps,
            request.sigma_shift,
            request.tiled,
            request.rand_device,
            request.text_cfg_scale,
            self.layer,
            self.stage,
        )

    def capture(self, request, *, donor=None):
        """Return (prediction, original future at this site), optionally replaying a donor."""
        schedule = self._schedule(request)
        if donor is not None and (
            not isinstance(donor, FutureDonor)
            or donor._binding is not self._binding
            or donor.schedule != schedule
        ):
            raise ValueError("Donor must use this loaded adapter, site and inference schedule")
        frame_start = None
        if self.backend == "fastwam":
            if donor is not None and len(donor.parts) != 1:
                raise ValueError("FastWAM requires one future tensor")
            replacement = None if donor is None else donor.parts[0]
            with capture_future(self.owner, replacement, future_only=True) as report:
                result = self.adapter.predict_action(request)
            parts = (report["original"],)
        else:
            replacements = None if donor is None else {self.stage: donor.parts}
            with native_features(
                self.owner,
                layer=self.layer,
                stage=self.stage,
                donors=replacements,
                expected_frame_start=donor.frame_start if donor is not None else None,
            ) as report:
                result = self.adapter.predict_action(request)
            parts = report.captured[self.stage]
            frame_start = report.frame_start
        return result, FutureDonor(parts, self.backend, schedule, self._binding, frame_start)

    def predict_with_future(self, request, donor):
        """Replay a donor through the native action policy."""
        if donor is None:
            raise ValueError("A future donor is required")
        return self.capture(request, donor=donor)[0]

    def from_clip(self, request, clip, *, noise_seed=None):
        """Encode an anchor-first uint8 RGB montage at the backend's native sampled times.

        Long-WAM uses its current episode history and training noise schedule. Use the
        same noise_seed for every arm of a comparison, independent of the action seed.
        """
        if not self.supports_clip:
            raise NotImplementedError(
                f"{self.backend} supports native feature donors; no clip encoder is validated"
            )
        import numpy as np
        import torch

        rgb = np.asarray(clip)
        if rgb.dtype != np.uint8 or rgb.ndim != 4 or rgb.shape[-1] != 3 or len(rgb) < 2:
            raise ValueError("Expected [time, height, width, 3] uint8 RGB")
        model = self.owner
        with torch.inference_mode():
            video = torch.as_tensor(np.ascontiguousarray(rgb), device=model.device).permute(3, 0, 1, 2)[
                None
            ]
            video = video.to(model.torch_dtype) * (2.0 / 255.0) - 1.0
            return self._from_prepared_video(request, video, noise_seed)

    def from_camera_frames(self, request, frames, *, noise_seed):
        """Use the native Long-WAM RoboTwin preprocessing for every raw camera triplet."""
        import torch
        from wam_interp.models.longwam.transforms import robotwin_model_image

        if self.backend != "longwam" or getattr(self.adapter, "benchmark", None) != "robotwin":
            raise ValueError("Raw camera triplets require Long-WAM RoboTwin")
        adapter = self.adapter
        with torch.inference_mode():
            images = [
                robotwin_model_image(
                    cameras, adapter.image_meta, device=self.owner.device, dtype=self.owner.torch_dtype
                )
                for cameras in frames
            ]
            return self._from_prepared_video(request, torch.stack(images, dim=2), noise_seed)

    def _from_prepared_video(self, request, video, noise_seed):
        import torch

        schedule = self._schedule(request)
        model = self.owner
        with torch.inference_mode():
            if not torch.equal(video[:, :, 0], request.observation.image):
                raise ValueError("Clip frame zero differs from the recipient observation")
            if self.backend == "fastwam":
                if video.shape[2] != request.num_video_frames or request.tiled or noise_seed is not None:
                    raise ValueError(
                        "FastWAM clip requires native frame count and untiled, clean encoding"
                    )
                future = model.vae.model.encode(video, model.vae.scale)[:, :, 1:]
            else:
                from wam_interp.models.longwam.transforms import (
                    build_observation_window,
                )

                adapter = self.adapter
                k = int(model.num_imagine_frames)
                if (
                    video.shape[2] != 1 + k * model.vae.temporal_downsample_factor
                    or type(noise_seed) is not int
                ):
                    raise ValueError(
                        "Long-WAM clip requires its native future frame count and an explicit noise_seed"
                    )
                if (
                    not adapter._in_episode
                    or not adapter._frames
                    or not torch.equal(adapter._frames[-1], video[0, :, 0])
                ):
                    raise ValueError("Long-WAM requires the recipient's matching episode history")
                history = build_observation_window(
                    list(adapter._frames), adapter.past_obs_size, adapter.sample_stride
                )
                if history is None:
                    history = video[:, :, :1]
                full = model._encode_video_latents(
                    torch.cat((history, video[:, :, 1:]), dim=2), tiled=request.tiled
                )
                if full.shape[2] != model.num_clean_frames + k:
                    raise ValueError("Encoded history/future length differs from the checkpoint")
                clean = full[:, :, model.num_clean_frames :]
                generator = torch.Generator(device="cpu").manual_seed(noise_seed)
                noise = torch.randn(clean.shape, generator=generator, dtype=torch.float32).to(clean)
                timestep = torch.as_tensor(model.imagine_timestep, device=clean.device)
                future = model.train_video_scheduler.add_noise(clean, noise, timestep)
            if not torch.isfinite(future).all():
                raise ValueError("Nonfinite clip encoding")
            return FutureDonor(
                (future.detach().to("cpu", copy=True),), self.backend, schedule, self._binding
            )
