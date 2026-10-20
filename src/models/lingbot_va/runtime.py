"""Single-process port of upstream wan_va_server.VA_Server (7c6ffa9), without the server, FSDP, or debug dumps."""

from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange

from .model_code.loading import (
    WanVAEStreamingWrapper,
    load_text_encoder,
    load_tokenizer,
    load_transformer,
    load_vae,
)
from .model_code.scheduler import FlowMatchScheduler
from .model_code.utils import data_seq_to_patch, get_mesh_id


class LingBotVARuntime:
    cache_name = "pos"

    def __init__(self, config: Any, checkpoint_dir: str | Path, *, device: str = "cuda",
                 dtype: torch.dtype = torch.bfloat16):
        self.config = config
        self.device = torch.device(device)
        self.dtype = dtype
        root = Path(checkpoint_dir)

        self.scheduler = FlowMatchScheduler(shift=config.snr_shift, sigma_min=0.0,
                                            extra_one_step=True)
        self.action_scheduler = FlowMatchScheduler(shift=config.action_snr_shift,
                                                   sigma_min=0.0, extra_one_step=True)
        self.scheduler.set_timesteps(1000, training=True)
        self.action_scheduler.set_timesteps(1000, training=True)

        self.vae = load_vae(root / "vae", torch_dtype=dtype, torch_device=self.device)
        self.streaming_vae = WanVAEStreamingWrapper(self.vae)
        self.tokenizer = load_tokenizer(root / "tokenizer")
        self.text_encoder = load_text_encoder(root / "text_encoder", torch_dtype=dtype,
                                              torch_device=self.device)
        self.transformer = load_transformer(root / "transformer", torch_dtype=dtype,
                                            torch_device=self.device, attn_mode="torch")
        # from_pretrained keeps _keep_in_fp32_modules in fp32; upstream runs them in bf16
        # (FSDP param_dtype, or .to(param_dtype) without torch.distributed).
        self.transformer.eval().requires_grad_(False).to(dtype)

        self.actions_q01 = torch.tensor(config.norm_stat["q01"], dtype=torch.float32).reshape(-1, 1, 1)
        self.actions_q99 = torch.tensor(config.norm_stat["q99"], dtype=torch.float32).reshape(-1, 1, 1)
        self.action_mask = torch.zeros([config.action_dim]).bool()
        self.action_mask[config.used_action_channel_ids] = True
        self.frame_st_id = 0
        self.init_latent = None
        self.prompt_embeds = self.negative_prompt_embeds = None

    # Prompt encoding (upstream _get_t5_prompt_embeds / encode_prompt, single prompt).
    def _t5_embeds(self, prompt: str, max_sequence_length: int = 512) -> torch.Tensor:
        from diffusers.pipelines.wan.pipeline_wan import prompt_clean

        text_inputs = self.tokenizer(
            [prompt_clean(prompt)],
            padding="max_length",
            max_length=max_sequence_length,
            truncation=True,
            add_special_tokens=True,
            return_attention_mask=True,
            return_tensors="pt",
        )
        ids, mask = text_inputs.input_ids, text_inputs.attention_mask
        seq_lens = mask.gt(0).sum(dim=1).long()
        embeds = self.text_encoder(ids.to(self.device), mask.to(self.device)).last_hidden_state
        embeds = embeds.to(dtype=self.dtype, device=self.device)
        embeds = [u[:v] for u, v in zip(embeds, seq_lens)]
        return torch.stack([
            torch.cat([u, u.new_zeros(max_sequence_length - u.size(0), u.size(1))]) for u in embeds
        ], dim=0)

    @property
    def use_cfg(self) -> bool:
        return self.config.guidance_scale > 1 or self.config.action_guidance_scale > 1

    @torch.no_grad()
    def reset(self, prompt: str) -> None:
        cfg = self.config
        self.frame_st_id = 0
        self.init_latent = None
        self.transformer.clear_cache(self.cache_name)
        self.streaming_vae.clear_cache()

        self.action_per_frame = cfg.action_per_frame
        self.height, self.width = cfg.height, cfg.width
        self.latent_height = self.height // 16
        self.latent_width = self.width // 16 * len(cfg.obs_cam_keys)

        p = cfg.patch_size
        latent_token_per_chunk = (cfg.frame_chunk_size * self.latent_height * self.latent_width) // (p[0] * p[1] * p[2])
        action_token_per_chunk = cfg.frame_chunk_size * self.action_per_frame
        self.transformer.create_empty_cache(self.cache_name, cfg.attn_window,
                                            latent_token_per_chunk, action_token_per_chunk,
                                            dtype=self.dtype, device=self.device,
                                            batch_size=2 if self.use_cfg else 1)

        self.prompt_embeds = self._t5_embeds(prompt)
        self.negative_prompt_embeds = self._t5_embeds("") if self.use_cfg else None
        torch.cuda.empty_cache()

    @staticmethod
    def _normalize_latents(latents, latents_mean, latents_std):
        latents_mean = latents_mean.view(1, -1, 1, 1, 1).to(device=latents.device)
        latents_std = latents_std.view(1, -1, 1, 1, 1).to(device=latents.device)
        return ((latents.float() - latents_mean) * latents_std).to(latents)

    def encode_frames(self, frames: list[torch.Tensor]) -> torch.Tensor | None:
        """frames: list over time of [num_cams, 3, H, W] floats in [-1, 1], camera order fixed."""
        if not frames:
            return None
        videos = torch.stack(frames, dim=0).permute(1, 2, 0, 3, 4)  # cams, C, T, H, W
        enc_out = self.streaming_vae.encode_chunk(videos.to(self.device, self.dtype))
        mu, _logvar = torch.chunk(enc_out, 2, dim=1)
        latents_mean = torch.tensor(self.vae.config.latents_mean).to(mu.device)
        latents_std = torch.tensor(self.vae.config.latents_std).to(mu.device)
        mu_norm = self._normalize_latents(mu, latents_mean, 1.0 / latents_std)
        return torch.cat(mu_norm.split(1, dim=0), dim=-1).to(self.device)

    def preprocess_action(self, action: np.ndarray) -> torch.Tensor:
        action_model_input = torch.from_numpy(action)
        action_model_input = F.pad(action_model_input, [0, 0, 0, 0, 0, 1], mode="constant", value=0)
        action_model_input = action_model_input[self.config.inverse_used_action_channel_ids]
        action_model_input = (action_model_input - self.actions_q01) / (
            self.actions_q99 - self.actions_q01 + 1e-6) * 2. - 1.
        return action_model_input.unsqueeze(0).unsqueeze(-1)  # B, C, F, H, W

    def postprocess_action(self, action: torch.Tensor) -> np.ndarray:
        # Same op order and dtypes as upstream: (action + 1) / 2 stays bf16, then promotes.
        action = action.cpu()[0, ..., 0]  # C, F, H
        action = (action + 1) / 2 * (self.actions_q99 - self.actions_q01 + 1e-6) + self.actions_q01
        return action.numpy()[self.config.used_action_channel_ids]

    def _repeat_input_for_cfg(self, input_dict):
        if self.use_cfg:
            input_dict["noisy_latents"] = input_dict["noisy_latents"].repeat(2, 1, 1, 1, 1)
            input_dict["text_emb"] = torch.cat([self.prompt_embeds.to(self.dtype).clone(),
                                                self.negative_prompt_embeds.to(self.dtype).clone()], dim=0)
            input_dict["grid_id"] = input_dict["grid_id"][None].repeat(2, 1, 1)
            input_dict["timesteps"] = input_dict["timesteps"][None].repeat(2, 1)
        else:
            input_dict["grid_id"] = input_dict["grid_id"][None]
            input_dict["timesteps"] = input_dict["timesteps"][None]
        return input_dict

    def _prepare_latent_input(self, latent_model_input, action_model_input, latent_t=0, action_t=0,
                              latent_cond=None, action_cond=None, frame_st_id=0):
        patch_size = self.config.patch_size
        input_dict = {}
        if latent_model_input is not None:
            input_dict["latent_res_lst"] = {
                "noisy_latents": latent_model_input,
                "timesteps": torch.ones([latent_model_input.shape[2]], dtype=torch.float32,
                                        device=self.device) * latent_t,
                "grid_id": get_mesh_id(latent_model_input.shape[-3] // patch_size[0],
                                       latent_model_input.shape[-2] // patch_size[1],
                                       latent_model_input.shape[-1] // patch_size[2], 0,
                                       1, frame_st_id).to(self.device),
                "text_emb": self.prompt_embeds.to(self.dtype).clone(),
            }
            if latent_cond is not None:
                input_dict["latent_res_lst"]["noisy_latents"][:, :, 0:1] = latent_cond[:, :, 0:1]
                input_dict["latent_res_lst"]["timesteps"][0:1] *= 0

        if action_model_input is not None:
            input_dict["action_res_lst"] = {
                "noisy_latents": action_model_input,
                "timesteps": torch.ones([action_model_input.shape[2]], dtype=torch.float32,
                                        device=self.device) * action_t,
                "grid_id": get_mesh_id(action_model_input.shape[-3], action_model_input.shape[-2],
                                       action_model_input.shape[-1], 1, 1, frame_st_id,
                                       action=True).to(self.device),
                "text_emb": self.prompt_embeds.to(self.dtype).clone(),
            }
            if action_cond is not None:
                input_dict["action_res_lst"]["noisy_latents"][:, :, 0:1] = action_cond[:, :, 0:1]
                input_dict["action_res_lst"]["timesteps"][0:1] *= 0
            input_dict["action_res_lst"]["noisy_latents"][:, ~self.action_mask] *= 0
        return input_dict

    @torch.no_grad()
    def infer_chunk(self, init_frames: list[torch.Tensor] | None = None,
                    generator: torch.Generator | None = None,
                    num_inference_steps: int | None = None):
        """Denoise one video chunk, then the action chunk reading its KV cache.

        Returns normalized actions [1, action_dim, F, A, 1] and video latents.
        init_frames is required for the first chunk of an episode and ignored afterwards.
        """
        cfg = self.config
        frame_st_id = self.frame_st_id
        frame_chunk_size = cfg.frame_chunk_size
        if frame_st_id == 0:
            if not init_frames:
                raise ValueError("The first chunk of an episode needs the current observation")
            self.init_latent = self.encode_frames(init_frames)
        init_latent = self.init_latent

        latents = torch.randn(1, 48, frame_chunk_size, self.latent_height, self.latent_width,
                              generator=generator, device=self.device, dtype=self.dtype)
        actions = torch.randn(1, cfg.action_dim, frame_chunk_size, self.action_per_frame, 1,
                              generator=generator, device=self.device, dtype=self.dtype)

        video_step = cfg.video_exec_step
        self.scheduler.set_timesteps(num_inference_steps or cfg.num_inference_steps)
        self.action_scheduler.set_timesteps(cfg.action_num_inference_steps)
        timesteps = F.pad(self.scheduler.timesteps, (0, 1), mode="constant", value=0)
        if video_step != -1:
            timesteps = timesteps[:video_step]
        action_timesteps = F.pad(self.action_scheduler.timesteps, (0, 1), mode="constant", value=0)

        # 1. Video chunk. The last step writes its K/V into the cache, flagged as predicted.
        for i, t in enumerate(timesteps):
            last_step = i == len(timesteps) - 1
            latent_cond = init_latent[:, :, 0:1].to(self.dtype) if frame_st_id == 0 else None
            input_dict = self._prepare_latent_input(latents, None, t, t, latent_cond, None,
                                                    frame_st_id=frame_st_id)
            video_noise_pred = self.transformer(
                self._repeat_input_for_cfg(input_dict["latent_res_lst"]),
                update_cache=1 if last_step else 0,
                cache_name=self.cache_name,
                action_mode=False)
            if not last_step or video_step != -1:
                video_noise_pred = data_seq_to_patch(cfg.patch_size, video_noise_pred, frame_chunk_size,
                                                     self.latent_height, self.latent_width,
                                                     batch_size=2 if self.use_cfg else 1)
                if cfg.guidance_scale > 1:
                    video_noise_pred = video_noise_pred[1:] + cfg.guidance_scale * (
                        video_noise_pred[:1] - video_noise_pred[1:])
                else:
                    video_noise_pred = video_noise_pred[:1]
                latents = self.scheduler.step(video_noise_pred, t, latents, return_dict=False)
            latents[:, :, 0:1] = latent_cond if frame_st_id == 0 else latents[:, :, 0:1]

        # 2. Action chunk attending to the cache (past real frames/actions + predicted video).
        for i, t in enumerate(action_timesteps):
            last_step = i == len(action_timesteps) - 1
            action_cond = torch.zeros([1, cfg.action_dim, 1, self.action_per_frame, 1],
                                      device=self.device, dtype=self.dtype) if frame_st_id == 0 else None
            input_dict = self._prepare_latent_input(None, actions, t, t, None, action_cond,
                                                    frame_st_id=frame_st_id)
            action_noise_pred = self.transformer(
                self._repeat_input_for_cfg(input_dict["action_res_lst"]),
                update_cache=1 if last_step else 0,
                cache_name=self.cache_name,
                action_mode=True)
            if not last_step:
                action_noise_pred = rearrange(action_noise_pred, "b (f n) c -> b c f n 1",
                                              f=frame_chunk_size)
                if cfg.action_guidance_scale > 1:
                    action_noise_pred = action_noise_pred[1:] + cfg.action_guidance_scale * (
                        action_noise_pred[:1] - action_noise_pred[1:])
                else:
                    action_noise_pred = action_noise_pred[:1]
                actions = self.action_scheduler.step(action_noise_pred, t, actions, return_dict=False)
            actions[:, :, 0:1] = action_cond if frame_st_id == 0 else actions[:, :, 0:1]

        actions[:, ~self.action_mask] *= 0
        return actions, latents

    @torch.no_grad()
    def compute_kv_cache(self, frames: list[torch.Tensor], executed_chunk: np.ndarray) -> None:
        """Replace the predicted K/V with real observed frames and the executed action chunk.

        executed_chunk is the denormalized [7, F, A] chunk returned for the last call.
        """
        self.transformer.clear_pred_cache(self.cache_name)
        latent_model_input = self.encode_frames(frames)
        if self.frame_st_id == 0:
            latent_model_input = (torch.cat([self.init_latent, latent_model_input], dim=2)
                                  if latent_model_input is not None else self.init_latent)
        action_model_input = self.preprocess_action(executed_chunk).to(latent_model_input)
        input_dict = self._prepare_latent_input(latent_model_input, action_model_input,
                                                frame_st_id=self.frame_st_id)
        self.transformer(self._repeat_input_for_cfg(input_dict["latent_res_lst"]),
                         update_cache=2, cache_name=self.cache_name, action_mode=False)
        self.transformer(self._repeat_input_for_cfg(input_dict["action_res_lst"]),
                         update_cache=2, cache_name=self.cache_name, action_mode=True)
        torch.cuda.empty_cache()
        self.frame_st_id += latent_model_input.shape[2]
