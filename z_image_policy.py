"""
Z-Image policy adapter (extensible backend).

This adapter loads a user-provided z-image implementation and exposes a
consistent generate + policy_gradient_step interface for PPO training.
"""

from __future__ import annotations

import contextlib
import importlib
import importlib.util
import inspect
import logging
import os
from typing import Any, Dict, List, Optional, Tuple

from PIL import Image
import torch

logger = logging.getLogger(__name__)


def _env_get_bool(name: str, default: Optional[bool] = None) -> Optional[bool]:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in ("1", "true", "yes", "y", "on")


def _env_get_int(name: str, default: Optional[int] = None) -> Optional[int]:
    value = os.getenv(name)
    if value is None or value == "":
        return default
    try:
        return int(value)
    except ValueError:
        return default


def _env_get_float(name: str, default: Optional[float] = None) -> Optional[float]:
    value = os.getenv(name)
    if value is None or value == "":
        return default
    try:
        return float(value)
    except ValueError:
        return default


def _select_torch_dtype(device: str, override: Optional[str] = None):
    if override:
        key = override.strip().lower()
        if key in ("bf16", "bfloat16"):
            return torch.bfloat16
        if key in ("fp16", "float16"):
            return torch.float16
        if key in ("fp32", "float32"):
            return torch.float32

    if device and "cuda" in device and torch.cuda.is_available():
        try:
            if hasattr(torch.cuda, "is_bf16_supported") and torch.cuda.is_bf16_supported():
                return torch.bfloat16
        except Exception:
            pass
        return torch.float16
    return torch.float32


def _is_diffusers_model_dir(path: str) -> bool:
    return bool(path and os.path.isdir(path) and os.path.exists(os.path.join(path, "model_index.json")))


def _maybe_snapshot_download(model_id: str) -> str:
    if not model_id:
        return model_id
    if os.path.exists(model_id):
        return model_id
    if not _env_get_bool("Z_IMAGE_MODELSCOPE_SNAPSHOT", False):
        return model_id
    try:
        from modelscope import snapshot_download
    except Exception:
        return model_id
    try:
        return snapshot_download(model_id)
    except Exception:
        return model_id


def _load_module_from_path(path: str):
    spec = importlib.util.spec_from_file_location("z_image_user_impl", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import module from path: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _import_module_by_name(name: str):
    return importlib.import_module(name)


def _call_with_fallback(fn, **kwargs):
    try:
        return fn(**kwargs)
    except TypeError:
        # Fallback to positional with common order
        ordered = []
        for key in ("model_path", "device", "prompts", "init_images", "images", "advantages"):
            if key in kwargs:
                ordered.append(kwargs[key])
        return fn(*ordered)


def _set_requires_grad(module, enabled: bool) -> None:
    if module is None:
        return
    for param in module.parameters():
        param.requires_grad = enabled


class ZImagePolicy:
    """Thin adapter around the z-image generator."""

    def __init__(
        self,
        model_path: Optional[str],
        device: str = "cuda",
        allow_stub: bool = False,
        learning_rate: float = 1e-5,
        max_grad_norm: float = 1.0,
    ):
        self.model_path = model_path
        if str(device).lower().startswith("cuda") and not torch.cuda.is_available():
            logger.warning("CUDA not available for Z-Image; falling back to CPU.")
            device = "cpu"
        self.device = device
        self.allow_stub = allow_stub
        self.learning_rate = learning_rate
        self.max_grad_norm = max_grad_norm
        self.backend = None

        self._load_backend()

    def _load_backend(self) -> None:
        if self.allow_stub:
            logger.warning("Z-Image backend not configured; using stub backend.")
            self.backend = _StubBackend()
            return

        if not self.model_path:
            raise RuntimeError(
                "z-image model path is not set. Set --z_image_model_path or enable --allow_zimage_stub."
            )

        entrypoint = os.getenv("Z_IMAGE_ENTRYPOINT")
        if entrypoint:
            if entrypoint.endswith(".py") and os.path.exists(entrypoint):
                module = _load_module_from_path(entrypoint)
            else:
                module = _import_module_by_name(entrypoint)
            self.backend = _PythonBackend(
                module=module,
                model_path=self.model_path,
                device=self.device,
                learning_rate=self.learning_rate,
                max_grad_norm=self.max_grad_norm,
            )
            return

        model_path = _maybe_snapshot_download(self.model_path)
        self.model_path = model_path

        if _is_diffusers_model_dir(model_path):
            self.backend = _DiffusersZImageBackend(
                model_path=model_path,
                device=self.device,
                learning_rate=self.learning_rate,
                max_grad_norm=self.max_grad_norm,
            )
            return

        module = None
        if model_path.endswith(".py") and os.path.exists(model_path):
            module = _load_module_from_path(model_path)
        else:
            # Try standard module name
            try:
                module = _import_module_by_name("z_image")
            except Exception as exc:
                raise RuntimeError(
                    "Unable to import z-image backend. Set Z_IMAGE_ENTRYPOINT or pass a .py path."
                ) from exc

        self.backend = _PythonBackend(
            module=module,
            model_path=model_path,
            device=self.device,
            learning_rate=self.learning_rate,
            max_grad_norm=self.max_grad_norm,
        )

    def generate(
        self,
        prompts: List[str],
        init_images: Optional[List[Image.Image]] = None,
        generation_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[List[Image.Image], Optional[torch.Tensor], Dict[str, Any]]:
        return self.backend.generate(
            prompts,
            init_images=init_images,
            generation_kwargs=generation_kwargs,
        )

    def policy_gradient_step(
        self,
        prompts: List[str],
        images: List[Image.Image],
        log_probs: Optional[torch.Tensor],
        advantages: Optional[torch.Tensor],
    ) -> Dict[str, Any]:
        if hasattr(self.backend, "policy_gradient_step"):
            return self.backend.policy_gradient_step(
                prompts=prompts,
                images=images,
                log_probs=log_probs,
                advantages=advantages,
            )
        return {"policy_update_skipped": True}

    def save_checkpoint(self, checkpoint_path: str) -> None:
        if hasattr(self.backend, "save_checkpoint"):
            self.backend.save_checkpoint(checkpoint_path)

    def load_checkpoint(self, checkpoint_path: str) -> None:
        if hasattr(self.backend, "load_checkpoint"):
            self.backend.load_checkpoint(checkpoint_path)


class _DiffusersZImageBackend:
    def __init__(self, model_path: str, device: str, learning_rate: float, max_grad_norm: float):
        self.model_path = model_path
        if str(device).lower().startswith("cuda") and not torch.cuda.is_available():
            logger.warning("CUDA not available for Z-Image; falling back to CPU.")
            device = "cpu"
        self.device = device
        self.learning_rate = learning_rate
        self.max_grad_norm = max_grad_norm
        self.turbo = self._infer_turbo()
        self.enable_grad = _env_get_bool("Z_IMAGE_ENABLE_GRAD", False)
        self.train_transformer = _env_get_bool("Z_IMAGE_TRAIN_TRANSFORMER", True)
        self.train_text_encoder = _env_get_bool("Z_IMAGE_TRAIN_TEXT_ENCODER", False)
        self.train_vae = _env_get_bool("Z_IMAGE_TRAIN_VAE", False)

        self.pipe = self._init_pipeline()
        self.scheduler = getattr(self.pipe, "scheduler", None)
        self.default_kwargs = self._build_default_kwargs()
        self.optimizer, self.trainable_params = self._init_optimizer()

    def _infer_turbo(self) -> bool:
        name = os.path.basename(self.model_path).lower()
        return "turbo" in name or "turbo" in self.model_path.lower()

    def _init_pipeline(self):
        try:
            from diffusers import ZImagePipeline
        except Exception as exc:
            raise RuntimeError(
                "diffusers is required for Z-Image pipeline. Install diffusers to use this backend."
            ) from exc

        dtype = _select_torch_dtype(self.device, os.getenv("Z_IMAGE_DTYPE"))
        local_only = _env_get_bool("Z_IMAGE_LOCAL_ONLY", None)
        if local_only is None:
            local_only = True

        pipe = ZImagePipeline.from_pretrained(
            self.model_path,
            torch_dtype=dtype,
            local_files_only=local_only,
        )
        if self.device:
            pipe.to(self.device)
        try:
            pipe.set_progress_bar_config(disable=True)
        except Exception:
            pass

        if _env_get_bool("Z_IMAGE_CPU_OFFLOAD", False):
            try:
                pipe.enable_model_cpu_offload()
            except Exception:
                pass
        if _env_get_bool("Z_IMAGE_ENABLE_XFORMERS", False):
            try:
                pipe.enable_xformers_memory_efficient_attention()
            except Exception:
                pass

        self._configure_trainable_modules(pipe)
        return pipe

    def _configure_trainable_modules(self, pipe) -> None:
        if hasattr(pipe, "transformer"):
            _set_requires_grad(pipe.transformer, self.train_transformer)
            if self.train_transformer:
                pipe.transformer.train()
                if self.enable_grad and _env_get_bool("Z_IMAGE_GRADIENT_CHECKPOINTING", True):
                    for method_name in ("enable_gradient_checkpointing", "gradient_checkpointing_enable"):
                        method = getattr(pipe.transformer, method_name, None)
                        if callable(method):
                            try:
                                method()
                                logger.info("Enabled z-image transformer gradient checkpointing.")
                                break
                            except Exception:
                                continue
            else:
                pipe.transformer.eval()
        if hasattr(pipe, "text_encoder"):
            _set_requires_grad(pipe.text_encoder, self.train_text_encoder)
            if self.train_text_encoder:
                pipe.text_encoder.train()
            else:
                pipe.text_encoder.eval()
        if hasattr(pipe, "vae"):
            _set_requires_grad(pipe.vae, self.train_vae)
            if self.train_vae:
                pipe.vae.train()
            else:
                pipe.vae.eval()

    def _init_optimizer(self):
        if not self.enable_grad:
            return None, []

        params: List[torch.Tensor] = []
        if self.train_transformer and hasattr(self.pipe, "transformer"):
            params.extend(list(self.pipe.transformer.parameters()))
        if self.train_text_encoder and hasattr(self.pipe, "text_encoder"):
            params.extend(list(self.pipe.text_encoder.parameters()))
        if self.train_vae and hasattr(self.pipe, "vae"):
            params.extend(list(self.pipe.vae.parameters()))

        params = [p for p in params if p.requires_grad]
        if not params:
            return None, []

        optimizer = torch.optim.AdamW(params, lr=self.learning_rate)
        return optimizer, params

    def _build_default_kwargs(self) -> Dict[str, Any]:
        kwargs: Dict[str, Any] = {}

        steps = _env_get_int("Z_IMAGE_STEPS")
        if steps is None:
            steps = 9 if self.turbo else 50
        kwargs["num_inference_steps"] = steps

        guidance = _env_get_float("Z_IMAGE_GUIDANCE")
        if guidance is None:
            guidance = 0.0 if self.turbo else 4.0
        kwargs["guidance_scale"] = guidance

        height = _env_get_int("Z_IMAGE_HEIGHT")
        width = _env_get_int("Z_IMAGE_WIDTH")
        if height is None:
            height = 1024 if self.turbo else 720
        if width is None:
            width = 1024 if self.turbo else 1280
        kwargs["height"] = height
        kwargs["width"] = width

        negative_prompt = os.getenv("Z_IMAGE_NEGATIVE_PROMPT")
        if negative_prompt:
            kwargs["negative_prompt"] = negative_prompt

        cfg_norm = _env_get_bool("Z_IMAGE_CFG_NORMALIZATION", None)
        if cfg_norm is None and not self.turbo:
            cfg_norm = True
        if cfg_norm is not None:
            kwargs["cfg_normalization"] = cfg_norm

        return kwargs

    def _normalize_prompt_embed_list(self, embeds: Any) -> List[torch.Tensor]:
        if embeds is None:
            return []
        if isinstance(embeds, tuple):
            embeds = embeds[0]
        if isinstance(embeds, list):
            result: List[torch.Tensor] = []
            for item in embeds:
                if isinstance(item, torch.Tensor):
                    result.append(item.to(self.device))
                else:
                    result.append(torch.as_tensor(item, device=self.device))
            return result
        if isinstance(embeds, torch.Tensor):
            if embeds.dim() == 3:
                return [embeds[i].to(self.device) for i in range(embeds.shape[0])]
            if embeds.dim() == 2:
                return [embeds.to(self.device)]
        raise RuntimeError(f"Unsupported prompt embedding type: {type(embeds)}")

    def _call_by_signature(self, fn, candidate_kwargs: Dict[str, Any], positional_fallback: Optional[List[Any]] = None):
        signature = inspect.signature(fn)
        accepted = {}
        for name in signature.parameters.keys():
            if name in candidate_kwargs:
                accepted[name] = candidate_kwargs[name]
        try:
            return fn(**accepted)
        except TypeError:
            if positional_fallback is not None:
                return fn(*positional_fallback)
            raise

    def _encode_prompt(
        self,
        prompts: List[str],
        num_images_per_prompt: int,
        do_cfg: bool,
        negative_prompt: Optional[str],
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        max_sequence_length = int(os.getenv("Z_IMAGE_MAX_SEQUENCE_LENGTH", "512"))

        if hasattr(self.pipe, "encode_prompt"):
            out = self._call_by_signature(
                self.pipe.encode_prompt,
                {
                    "prompt": prompts,
                    "device": self.device,
                    "do_classifier_free_guidance": do_cfg,
                    "negative_prompt": negative_prompt,
                    "prompt_embeds": None,
                    "negative_prompt_embeds": None,
                    "max_sequence_length": max_sequence_length,
                },
            )
            if isinstance(out, tuple) and len(out) >= 2:
                prompt_embeds, negative_prompt_embeds = out[0], out[1]
            else:
                prompt_embeds, negative_prompt_embeds = out, []

            prompt_embeds_list = self._normalize_prompt_embed_list(prompt_embeds)
            negative_prompt_embeds_list = self._normalize_prompt_embed_list(negative_prompt_embeds)

            if num_images_per_prompt > 1:
                prompt_embeds_list = [pe for pe in prompt_embeds_list for _ in range(num_images_per_prompt)]
                if negative_prompt_embeds_list:
                    negative_prompt_embeds_list = [
                        ne for ne in negative_prompt_embeds_list for _ in range(num_images_per_prompt)
                    ]

            return prompt_embeds_list, negative_prompt_embeds_list

        if hasattr(self.pipe, "_encode_prompt"):
            prompt_embeds = self._call_by_signature(
                self.pipe._encode_prompt,
                {
                    "prompt": prompts,
                    "device": self.device,
                    "prompt_embeds": None,
                    "max_sequence_length": max_sequence_length,
                },
                positional_fallback=[prompts, self.device, None, max_sequence_length],
            )

            if do_cfg:
                if negative_prompt is None:
                    negative_prompt = ["" for _ in prompts]
                elif isinstance(negative_prompt, str):
                    negative_prompt = [negative_prompt for _ in prompts]
                negative_prompt_embeds = self._call_by_signature(
                    self.pipe._encode_prompt,
                    {
                        "prompt": negative_prompt,
                        "device": self.device,
                        "prompt_embeds": None,
                        "max_sequence_length": max_sequence_length,
                    },
                    positional_fallback=[negative_prompt, self.device, None, max_sequence_length],
                )
            else:
                negative_prompt_embeds = []

            prompt_embeds_list = self._normalize_prompt_embed_list(prompt_embeds)
            negative_prompt_embeds_list = self._normalize_prompt_embed_list(negative_prompt_embeds)

            if num_images_per_prompt > 1:
                prompt_embeds_list = [pe for pe in prompt_embeds_list for _ in range(num_images_per_prompt)]
                if negative_prompt_embeds_list:
                    negative_prompt_embeds_list = [
                        ne for ne in negative_prompt_embeds_list for _ in range(num_images_per_prompt)
                    ]

            return prompt_embeds_list, negative_prompt_embeds_list

        raise RuntimeError("Pipeline does not expose encode_prompt/_encode_prompt.")

    def _prepare_latents(
        self,
        batch_size: int,
        num_images_per_prompt: int,
        height: int,
        width: int,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        if hasattr(self.pipe, "prepare_latents"):
            num_channels = getattr(self.pipe.transformer, "config", None)
            num_channels = getattr(num_channels, "in_channels", None)
            if num_channels is None and hasattr(self.pipe, "vae"):
                num_channels = getattr(self.pipe.vae.config, "latent_channels", 4)
            latent_args = [
                (batch_size * num_images_per_prompt, num_channels, height, width, dtype, self.device, None, None),
                (batch_size * num_images_per_prompt, num_channels, height, width, dtype, self.device, None),
                (batch_size * num_images_per_prompt, height, width, dtype, self.device, None),
            ]
            for args in latent_args:
                try:
                    return self.pipe.prepare_latents(*args)
                except TypeError:
                    continue

        scale_factor = getattr(self.pipe, "vae_scale_factor", 8)
        height = height - height % scale_factor
        width = width - width % scale_factor
        num_channels = 4
        if hasattr(self.pipe, "vae"):
            num_channels = getattr(self.pipe.vae.config, "latent_channels", num_channels)
        latents = torch.randn(
            (batch_size * num_images_per_prompt, num_channels, height // scale_factor, width // scale_factor),
            device=self.device,
            dtype=dtype,
        )
        if self.scheduler is not None and hasattr(self.scheduler, "init_noise_sigma"):
            latents = latents * self.scheduler.init_noise_sigma
        return latents

    def _decode_latents(self, latents: torch.Tensor) -> torch.Tensor:
        vae = getattr(self.pipe, "vae", None)
        if vae is None:
            raise RuntimeError("Pipeline does not expose VAE for decoding latents.")

        latents = latents.to(getattr(vae, "dtype", latents.dtype))
        vae_cfg = getattr(vae, "config", None)
        scaling = getattr(vae_cfg, "scaling_factor", 1.0)
        shift = getattr(vae_cfg, "shift_factor", 0.0)
        latents = (latents / scaling) + shift

        decoded = vae.decode(latents, return_dict=False)
        if isinstance(decoded, (tuple, list)):
            image = decoded[0]
        elif hasattr(decoded, "sample"):
            image = decoded.sample
        else:
            image = decoded

        image = (image / 2 + 0.5).clamp(0, 1)
        return image

    def _tensor_to_pil(self, image_tensors: torch.Tensor) -> List[Image.Image]:
        import numpy as np

        images: List[Image.Image] = []
        if image_tensors.dim() == 3:
            image_tensors = image_tensors.unsqueeze(0)
        image_tensors = image_tensors.detach().float().cpu()
        for img in image_tensors:
            if img.shape[0] == 1:
                arr = img.squeeze(0).numpy()
                arr = (arr * 255).clip(0, 255).astype("uint8")
                images.append(Image.fromarray(arr, mode="L"))
            else:
                arr = img.permute(1, 2, 0).numpy()
                arr = (arr * 255).clip(0, 255).astype("uint8")
                images.append(Image.fromarray(arr))
        return images

    def _generate_no_grad(
        self,
        prompts: List[str],
        init_images: Optional[List[Image.Image]] = None,
        generation_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[List[Image.Image], Optional[torch.Tensor], Dict[str, Any]]:
        if init_images:
            logger.warning("Z-Image diffusers backend ignores init_images.")

        kwargs = dict(self.default_kwargs)
        if generation_kwargs:
            kwargs.update(generation_kwargs)

        seed = kwargs.pop("seed", None)
        if seed is not None:
            try:
                seed = int(seed)
                torch.manual_seed(seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(seed)
            except Exception:
                seed = None

        def _call_pipe(call_kwargs: Dict[str, Any]):
            with torch.no_grad():
                return self.pipe(prompt=prompts, **call_kwargs)

        try:
            output = _call_pipe(kwargs)
        except TypeError:
            if "cfg_normalization" in kwargs:
                kwargs.pop("cfg_normalization", None)
                output = _call_pipe(kwargs)
            else:
                raise

        images = None
        if hasattr(output, "images"):
            images = output.images
        elif isinstance(output, tuple) and output:
            images = output[0]
        else:
            images = output

        if images is None:
            raise RuntimeError("Z-Image pipeline returned no images")

        extra = {
            "backend": "diffusers",
            "num_inference_steps": kwargs.get("num_inference_steps"),
            "guidance_scale": kwargs.get("guidance_scale"),
            "height": kwargs.get("height"),
            "width": kwargs.get("width"),
            "seed": seed,
            "negative_prompt": kwargs.get("negative_prompt"),
        }

        return images, None, extra

    def _generate_with_grad(
        self,
        prompts: List[str],
        init_images: Optional[List[Image.Image]] = None,
        grad_last_steps_override: Optional[int] = None,
        generation_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[List[Image.Image], Optional[torch.Tensor], Dict[str, Any]]:
        if init_images:
            logger.warning("Z-Image diffusers backend ignores init_images.")

        kwargs = dict(self.default_kwargs)
        if generation_kwargs:
            kwargs.update(generation_kwargs)

        seed = kwargs.pop("seed", None)
        if seed is not None:
            try:
                seed = int(seed)
                torch.manual_seed(seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(seed)
            except Exception:
                seed = None

        guidance_scale = float(kwargs.get("guidance_scale", 0.0) or 0.0)
        num_steps = int(kwargs.get("num_inference_steps", 50))
        height = int(kwargs.get("height", 512))
        width = int(kwargs.get("width", 512))
        negative_prompt = kwargs.get("negative_prompt")
        cfg_normalization = bool(kwargs.get("cfg_normalization", False))
        cfg_truncation = kwargs.get("cfg_truncation", 1.0)

        if self.scheduler is None:
            raise RuntimeError("Pipeline scheduler is missing; cannot run manual diffusion loop.")

        num_images_per_prompt = 1
        do_cfg = guidance_scale > 1e-5
        batch_size = len(prompts)

        vae_scale = getattr(self.pipe, "vae_scale_factor", 8) * 2
        if vae_scale > 0:
            height = max(vae_scale, (height // vae_scale) * vae_scale)
            width = max(vae_scale, (width // vae_scale) * vae_scale)

        prompt_embeds, negative_prompt_embeds = self._encode_prompt(
            prompts,
            num_images_per_prompt=num_images_per_prompt,
            do_cfg=do_cfg,
            negative_prompt=negative_prompt,
        )

        if do_cfg and len(negative_prompt_embeds) != len(prompt_embeds):
            logger.warning("Negative prompt embeddings are unavailable; disabling CFG in gradient path.")
            do_cfg = False
            negative_prompt_embeds = []

        transformer = getattr(self.pipe, "transformer", None)
        if transformer is None:
            raise RuntimeError("Pipeline does not expose transformer.")

        transformer_dtype = getattr(transformer, "dtype", torch.float32)
        prompt_embeds = [pe.to(device=self.device, dtype=transformer_dtype) for pe in prompt_embeds]
        negative_prompt_embeds = [pe.to(device=self.device, dtype=transformer_dtype) for pe in negative_prompt_embeds]

        latents = self._prepare_latents(
            batch_size=batch_size,
            num_images_per_prompt=num_images_per_prompt,
            height=height,
            width=width,
            dtype=transformer_dtype,
        )
        latents = latents.to(torch.float32)

        image_seq_len = (latents.shape[2] // 2) * (latents.shape[3] // 2)
        sched_cfg = getattr(self.scheduler, "config", {})
        base_seq_len = sched_cfg.get("base_image_seq_len", 256)
        max_seq_len = sched_cfg.get("max_image_seq_len", 4096)
        base_shift = sched_cfg.get("base_shift", 0.5)
        max_shift = sched_cfg.get("max_shift", 1.15)
        mu = (image_seq_len - base_seq_len) * (max_shift - base_shift) / max(max_seq_len - base_seq_len, 1) + base_shift

        if hasattr(self.scheduler, "sigma_min"):
            self.scheduler.sigma_min = 0.0

        try:
            self.scheduler.set_timesteps(num_steps, device=self.device, mu=mu)
        except TypeError:
            try:
                self.scheduler.set_timesteps(num_steps, device=self.device)
            except TypeError:
                self.scheduler.set_timesteps(num_steps)

        timesteps = list(self.scheduler.timesteps)
        num_timesteps = len(timesteps)

        grad_last_steps = grad_last_steps_override
        if grad_last_steps is None:
            grad_last_steps = _env_get_int("Z_IMAGE_GRAD_LAST_STEPS", 1)
        if grad_last_steps is None or grad_last_steps <= 0:
            grad_last_steps = num_timesteps
        grad_last_steps = min(max(1, int(grad_last_steps)), num_timesteps)

        use_autocast = bool(self.device.startswith("cuda")) and _env_get_bool("Z_IMAGE_GRAD_AUTOCAST", True)
        actual_batch_size = batch_size * num_images_per_prompt

        for step_idx, t in enumerate(timesteps):
            keep_grad = step_idx >= (num_timesteps - grad_last_steps)
            grad_ctx = contextlib.nullcontext() if keep_grad else torch.no_grad()
            autocast_ctx = (
                torch.autocast(device_type="cuda", dtype=transformer_dtype, enabled=use_autocast)
                if use_autocast
                else contextlib.nullcontext()
            )

            with grad_ctx:
                with autocast_ctx:
                    timestep = t.expand(latents.shape[0])
                    timestep = (1000 - timestep) / 1000
                    t_norm = float(timestep[0].item())

                    current_guidance_scale = guidance_scale
                    if do_cfg and cfg_truncation is not None:
                        try:
                            trunc = float(cfg_truncation)
                            if trunc <= 1 and t_norm > trunc:
                                current_guidance_scale = 0.0
                        except Exception:
                            pass

                    apply_cfg = do_cfg and current_guidance_scale > 0

                    if apply_cfg:
                        latents_typed = latents.to(transformer_dtype)
                        latent_model_input = latents_typed.repeat(2, 1, 1, 1)
                        prompt_embeds_model_input = prompt_embeds + negative_prompt_embeds
                        timestep_model_input = timestep.repeat(2)
                    else:
                        latent_model_input = latents.to(transformer_dtype)
                        prompt_embeds_model_input = prompt_embeds
                        timestep_model_input = timestep

                    latent_model_input = latent_model_input.unsqueeze(2)
                    latent_model_input_list = list(latent_model_input.unbind(dim=0))

                    try:
                        model_out_list = transformer(
                            latent_model_input_list,
                            timestep_model_input,
                            prompt_embeds_model_input,
                            return_dict=False,
                        )[0]
                    except TypeError:
                        model_out_list = transformer(
                            latent_model_input_list,
                            timestep_model_input,
                            prompt_embeds_model_input,
                        )[0]

                    if apply_cfg:
                        pos_out = model_out_list[:actual_batch_size]
                        neg_out = model_out_list[actual_batch_size:]
                        noise_pred = []
                        for i in range(actual_batch_size):
                            pos = pos_out[i].float()
                            neg = neg_out[i].float()
                            pred = pos + current_guidance_scale * (pos - neg)

                            if cfg_normalization:
                                ori_pos_norm = torch.linalg.vector_norm(pos)
                                new_pos_norm = torch.linalg.vector_norm(pred)
                                max_new_norm = ori_pos_norm
                                if new_pos_norm > max_new_norm and new_pos_norm > 0:
                                    pred = pred * (max_new_norm / new_pos_norm)

                            noise_pred.append(pred)
                        noise_pred = torch.stack(noise_pred, dim=0)
                    else:
                        noise_pred = torch.stack([item.float() for item in model_out_list], dim=0)

                    noise_pred = -noise_pred.squeeze(2)

                step_output = self.scheduler.step(noise_pred.to(torch.float32), t, latents, return_dict=False)
                if isinstance(step_output, (tuple, list)):
                    latents = step_output[0]
                elif hasattr(step_output, "prev_sample"):
                    latents = step_output.prev_sample
                else:
                    latents = step_output

                if not keep_grad:
                    latents = latents.detach()

        image_tensors = self._decode_latents(latents)
        images = self._tensor_to_pil(image_tensors)

        extra = {
            "backend": "diffusers",
            "num_inference_steps": num_steps,
            "guidance_scale": guidance_scale,
            "height": height,
            "width": width,
            "image_tensors": image_tensors,
            "grad_last_steps": grad_last_steps,
            "seed": seed,
            "negative_prompt": negative_prompt,
        }

        return images, None, extra
    def generate(
        self,
        prompts: List[str],
        init_images: Optional[List[Image.Image]] = None,
        generation_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[List[Image.Image], Optional[torch.Tensor], Dict[str, Any]]:
        if self.enable_grad:
            try:
                return self._generate_with_grad(
                    prompts,
                    init_images=init_images,
                    generation_kwargs=generation_kwargs,
                )
            except torch.OutOfMemoryError as exc:
                logger.warning("Z-Image grad path OOM. Retrying with grad on last step only.")
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                try:
                    return self._generate_with_grad(
                        prompts,
                        init_images=init_images,
                        grad_last_steps_override=1,
                        generation_kwargs=generation_kwargs,
                    )
                except Exception as retry_exc:
                    if _env_get_bool("Z_IMAGE_OOM_FALLBACK_NO_GRAD", True):
                        logger.warning("Z-Image grad retry failed (%s). Falling back to no-grad generation.", retry_exc)
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        return self._generate_no_grad(
                            prompts,
                            init_images=init_images,
                            generation_kwargs=generation_kwargs,
                        )
                    raise RuntimeError(
                        "Z-Image gradient generation failed after OOM retry."
                    ) from retry_exc
            except Exception as exc:
                raise RuntimeError(
                    "Z-Image gradient generation failed. Ensure the pipeline components are compatible."
                ) from exc
        return self._generate_no_grad(prompts, init_images=init_images, generation_kwargs=generation_kwargs)

    def policy_gradient_step(
        self,
        prompts: List[str],
        images: List[Image.Image],
        log_probs: Optional[torch.Tensor],
        advantages: Optional[torch.Tensor],
    ) -> Dict[str, Any]:
        loss = None
        if log_probs is not None and advantages is not None:
            if not isinstance(log_probs, torch.Tensor):
                log_probs = torch.tensor(log_probs)
            if not isinstance(advantages, torch.Tensor):
                advantages = torch.tensor(advantages)
            loss = -(log_probs * advantages.detach()).mean()
        elif advantages is not None:
            if not isinstance(advantages, torch.Tensor):
                advantages = torch.tensor(advantages)
            loss = -advantages.mean()

        if loss is None:
            return {"policy_update_skipped": True, "backend": "diffusers"}

        if not isinstance(loss, torch.Tensor) or not loss.requires_grad:
            return {"policy_update_skipped": True, "backend": "diffusers"}

        if self.optimizer is None:
            return {"policy_update_skipped": True, "backend": "diffusers"}

        self.optimizer.zero_grad()
        loss.backward()
        if self.max_grad_norm and self.trainable_params:
            torch.nn.utils.clip_grad_norm_(self.trainable_params, self.max_grad_norm)
        self.optimizer.step()

        return {"policy_loss": float(loss.detach().cpu().item()), "backend": "diffusers"}

    def save_checkpoint(self, checkpoint_path: str) -> None:
        save_full = _env_get_bool("Z_IMAGE_SAVE_FULL_CHECKPOINTS", False)
        if not save_full:
            os.makedirs(checkpoint_path, exist_ok=True)
            with open(os.path.join(checkpoint_path, "README.txt"), "w", encoding="utf-8") as f:
                f.write("Skipped full diffusers save. Set Z_IMAGE_SAVE_FULL_CHECKPOINTS=1 to enable.\n")
            return

        if hasattr(self.pipe, "save_pretrained"):
            try:
                self.pipe.save_pretrained(checkpoint_path)
            except OSError as exc:
                logger.warning("Failed to save full z-image checkpoint: %s", exc)

    def load_checkpoint(self, checkpoint_path: str) -> None:
        if not os.path.exists(checkpoint_path):
            return
        try:
            from diffusers import ZImagePipeline
        except Exception:
            return

        dtype = _select_torch_dtype(self.device, os.getenv("Z_IMAGE_DTYPE"))
        self.pipe = ZImagePipeline.from_pretrained(
            checkpoint_path,
            torch_dtype=dtype,
            local_files_only=True,
        )
        if self.device:
            self.pipe.to(self.device)
        self.scheduler = getattr(self.pipe, "scheduler", None)
        self._configure_trainable_modules(self.pipe)
        self.optimizer, self.trainable_params = self._init_optimizer()


class _PythonBackend:
    def __init__(
        self,
        module,
        model_path: str,
        device: str,
        learning_rate: float,
        max_grad_norm: float,
    ):
        self.module = module
        self.model_path = model_path
        if str(device).lower().startswith("cuda") and not torch.cuda.is_available():
            logger.warning("CUDA not available for Z-Image; falling back to CPU.")
            device = "cpu"
        self.device = device
        self.learning_rate = learning_rate
        self.max_grad_norm = max_grad_norm
        self.model = self._init_model()
        self.optimizer = self._init_optimizer()

    def _init_model(self):
        if hasattr(self.module, "load"):
            return _call_with_fallback(self.module.load, model_path=self.model_path, device=self.device)
        if hasattr(self.module, "ZImageModel"):
            cls = getattr(self.module, "ZImageModel")
            return _call_with_fallback(cls, model_path=self.model_path, device=self.device)
        if hasattr(self.module, "build"):
            return _call_with_fallback(self.module.build, model_path=self.model_path, device=self.device)
        raise RuntimeError("z-image module must define load(), ZImageModel, or build().")

    def _init_optimizer(self):
        if hasattr(self.module, "get_optimizer"):
            return _call_with_fallback(self.module.get_optimizer, model=self.model, lr=self.learning_rate)
        if hasattr(self.model, "optimizer"):
            return self.model.optimizer
        if hasattr(self.model, "parameters"):
            params = [p for p in self.model.parameters() if p.requires_grad]
            if params:
                return torch.optim.AdamW(params, lr=self.learning_rate)
        return None

    def _normalize_generate_output(self, output):
        extra: Dict[str, Any] = {}
        images: List[Image.Image]
        log_probs: Optional[torch.Tensor] = None

        if isinstance(output, dict):
            images = output.get("images")
            log_probs = output.get("log_probs")
            extra = {k: v for k, v in output.items() if k not in ("images", "log_probs")}
        elif isinstance(output, tuple):
            if len(output) == 2:
                images, log_probs = output
            elif len(output) >= 3:
                images, log_probs, extra = output[0], output[1], output[2]
            else:
                images = list(output)
        else:
            images = output

        if log_probs is not None and not isinstance(log_probs, torch.Tensor):
            log_probs = torch.tensor(log_probs)

        return images, log_probs, extra

    def generate(
        self,
        prompts: List[str],
        init_images: Optional[List[Image.Image]] = None,
        generation_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[List[Image.Image], Optional[torch.Tensor], Dict[str, Any]]:
        if not hasattr(self.model, "generate"):
            raise RuntimeError("z-image model must implement generate()")

        if generation_kwargs is None:
            generation_kwargs = {}
        output = _call_with_fallback(
            self.model.generate,
            prompts=prompts,
            init_images=init_images,
            generation_kwargs=generation_kwargs,
            **generation_kwargs,
        )
        images, log_probs, extra = self._normalize_generate_output(output)

        if images is None:
            raise RuntimeError("z-image generate() returned no images")

        return images, log_probs, extra

    def policy_gradient_step(
        self,
        prompts: List[str],
        images: List[Image.Image],
        log_probs: Optional[torch.Tensor],
        advantages: Optional[torch.Tensor],
    ) -> Dict[str, Any]:
        if hasattr(self.model, "policy_gradient_step"):
            return _call_with_fallback(
                self.model.policy_gradient_step,
                prompts=prompts,
                images=images,
                log_probs=log_probs,
                advantages=advantages,
            )

        loss = None
        if hasattr(self.model, "loss"):
            loss_out = _call_with_fallback(
                self.model.loss,
                prompts=prompts,
                images=images,
                advantages=advantages,
            )
            if isinstance(loss_out, dict):
                loss = loss_out.get("loss") or loss_out.get("policy_loss")
            else:
                loss = loss_out
        elif log_probs is not None and advantages is not None:
            if not isinstance(log_probs, torch.Tensor):
                log_probs = torch.tensor(log_probs)
            if not isinstance(advantages, torch.Tensor):
                advantages = torch.tensor(advantages)
            loss = -(log_probs * advantages.detach()).mean()
        elif advantages is not None:
            if not isinstance(advantages, torch.Tensor):
                advantages = torch.tensor(advantages)
            loss = -advantages.mean()

        if loss is None:
            return {"policy_update_skipped": True}

        if not isinstance(loss, torch.Tensor) or not loss.requires_grad:
            return {"policy_update_skipped": True, "loss": float(torch.as_tensor(loss).detach().cpu().item())}

        if self.optimizer is None:
            return {"policy_update_skipped": True, "loss": float(loss.detach().cpu().item())}

        self.optimizer.zero_grad()
        loss.backward()
        if self.max_grad_norm and hasattr(self.model, "parameters"):
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
        self.optimizer.step()

        return {"policy_loss": float(loss.detach().cpu().item())}

    def save_checkpoint(self, checkpoint_path: str) -> None:
        if hasattr(self.model, "save_checkpoint"):
            self.model.save_checkpoint(checkpoint_path)
        elif hasattr(self.model, "state_dict"):
            os.makedirs(checkpoint_path, exist_ok=True)
            torch.save(self.model.state_dict(), os.path.join(checkpoint_path, "z_image.pt"))

    def load_checkpoint(self, checkpoint_path: str) -> None:
        if hasattr(self.model, "load_checkpoint"):
            self.model.load_checkpoint(checkpoint_path)
        else:
            state_path = os.path.join(checkpoint_path, "z_image.pt")
            if os.path.exists(state_path) and hasattr(self.model, "load_state_dict"):
                self.model.load_state_dict(torch.load(state_path, map_location=self.device))


class _StubBackend:
    """Fallback backend for wiring/testing without a real z-image model."""

    def generate(
        self,
        prompts: List[str],
        init_images: Optional[List[Image.Image]] = None,
        generation_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[List[Image.Image], Optional[torch.Tensor], Dict[str, Any]]:
        images: List[Image.Image] = []
        for i, _ in enumerate(prompts):
            if init_images and i < len(init_images) and init_images[i] is not None:
                images.append(init_images[i])
            else:
                images.append(Image.new("RGB", (256, 256), (128, 128, 128)))
        return images, None, {"backend": "stub"}

    def policy_gradient_step(
        self,
        prompts: List[str],
        images: List[Image.Image],
        log_probs: Optional[torch.Tensor],
        advantages: Optional[torch.Tensor],
    ) -> Dict[str, Any]:
        return {"policy_update_skipped": True}
























