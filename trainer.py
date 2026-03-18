"""
Adaptive PPO trainer (text and z-image modes).
"""

import hashlib
import json
import logging
import os
import random
import time
from datetime import datetime
from typing import List, Dict, Tuple, Optional, Any

import torch
import torch.nn.functional as F

from models import PPOConfig, PPOMemory, ValueNetwork, HybridRewardModel
from transformers import (
    Qwen2VLForConditionalGeneration,
    Qwen2VLProcessor,
    get_cosine_schedule_with_warmup,
)
from peft import get_peft_model, LoraConfig, TaskType
from PIL import Image

from adaptive import AdaptiveController, AdaptiveConfig, RewardWeightOptimizer
from z_image_policy import ZImagePolicy

logger = logging.getLogger(__name__)

try:
    from reward_models import EnhancedRewardModel, EnhancedRewardConfig
    ENHANCED_REWARD_AVAILABLE = True
except ImportError:
    ENHANCED_REWARD_AVAILABLE = False
    logger.warning("Enhanced reward model not available; using default reward model.")

try:
    from reward_models import DifferentiableImageRewardModel
    DIFF_REWARD_AVAILABLE = True
except ImportError:
    DIFF_REWARD_AVAILABLE = False
    logger.warning("Differentiable reward model not available; z-image gradients disabled.")

try:
    from qwen_judge import QwenVLJudge
    QWEN_JUDGE_AVAILABLE = True
except ImportError:
    QWEN_JUDGE_AVAILABLE = False
    logger.warning("Qwen judge not available; Qwen evaluation disabled.")


class AdaptivePPOTrainer:
    """Adaptive PPO trainer."""

    def __init__(self, config: PPOConfig, use_adaptive: bool = True):
        self.config = config
        self.use_adaptive = use_adaptive
        if str(config.device).lower().startswith("cuda") and not torch.cuda.is_available():
            logger.warning("CUDA not available; falling back to CPU.")
            config.device = "cpu"
        self.device = torch.device(config.device)
        self.actor_type = getattr(config, "actor_type", "qwen-vl")

        if self.actor_type == "z-image":
            self.z_image = ZImagePolicy(
                model_path=getattr(config, "z_image_model_path", None),
                device=config.device,
                allow_stub=getattr(config, "allow_zimage_stub", False),
                learning_rate=config.learning_rate,
                max_grad_norm=config.max_grad_norm,
            )
            self.processor = None
            self.policy_model = None
            self.ref_model = None
            self.value_model = None
            self.policy_optimizer = None
            self.value_optimizer = None
            self.policy_scheduler = None
            self.value_scheduler = None
            self.scaler = None
            self.memory = None
            self.reward_baseline = 0.0
            self.reward_baseline_alpha = getattr(config, "reward_baseline_alpha", 0.1)
        else:
            # Text-based policy (Qwen2-VL)
            self.processor = Qwen2VLProcessor.from_pretrained(config.model_name)

            logger.info(f"Loading policy model: {config.model_name}")
            self.policy_model = Qwen2VLForConditionalGeneration.from_pretrained(
                config.model_name,
                torch_dtype=torch.bfloat16,
                device_map="auto",
            )

            lora_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                r=config.lora_r,
                lora_alpha=config.lora_alpha,
                lora_dropout=config.lora_dropout,
                target_modules=config.target_modules,
            )
            self.policy_model = get_peft_model(self.policy_model, lora_config)
            self.policy_model.print_trainable_parameters()

            # Reference model for KL
            self.ref_model = Qwen2VLForConditionalGeneration.from_pretrained(
                config.model_name,
                torch_dtype=torch.bfloat16,
                device_map="auto",
            )
            self.ref_model.eval()
            for param in self.ref_model.parameters():
                param.requires_grad = False

            logger.info("Initializing value network")
            self.value_model = ValueNetwork(config.model_name)

        # Reward model (shared)
        logger.info("Initializing reward model")
        if getattr(config, "use_enhanced_reward", False) and ENHANCED_REWARD_AVAILABLE:
            reward_config = EnhancedRewardConfig(
                model_name=config.model_name,
                reward_model_path=config.reward_model_path,
                use_aesthetic_reward=config.use_aesthetic_reward,
                aesthetic_model_name=config.aesthetic_model_name,
                gray_smoothness_weight=config.gray_smoothness_weight,
                line_smoothness_weight=config.line_smoothness_weight,
                noise_artifact_weight=config.noise_artifact_weight,
                human_weight=config.human_weight,
                aesthetic_weight=config.aesthetic_weight,
            )
            self.reward_model = EnhancedRewardModel(reward_config)
            logger.info(
                "Using enhanced reward model: gray/line smoothness + noise + human preference + aesthetic"
            )
        else:
            self.reward_model = HybridRewardModel(config)

        self.diff_reward_model = None
        if self.actor_type == "z-image":
            if DIFF_REWARD_AVAILABLE and os.getenv("Z_IMAGE_ENABLE_GRAD", "").lower() in ("1", "true", "yes", "y", "on"):
                edge_weight = float(os.getenv("Z_IMAGE_DIFF_EDGE_WEIGHT", "0.6"))
                tv_weight = float(os.getenv("Z_IMAGE_DIFF_TV_WEIGHT", "0.4"))
                color_weight = float(os.getenv("Z_IMAGE_DIFF_COLOR_WEIGHT", "0.05"))
                self.diff_reward_model = DifferentiableImageRewardModel(
                    edge_weight=edge_weight,
                    tv_weight=tv_weight,
                    color_weight=color_weight,
                )
                logger.info("Using differentiable reward model for z-image training")

        self.eval_output_dir = getattr(config, "eval_output_dir", "./eval_output")
        self.save_eval_every = max(0, int(getattr(config, "save_eval_every", 0) or 0))
        self.qwen_eval_every = max(0, int(getattr(config, "qwen_eval_every", self.save_eval_every) or 0))
        self.enable_wandb = getattr(config, "enable_wandb", False)
        self.wandb = None
        self.wandb_run = None
        if self.enable_wandb:
            self._init_wandb()

        self.schema_version = getattr(config, "schema_version", "v1")
        self.judge_prompt_version = getattr(config, "qwen_judge_prompt_version", "v1")
        self.run_id = getattr(config, "run_id", None) or f"run_{int(time.time())}"
        self.model_version = getattr(config, "z_image_model_path", None) or getattr(config, "model_name", None)
        self.judge_model_version = getattr(config, "qwen_judge_model_name", None)
        self.lora_version = getattr(config, "lora_version", None)

        self.qwen_judge = None
        self.qwen_reward_weight = getattr(config, "qwen_reward_weight", 1.0)
        self.qwen_score_weights = {
            "aesthetic": getattr(config, "qwen_aesthetic_weight", 0.4),
            "gray_smoothness": getattr(config, "qwen_gray_weight", 0.4),
            "noise_artifact": getattr(config, "qwen_noise_weight", 0.2),
            "prompt_alignment": getattr(config, "qwen_prompt_alignment_weight", 0.2),
        }
        self.qwen_confidence_low = float(getattr(config, "qwen_confidence_low", 0.3))
        self.qwen_confidence_high = float(getattr(config, "qwen_confidence_high", 0.7))
        self.qwen_gate_min_aesthetic = float(getattr(config, "qwen_gate_min_aesthetic", 3.0))
        self.qwen_gate_min_gray = float(getattr(config, "qwen_gate_min_gray", 3.0))
        self.qwen_gate_min_prompt_alignment = float(getattr(config, "qwen_gate_min_prompt_alignment", 3.0))
        self.qwen_gate_max_noise = float(getattr(config, "qwen_gate_max_noise", 7.0))
        self.qwen_gate_penalty = float(getattr(config, "qwen_gate_penalty", -1.0))
        fatal_labels = getattr(config, "qwen_fatal_labels", "") or ""
        self.qwen_fatal_labels = set([label.strip() for label in fatal_labels.split(",") if label.strip()])
        self.qwen_pairwise_samples = max(1, int(getattr(config, "qwen_pairwise_samples", 1) or 1))
        self.human_review_path = os.path.join(self.eval_output_dir, "human_reviews.jsonl")
        self.human_disagreement_path = os.path.join(self.eval_output_dir, "human_disagreements.jsonl")
        self.human_review_every = max(0, int(getattr(config, "human_review_every", 0) or 0))
        self.human_disagreement_threshold = float(getattr(config, "human_disagreement_threshold", 2.0))
        self._seen_human_reviews = set()
        self._eval_cache = {}
        self.qwen_pairwise_every = max(0, int(getattr(config, "qwen_pairwise_every", self.qwen_eval_every) or 0))

        # Task queue mode
        self.task_pass_threshold = float(getattr(config, "task_pass_threshold", 0.5))
        self.task_max_retry = max(1, int(getattr(config, "task_max_retry", 3) or 3))
        self.task_use_row_threshold = bool(getattr(config, "task_use_row_threshold", True))
        self.task_save_best_only = bool(getattr(config, "task_save_best_only", False))
        self.task_enable_auto_repair = bool(getattr(config, "task_enable_auto_repair", True))
        self.task_enable_negative_repair = bool(getattr(config, "task_enable_negative_repair", True))
        self.task_enable_sampling_repair = bool(getattr(config, "task_enable_sampling_repair", True))
        self.task_continue_on_fail = bool(getattr(config, "task_continue_on_fail", True))

        self.task_run_records_path = os.path.join(self.eval_output_dir, "task_run_records.jsonl")
        self.task_summary_path = os.path.join(self.eval_output_dir, "task_summary.jsonl")
        self.failed_tasks_path = os.path.join(self.eval_output_dir, "failed_tasks.jsonl")
        self.prompt_evolution_path = os.path.join(self.eval_output_dir, "prompt_evolution.jsonl")

        if self.actor_type == "z-image":
            self._init_qwen_judge()

        # Optimizers and schedulers (text policy only)
        if self.actor_type != "z-image":
            self.policy_optimizer = torch.optim.AdamW(
                self.policy_model.parameters(),
                lr=config.learning_rate,
                betas=(0.9, 0.999),
                weight_decay=0.01,
            )

            self.value_optimizer = torch.optim.AdamW(
                self.value_model.parameters(),
                lr=config.learning_rate,
                betas=(0.9, 0.999),
                weight_decay=0.01,
            )

            self.policy_scheduler = get_cosine_schedule_with_warmup(
                self.policy_optimizer,
                num_warmup_steps=config.warmup_steps,
                num_training_steps=10000,
            )

            self.value_scheduler = get_cosine_schedule_with_warmup(
                self.value_optimizer,
                num_warmup_steps=config.warmup_steps,
                num_training_steps=10000,
            )

            self.scaler = torch.cuda.amp.GradScaler() if config.mixed_precision else None
            self.memory = PPOMemory()

        # Training stats
        self.global_step = 0
        self.epoch = 0

        if self.use_adaptive:
            self._init_adaptive()

        os.makedirs(config.output_dir, exist_ok=True)
        os.makedirs(config.checkpoint_dir, exist_ok=True)
        os.makedirs(self.eval_output_dir, exist_ok=True)

    def _init_adaptive(self):
        adaptive_config = AdaptiveConfig(
            reward_threshold=0.8,
            early_stop_reward=0.95,
            early_stop_patience=5,
            kl_target=0.02,
            kl_adapt_rate=0.1,
            kl_min=0.01,
            kl_max=0.5,
            lr_decay_factor=0.95,
            lr_min=1e-7,
            lr_patience=3,
            clip_min=0.1,
            clip_max=0.3,
            entropy_min=0.001,
            entropy_target=0.05,
        )

        initial_params = {
            'kl_coef': self.config.kl_coef,
            'learning_rate': self.config.learning_rate,
            'clip_epsilon': self.config.clip_epsilon,
            'entropy_coef': self.config.entropy_coef,
        }

        self.adaptive_controller = AdaptiveController(adaptive_config, initial_params)
        self.reward_optimizer = RewardWeightOptimizer(
            human_weight=0.7,
            aesthetic_weight=0.3,
        )

        logger.info("Adaptive controller initialized")

    def _init_wandb(self):
        try:
            import wandb
        except Exception as exc:
            logger.warning("W&B not available: %s", exc)
            return

        project = getattr(self.config, "wandb_project", None) or "z-image-rl"
        entity = getattr(self.config, "wandb_entity", None)
        name = getattr(self.config, "wandb_run_name", None)
        mode = getattr(self.config, "wandb_mode", "online") or "online"

        self.wandb_run = wandb.init(
            project=project,
            entity=entity,
            name=name,
            mode=mode,
            config=vars(self.config),
        )
        self.wandb = wandb

    def _init_qwen_judge(self):
        if not getattr(self.config, "use_qwen_judge", False):
            return
        if not QWEN_JUDGE_AVAILABLE:
            logger.warning("Qwen judge not available; skipping Qwen evaluation.")
            return
        model_name = getattr(self.config, "qwen_judge_model_name", None)
        if not model_name:
            logger.warning("Qwen judge model name not set; skipping Qwen evaluation.")
            return

        judge_device = getattr(self.config, "qwen_judge_device", None) or str(self.device)
        if str(judge_device).lower().startswith("cuda") and not torch.cuda.is_available():
            logger.warning("Qwen judge requested CUDA but CUDA is unavailable; fallback to CPU.")
            judge_device = "cpu"

        local_files_only = getattr(self.config, "qwen_judge_local_files_only", None)
        trust_remote_code = bool(getattr(self.config, "qwen_judge_trust_remote_code", True))

        self.qwen_judge = QwenVLJudge(
            model_name=model_name,
            device=str(judge_device),
            system_prompt=getattr(self.config, "qwen_judge_system_prompt", ""),
            max_new_tokens=getattr(self.config, "qwen_judge_max_new_tokens", 512),
            temperature=getattr(self.config, "qwen_judge_temperature", 0.2),
            top_p=getattr(self.config, "qwen_judge_top_p", 0.9),
            local_files_only=local_files_only,
            trust_remote_code=trust_remote_code,
        )
        logger.info("Qwen judge initialized on %s", judge_device)

    def _compute_qwen_reward(self, eval_results, device, return_meta: bool = False):
        rewards = []
        confidences = []
        gate_hits = []
        base_scores = []
        for result in eval_results:
            scores = result.get("scores", {}) if isinstance(result, dict) else {}
            aesthetic = float(scores.get("aesthetic") or 0.0)
            gray = float(scores.get("gray_smoothness") or 0.0)
            noise = float(scores.get("noise_artifact") or 0.0)
            alignment = float(scores.get("prompt_alignment") or 0.0)
            conf = result.get("confidence") if isinstance(result, dict) else None
            labels = result.get("labels", []) if isinstance(result, dict) else []
            labels = set([str(label) for label in labels if label is not None])

            gate_hit = False
            if aesthetic < self.qwen_gate_min_aesthetic:
                gate_hit = True
            if gray < self.qwen_gate_min_gray:
                gate_hit = True
            if alignment < self.qwen_gate_min_prompt_alignment:
                gate_hit = True
            if noise > self.qwen_gate_max_noise:
                gate_hit = True
            if self.qwen_fatal_labels and labels.intersection(self.qwen_fatal_labels):
                gate_hit = True

            positive = (
                self.qwen_score_weights.get("aesthetic", 0.4) * aesthetic
                + self.qwen_score_weights.get("gray_smoothness", 0.4) * gray
                + self.qwen_score_weights.get("prompt_alignment", 0.2) * alignment
            )
            penalty = self.qwen_score_weights.get("noise_artifact", 0.2) * noise

            base_score = positive - penalty
            reward = base_score / 10.0
            if gate_hit:
                reward += self.qwen_gate_penalty

            if conf is None:
                conf_scale = 1.0
                conf_value = 1.0
            else:
                conf_value = float(conf)
                if conf_value < self.qwen_confidence_low:
                    conf_scale = 0.0
                elif conf_value >= self.qwen_confidence_high:
                    conf_scale = 1.0
                else:
                    denom = max(self.qwen_confidence_high - self.qwen_confidence_low, 1e-6)
                    conf_scale = (conf_value - self.qwen_confidence_low) / denom

            reward = reward * conf_scale
            rewards.append(reward)
            confidences.append(conf_value)
            gate_hits.append(1.0 if gate_hit else 0.0)
            base_scores.append(base_score)

        if not rewards:
            rewards = [0.0]
        tensor = torch.tensor(rewards, device=device)
        meta = {
            "confidence": confidences,
            "gate_hit": gate_hits,
            "base_score": base_scores,
            "reward": rewards,
        }
        if return_meta:
            return tensor, meta
        return tensor

    def _prompt_id(self, text: str) -> str:
        if not text:
            return ""
        return hashlib.sha1(text.encode("utf-8")).hexdigest()[:12]

    def _record_id(self, prompt_id: str, image_path: Optional[str], idx: int) -> str:
        raw = f"{prompt_id}|{image_path or ''}|{self.global_step}|{idx}"
        return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]

    def _save_image(self, img, prefix: str, idx: int) -> Optional[str]:
        if img is None:
            return None
        os.makedirs(self.eval_output_dir, exist_ok=True)
        timestamp = int(time.time() * 1000)
        filename = f"{prefix}_step{self.global_step}_idx{idx}_{timestamp}.png"
        image_path = os.path.join(self.eval_output_dir, filename)
        try:
            img.save(image_path)
            return image_path
        except Exception as exc:
            logger.warning("Failed to save image: %s", exc)
            return None

    def _log_eval_batch(self, prompts, images, qwen_eval, reward_components, eval_tag: str, force: bool,
                        sampling: Optional[Dict[str, Any]] = None, qwen_meta: Optional[Dict[str, Any]] = None):
        if not force:
            if self.save_eval_every <= 0:
                return
            if self.global_step % self.save_eval_every != 0:
                return
        if not images:
            return

        os.makedirs(self.eval_output_dir, exist_ok=True)
        record_path = os.path.join(self.eval_output_dir, "eval_records.jsonl")
        prompt_path = os.path.join(self.eval_output_dir, "prompt_suggestions.jsonl")
        review_path = os.path.join(self.eval_output_dir, "review_queue.jsonl")
        human_queue_path = os.path.join(self.eval_output_dir, "human_review_queue.jsonl")

        prompt_texts = [p.get("text", "") for p in prompts]

        for idx, img in enumerate(images):
            if img is None:
                continue
            image_path = self._save_image(img, eval_tag, idx)

            eval_item = qwen_eval[idx] if qwen_eval and idx < len(qwen_eval) else {}
            scores = eval_item.get("scores", {}) if isinstance(eval_item, dict) else {}
            confidence = eval_item.get("confidence") if isinstance(eval_item, dict) else None
            labels = eval_item.get("labels", []) if isinstance(eval_item, dict) else []

            prompt_text = prompt_texts[idx] if idx < len(prompt_texts) else ""
            prompt_id = self._prompt_id(prompt_text)
            record_id = self._record_id(prompt_id, image_path, idx)

            qwen_reward = None
            qwen_base_score = None
            qwen_gate_hit = None
            if qwen_meta and isinstance(qwen_meta, dict):
                if idx < len(qwen_meta.get("reward", [])):
                    qwen_reward = qwen_meta.get("reward")[idx]
                if idx < len(qwen_meta.get("base_score", [])):
                    qwen_base_score = qwen_meta.get("base_score")[idx]
                if idx < len(qwen_meta.get("gate_hit", [])):
                    qwen_gate_hit = qwen_meta.get("gate_hit")[idx]

            record = {
                "schema_version": self.schema_version,
                "judge_prompt_version": self.judge_prompt_version,
                "model_version": self.model_version,
                "judge_model_version": self.judge_model_version,
                "lora_version": self.lora_version,
                "run_id": self.run_id,
                "eval_tag": eval_tag,
                "step": self.global_step,
                "record_id": record_id,
                "prompt_id": prompt_id,
                "prompt": prompt_text,
                "image_path": image_path,
                "sampling": sampling or {},
                "scores": scores,
                "confidence": confidence,
                "labels": labels,
                "critique": eval_item.get("critique") if isinstance(eval_item, dict) else None,
                "prompt_optimization": eval_item.get("prompt_optimization") if isinstance(eval_item, dict) else None,
                "qwen_reward": qwen_reward,
                "qwen_base_score": qwen_base_score,
                "qwen_gate_hit": qwen_gate_hit,
            }

            if reward_components:
                reward_payload = {}
                for key, value in reward_components.items():
                    if hasattr(value, "mean"):
                        reward_payload[key] = float(value.detach().mean().item())
                if reward_payload:
                    record["reward_components"] = reward_payload

            try:
                with open(record_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")
            except Exception as exc:
                logger.warning("Failed to write eval record: %s", exc)

            if eval_item and eval_item.get("prompt_optimization"):
                optimized = eval_item.get("prompt_optimization")
                suggestion = {
                    "schema_version": self.schema_version,
                    "judge_prompt_version": self.judge_prompt_version,
                    "run_id": self.run_id,
                    "step": self.global_step,
                    "record_id": record_id,
                    "original_prompt_id": prompt_id,
                    "optimized_prompt_id": self._prompt_id(str(optimized)),
                    "original_prompt": prompt_text,
                    "optimized_prompt": optimized,
                    "scores": scores,
                    "confidence": confidence,
                    "labels": labels,
                    "critique": eval_item.get("critique") if isinstance(eval_item, dict) else None,
                }
                try:
                    with open(prompt_path, "a", encoding="utf-8") as f:
                        f.write(json.dumps(suggestion, ensure_ascii=False) + "\n")
                except Exception as exc:
                    logger.warning("Failed to write prompt suggestion: %s", exc)

            if confidence is not None and confidence < self.qwen_confidence_low:
                try:
                    with open(review_path, "a", encoding="utf-8") as f:
                        f.write(json.dumps(record, ensure_ascii=False) + "\n")
                except Exception as exc:
                    logger.warning("Failed to write review queue: %s", exc)

            if hasattr(self, "_eval_cache"):
                self._eval_cache[record_id] = {
                    "qwen_base_score": qwen_base_score,
                    "qwen_reward": qwen_reward,
                    "prompt": prompt_text,
                    "image_path": image_path,
                }

            human_queue = {
                "record_id": record_id,
                "prompt_id": prompt_id,
                "prompt": prompt_text,
                "image_path": image_path,
                "sampling": sampling or {},
                "qwen_scores": scores,
                "qwen_base_score": qwen_base_score,
                "qwen_reward": qwen_reward,
                "qwen_confidence": confidence,
                "labels": labels,
                "critique": record.get("critique"),
                "prompt_optimization": record.get("prompt_optimization"),
                "run_id": self.run_id,
                "step": self.global_step,
                "eval_tag": eval_tag,
                "human_score": None,
                "human_labels": [],
                "human_comment": None,
            }
            try:
                with open(human_queue_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(human_queue, ensure_ascii=False) + "\n")
            except Exception as exc:
                logger.warning("Failed to write human review queue: %s", exc)

            if self.wandb is not None:
                log_payload = {
                    "generated_image": self.wandb.Image(img, caption=prompt_text),
                    "qwen_aesthetic": scores.get("aesthetic"),
                    "qwen_gray_smoothness": scores.get("gray_smoothness"),
                    "qwen_noise_artifact": scores.get("noise_artifact"),
                    "qwen_prompt_alignment": scores.get("prompt_alignment"),
                    "qwen_confidence": confidence,
                    "qwen_base_score": qwen_base_score,
                    "qwen_reward": qwen_reward,
                }
                for key, value in reward_components.items():
                    if hasattr(value, "mean"):
                        log_payload[f"reward_{key}"] = float(value.detach().mean().item())
                self.wandb.log(log_payload, step=self.global_step)

    def _maybe_log_eval(self, prompts, images, qwen_eval, reward_components, sampling=None, qwen_meta=None):
        self._log_eval_batch(
            prompts,
            images,
            qwen_eval,
            reward_components,
            eval_tag="train",
            force=False,
            sampling=sampling,
            qwen_meta=qwen_meta,
        )

    def _maybe_collect_pairwise(self, prompts, base_images, sampling=None):
        if self.qwen_judge is None:
            return
        if self.qwen_pairwise_samples < 2:
            return
        if self.qwen_pairwise_every <= 0 or self.global_step % self.qwen_pairwise_every != 0:
            return

        os.makedirs(self.eval_output_dir, exist_ok=True)
        pair_path = os.path.join(self.eval_output_dir, "pairwise.jsonl")
        prompt_texts = [p.get("text", "") for p in prompts]

        for idx, prompt_text in enumerate(prompt_texts):
            if idx >= len(base_images):
                continue
            base_image = base_images[idx]
            if base_image is None:
                continue

            extra_needed = self.qwen_pairwise_samples - 1
            extra_images = []
            if extra_needed > 0:
                try:
                    extra_prompts = [prompt_text] * extra_needed
                    with torch.no_grad():
                        extra_images, _, _ = self.z_image.generate(extra_prompts, init_images=None)
                except Exception as exc:
                    logger.warning("Pairwise extra generation failed: %s", exc)
                    extra_images = []

            images = [base_image] + list(extra_images)
            image_paths = []
            for j, img in enumerate(images):
                image_paths.append(self._save_image(img, "pairwise", j))

            prompt_id = self._prompt_id(prompt_text)
            group_id = f"{prompt_id}_step{self.global_step}"

            for i in range(len(images)):
                for j in range(i + 1, len(images)):
                    try:
                        result = self.qwen_judge.compare_pair(images[i], images[j], prompt_text)
                    except Exception as exc:
                        logger.warning("Pairwise judge failed: %s", exc)
                        result = {"better": None, "confidence": None, "reason": None, "labels": []}

                    record = {
                        "schema_version": self.schema_version,
                        "judge_prompt_version": self.judge_prompt_version,
                        "model_version": self.model_version,
                        "judge_model_version": self.judge_model_version,
                        "run_id": self.run_id,
                        "group_id": group_id,
                        "step": self.global_step,
                        "prompt": prompt_text,
                        "prompt_id": prompt_id,
                        "sampling": sampling or {},
                        "image_a": image_paths[i],
                        "image_b": image_paths[j],
                        "better": result.get("better"),
                        "confidence": result.get("confidence"),
                        "reason": result.get("reason"),
                        "labels": result.get("labels"),
                    }
                    try:
                        with open(pair_path, "a", encoding="utf-8") as f:
                            f.write(json.dumps(record, ensure_ascii=False) + "\n")
                    except Exception as exc:
                        logger.warning("Failed to write pairwise record: %s", exc)

    def _maybe_ingest_human_reviews(self):
        if self.human_review_every <= 0:
            return
        if self.global_step % self.human_review_every != 0:
            return
        if not self.human_review_path or not os.path.exists(self.human_review_path):
            return

        try:
            with open(self.human_review_path, "r", encoding="utf-8") as f:
                lines = f.readlines()
        except Exception as exc:
            logger.warning("Failed to read human reviews: %s", exc)
            return

        for line in lines:
            line = line.strip()
            if not line:
                continue
            try:
                review = json.loads(line)
            except Exception:
                continue
            record_id = review.get("record_id")
            if not record_id or record_id in self._seen_human_reviews:
                continue

            self._seen_human_reviews.add(record_id)
            qwen_info = self._eval_cache.get(record_id, {}) if hasattr(self, "_eval_cache") else {}
            qwen_score = qwen_info.get("qwen_base_score")
            human_score = review.get("human_score")

            if human_score is None or qwen_score is None:
                continue

            try:
                diff = abs(float(human_score) - float(qwen_score))
            except Exception:
                continue

            if diff >= self.human_disagreement_threshold:
                record = {
                    "record_id": record_id,
                    "prompt_id": review.get("prompt_id"),
                    "prompt": review.get("prompt"),
                    "image_path": review.get("image_path"),
                    "qwen_score": qwen_score,
                    "human_score": human_score,
                    "diff": diff,
                    "labels": review.get("human_labels"),
                    "comment": review.get("human_comment"),
                    "run_id": self.run_id,
                    "step": review.get("step"),
                }
                try:
                    with open(self.human_disagreement_path, "a", encoding="utf-8") as f:
                        f.write(json.dumps(record, ensure_ascii=False) + "\n")
                except Exception as exc:
                    logger.warning("Failed to write disagreement record: %s", exc)

    def run_fixed_eval(self, dataloader, tag: str = "fixed"):
        if self.actor_type != "z-image":
            return
        if dataloader is None:
            return

        for batch in dataloader:
            prompts = batch['prompts']
            images = batch.get('images', None)
            try:
                with torch.no_grad():
                    generated_images, _, extra = self.generate_images(prompts, images)
            except Exception as exc:
                logger.warning("Fixed eval generation failed: %s", exc)
                continue

            sampling = {}
            if isinstance(extra, dict):
                for key in ("num_inference_steps", "guidance_scale", "height", "width"):
                    if key in extra:
                        sampling[key] = extra[key]

            qwen_eval = None
            if self.qwen_judge is not None:
                try:
                    prompt_texts = [p.get('text', '') for p in prompts]
                    qwen_eval = self.qwen_judge.evaluate_batch(generated_images, prompt_texts)
                except Exception as exc:
                    logger.warning("Fixed eval judge failed: %s", exc)
                    qwen_eval = None

            self._log_eval_batch(prompts, generated_images, qwen_eval, reward_components={}, eval_tag=tag, force=True, sampling=sampling)

    def _append_jsonl_record(self, path: str, record: Dict[str, Any]) -> None:
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        except Exception as exc:
            logger.warning("Failed to write JSONL record (%s): %s", path, exc)

    def _safe_float(self, value: Any, default: float = 0.0) -> float:
        try:
            if value is None:
                return default
            return float(value)
        except Exception:
            return default

    def _task_sampling_defaults(self) -> Dict[str, Any]:
        defaults: Dict[str, Any] = {}
        backend = getattr(getattr(self, "z_image", None), "backend", None)
        backend_defaults = getattr(backend, "default_kwargs", None)
        if isinstance(backend_defaults, dict):
            for key in ("num_inference_steps", "guidance_scale", "height", "width", "negative_prompt"):
                if key in backend_defaults:
                    defaults[key] = backend_defaults[key]

        # Environment variables have higher priority for runtime speed tuning.
        if os.getenv("Z_IMAGE_STEPS"):
            defaults["num_inference_steps"] = int(float(os.getenv("Z_IMAGE_STEPS")))
        if os.getenv("Z_IMAGE_GUIDANCE"):
            defaults["guidance_scale"] = float(os.getenv("Z_IMAGE_GUIDANCE"))
        if os.getenv("Z_IMAGE_HEIGHT"):
            defaults["height"] = int(float(os.getenv("Z_IMAGE_HEIGHT")))
        if os.getenv("Z_IMAGE_WIDTH"):
            defaults["width"] = int(float(os.getenv("Z_IMAGE_WIDTH")))

        defaults["seed"] = random.randint(1, 2_147_483_647)
        return defaults

    def _coerce_prompt_optimization_text(self, eval_item: Dict[str, Any]) -> Optional[str]:
        if not isinstance(eval_item, dict):
            return None
        value = eval_item.get("prompt_optimization")
        if isinstance(value, str):
            text = value.strip()
            return text or None

        if isinstance(value, dict):
            for key in ("new_prompt", "optimized_prompt", "prompt", "text"):
                v = value.get(key)
                if isinstance(v, str) and v.strip():
                    return v.strip()
                if isinstance(v, dict):
                    nested = v.get("text") or v.get("prompt")
                    if isinstance(nested, str) and nested.strip():
                        return nested.strip()

        if isinstance(value, list):
            for item in value:
                if isinstance(item, str) and item.strip():
                    return item.strip()
        return None

    def _dedupe_csv_terms(self, *raw_values: Any) -> str:
        terms: List[str] = []
        seen = set()
        for raw in raw_values:
            if raw is None:
                continue
            text = str(raw)
            text = text.replace(";", ",")
            for part in text.split(","):
                term = part.strip()
                if not term:
                    continue
                key = term.lower()
                if key in seen:
                    continue
                seen.add(key)
                terms.append(term)
        return ", ".join(terms)

    def _evaluate_single_qwen(self, image: Image.Image, prompt_text: str) -> Tuple[Dict[str, Any], float, Dict[str, Any], Optional[str]]:
        if self.qwen_judge is None:
            fallback = {
                "raw_text": None,
                "scores": {},
                "confidence": None,
                "labels": [],
                "critique": "qwen_judge_not_enabled",
                "prompt_optimization": None,
            }
            meta = {
                "confidence": [0.0],
                "gate_hit": [1.0],
                "base_score": [0.0],
                "reward": [0.0],
            }
            return fallback, 0.0, meta, "qwen_judge_not_enabled"

        try:
            eval_item = self.qwen_judge.evaluate(image, prompt_text)
        except Exception as exc:
            fallback = {
                "raw_text": None,
                "scores": {},
                "confidence": None,
                "labels": [],
                "critique": f"qwen_judge_error: {exc}",
                "prompt_optimization": None,
            }
            meta = {
                "confidence": [0.0],
                "gate_hit": [1.0],
                "base_score": [0.0],
                "reward": [0.0],
            }
            return fallback, 0.0, meta, str(exc)

        try:
            qwen_rewards, qwen_meta = self._compute_qwen_reward([eval_item], torch.device("cpu"), return_meta=True)
            qwen_reward = float(qwen_rewards.detach().cpu().view(-1)[0].item())
        except Exception as exc:
            qwen_reward = 0.0
            qwen_meta = {
                "confidence": [self._safe_float(eval_item.get("confidence"), 0.0)],
                "gate_hit": [1.0],
                "base_score": [0.0],
                "reward": [0.0],
            }
            return eval_item, qwen_reward, qwen_meta, f"qwen_reward_error: {exc}"

        return eval_item, qwen_reward, qwen_meta, None

    def _build_fail_reasons(
        self,
        eval_item: Dict[str, Any],
        qwen_reward: float,
        threshold: float,
        qwen_meta: Optional[Dict[str, Any]],
        judge_error: Optional[str],
    ) -> List[str]:
        reasons: List[str] = []
        if judge_error:
            reasons.append(judge_error)

        scores = eval_item.get("scores", {}) if isinstance(eval_item, dict) else {}
        aesthetic = self._safe_float(scores.get("aesthetic"), 0.0)
        gray = self._safe_float(scores.get("gray_smoothness"), 0.0)
        noise = self._safe_float(scores.get("noise_artifact"), 0.0)
        alignment = self._safe_float(scores.get("prompt_alignment"), 0.0)
        labels = set([str(l) for l in (eval_item.get("labels", []) or []) if l is not None])

        gate_hit = False
        if qwen_meta and isinstance(qwen_meta, dict):
            gate_values = qwen_meta.get("gate_hit", [])
            if gate_values:
                gate_hit = float(gate_values[0]) > 0.5

        fatal_labels = sorted(labels.intersection(self.qwen_fatal_labels)) if self.qwen_fatal_labels else []

        if qwen_reward < threshold:
            reasons.append("reward_below_threshold")
        if gate_hit:
            reasons.append("qwen_gate_hit")
        if fatal_labels:
            reasons.append("fatal_labels:" + ",".join(fatal_labels))
        if aesthetic < self.qwen_gate_min_aesthetic:
            reasons.append("low_aesthetic")
        if gray < self.qwen_gate_min_gray:
            reasons.append("low_gray_smoothness")
        if alignment < self.qwen_gate_min_prompt_alignment:
            reasons.append("low_prompt_alignment")
        if noise > self.qwen_gate_max_noise:
            reasons.append("high_noise_artifact")

        dedup: List[str] = []
        seen = set()
        for r in reasons:
            if r in seen:
                continue
            seen.add(r)
            dedup.append(r)
        return dedup

    def _auto_repair(
        self,
        current_prompt: str,
        current_negative_prompt: str,
        current_sampling: Dict[str, Any],
        eval_item: Dict[str, Any],
        fail_reasons: List[str],
    ) -> Tuple[str, str, Dict[str, Any], Dict[str, Any]]:
        labels = set([str(x).lower() for x in (eval_item.get("labels", []) or []) if x is not None])
        critique = str(eval_item.get("critique") or "")

        new_prompt = current_prompt
        new_negative = current_negative_prompt
        new_sampling = dict(current_sampling)

        repair_types: List[str] = []
        repair_reason_parts: List[str] = list(fail_reasons or [])
        repair_source = "fallback"

        prompt_opt = self._coerce_prompt_optimization_text(eval_item)
        if prompt_opt and prompt_opt != current_prompt:
            new_prompt = prompt_opt
            repair_types.append("prompt_rewrite")
            repair_source = "qwen_prompt_optimization"
            repair_reason_parts.append("used_prompt_optimization")
        else:
            hint_chunks: List[str] = []
            if "prompt_mismatch" in labels:
                hint_chunks.append("strict prompt alignment, preserve main subject and composition")
            if "gray_band" in labels or "dirty_edge" in labels:
                hint_chunks.append("smooth grayscale gradients, clean edges, smooth shading")
            if "broken_line" in labels:
                hint_chunks.append("clean linework, smooth line continuity")
            if "noise" in labels or "severe_artifact" in labels:
                hint_chunks.append("artifact-free texture, no noise speckles")
            if "structure_collapse" in labels or "local_collapse" in labels:
                hint_chunks.append("stable structure, coherent geometry, avoid collapse")
            if hint_chunks:
                hint_text = ", ".join(hint_chunks)
                if hint_text.lower() not in current_prompt.lower():
                    new_prompt = f"{current_prompt}. {hint_text}"
                    repair_types.append("prompt_rewrite")
                    repair_reason_parts.append("rule_prompt_rewrite")
                    repair_source = "rule_based"

        if self.task_enable_negative_repair:
            neg_parts: List[str] = []
            if "noise" in labels:
                neg_parts.extend(["noise", "grain", "speckles", "low quality"])
            if "dirty_edge" in labels:
                neg_parts.extend(["dirty edge", "jagged edge", "halo"])
            if "broken_line" in labels:
                neg_parts.extend(["broken line", "line discontinuity", "messy lines"])
            if "gray_band" in labels:
                neg_parts.extend(["gray banding", "posterization"])
            if "severe_artifact" in labels:
                neg_parts.extend(["severe artifact", "distortion", "glitch"])
            if "blurry" in critique.lower():
                neg_parts.extend(["blurry", "out of focus"])
            if "collapse" in critique.lower():
                neg_parts.extend(["deformed", "warped geometry"])

            merged_negative = self._dedupe_csv_terms(current_negative_prompt, ", ".join(neg_parts))
            if merged_negative != current_negative_prompt:
                new_negative = merged_negative
                repair_types.append("negative_prompt_append")
                repair_reason_parts.append("rule_negative_repair")
                if repair_source == "fallback":
                    repair_source = "rule_based"

        if self.task_enable_sampling_repair:
            old_seed = new_sampling.get("seed")
            new_sampling["seed"] = random.randint(1, 2_147_483_647)
            if old_seed != new_sampling["seed"]:
                repair_types.append("seed_change")

            steps = int(new_sampling.get("num_inference_steps") or 20)
            guidance = float(new_sampling.get("guidance_scale") or 4.0)
            changed_steps = False
            changed_guidance = False

            if "noise" in labels or "severe_artifact" in labels:
                new_steps = min(steps + 2, 80)
                if new_steps != steps:
                    new_sampling["num_inference_steps"] = new_steps
                    changed_steps = True
                new_guidance = max(1.0, guidance - 0.3)
                if abs(new_guidance - guidance) > 1e-6:
                    new_sampling["guidance_scale"] = new_guidance
                    changed_guidance = True

            if "prompt_mismatch" in labels:
                guidance_now = float(new_sampling.get("guidance_scale") or guidance)
                new_guidance = min(guidance_now + 0.5, 9.0)
                if abs(new_guidance - guidance_now) > 1e-6:
                    new_sampling["guidance_scale"] = new_guidance
                    changed_guidance = True

            if "structure_collapse" in labels or "local_collapse" in labels:
                steps_now = int(new_sampling.get("num_inference_steps") or steps)
                new_steps = min(steps_now + 4, 80)
                if new_steps != steps_now:
                    new_sampling["num_inference_steps"] = new_steps
                    changed_steps = True

            if changed_steps:
                repair_types.append("step_increase")
            if changed_guidance:
                repair_types.append("guidance_adjust")
            if changed_steps or changed_guidance:
                repair_reason_parts.append("rule_sampling_repair")
                if repair_source == "fallback":
                    repair_source = "rule_based"

        if not repair_types:
            repair_types = ["fallback_seed_change"]
            new_sampling["seed"] = random.randint(1, 2_147_483_647)
            repair_reason_parts.append("fallback_seed_change")

        # Keep deterministic order while removing duplicates.
        dedup_types: List[str] = []
        seen_types = set()
        for item in repair_types:
            if item in seen_types:
                continue
            seen_types.add(item)
            dedup_types.append(item)

        repair_action = {
            "repair_type": "+".join(dedup_types),
            "repair_reason": "; ".join(repair_reason_parts) if repair_reason_parts else "repair_needed",
            "repair_source": repair_source,
            "old_prompt": current_prompt,
            "new_prompt": new_prompt,
            "old_negative_prompt": current_negative_prompt,
            "new_negative_prompt": new_negative,
            "old_sampling": current_sampling,
            "new_sampling": new_sampling,
        }
        return new_prompt, new_negative, new_sampling, repair_action

    def process_prompt_until_pass(self, task: Dict[str, Any]) -> Dict[str, Any]:
        if self.actor_type != "z-image":
            raise RuntimeError("Task queue mode requires actor_type=z-image")

        task_id = str(task.get("prompt_id") or f"row_{task.get('row_index', 0)}")
        row_index = int(task.get("row_index") or 0)

        original_prompt = str(task.get("prompt") or "").strip()
        if not original_prompt:
            raise RuntimeError(f"Task {task_id} has empty prompt")

        current_prompt = original_prompt
        current_negative = str(task.get("negative_prompt") or "").strip()

        row_max_retry = task.get("max_retry")
        max_retry = self.task_max_retry
        try:
            if row_max_retry is not None and str(row_max_retry).strip() != "":
                max_retry = max(1, int(float(row_max_retry)))
        except Exception:
            max_retry = self.task_max_retry

        threshold_source = "global"
        pass_threshold = float(self.task_pass_threshold)
        if self.task_use_row_threshold and task.get("target_score") is not None:
            try:
                pass_threshold = float(task.get("target_score"))
                threshold_source = "row"
            except Exception:
                pass

        sampling = self._task_sampling_defaults()
        if current_negative:
            sampling["negative_prompt"] = current_negative

        best_reward = float("-inf")
        best_scores: Dict[str, Any] = {}
        best_image_path: Optional[str] = None
        used_repair_types: List[str] = []
        attempt_image_paths: List[str] = []

        task_start_time = datetime.utcnow().isoformat() + "Z"
        final_status = "failed"
        attempt_count = 0
        pending_evolution: Optional[Dict[str, Any]] = None
        final_reward = 0.0
        final_scores: Dict[str, Any] = {}

        for attempt_idx in range(1, max_retry + 1):
            attempt_count = attempt_idx
            generation_kwargs = dict(sampling)
            if current_negative:
                generation_kwargs["negative_prompt"] = current_negative
            else:
                generation_kwargs.pop("negative_prompt", None)
            if not generation_kwargs.get("seed"):
                generation_kwargs["seed"] = random.randint(1, 2_147_483_647)

            generated_image = None
            image_path = None
            extra: Dict[str, Any] = {}
            eval_item: Dict[str, Any] = {
                "raw_text": None,
                "scores": {},
                "confidence": None,
                "labels": [],
                "critique": None,
                "prompt_optimization": None,
            }
            qwen_reward = 0.0
            qwen_meta: Dict[str, Any] = {
                "confidence": [0.0],
                "gate_hit": [1.0],
                "base_score": [0.0],
                "reward": [0.0],
            }
            judge_error: Optional[str] = None

            try:
                generated_images, _, extra = self.generate_images(
                    prompts=[{"text": current_prompt}],
                    images=None,
                    generation_kwargs=generation_kwargs,
                )
                if generated_images:
                    generated_image = generated_images[0]
                if generated_image is None:
                    raise RuntimeError("z-image returned no image")
                image_path = self._save_image(generated_image, f"task_{task_id}", attempt_idx)
                if image_path:
                    attempt_image_paths.append(image_path)

                eval_item, qwen_reward, qwen_meta, judge_error = self._evaluate_single_qwen(
                    generated_image,
                    current_prompt,
                )
            except Exception as exc:
                judge_error = str(exc)
                eval_item = {
                    "raw_text": None,
                    "scores": {},
                    "confidence": None,
                    "labels": [],
                    "critique": f"task_generation_error: {exc}",
                    "prompt_optimization": None,
                }
                qwen_reward = -1.0
                qwen_meta = {
                    "confidence": [0.0],
                    "gate_hit": [1.0],
                    "base_score": [0.0],
                    "reward": [qwen_reward],
                }

            if pending_evolution is not None:
                evo = dict(pending_evolution)
                evo["reward_after"] = qwen_reward
                evo["accepted"] = bool(qwen_reward > self._safe_float(evo.get("reward_before"), float("-inf")))
                evo["timestamp"] = datetime.utcnow().isoformat() + "Z"
                self._append_jsonl_record(self.prompt_evolution_path, evo)
                pending_evolution = None

            fail_reasons = self._build_fail_reasons(
                eval_item=eval_item,
                qwen_reward=qwen_reward,
                threshold=pass_threshold,
                qwen_meta=qwen_meta,
                judge_error=judge_error,
            )
            passed = len(fail_reasons) == 0

            scores = eval_item.get("scores", {}) if isinstance(eval_item, dict) else {}
            confidence = eval_item.get("confidence") if isinstance(eval_item, dict) else None
            labels = eval_item.get("labels", []) if isinstance(eval_item, dict) else []
            critique = eval_item.get("critique") if isinstance(eval_item, dict) else None
            prompt_optimization = eval_item.get("prompt_optimization") if isinstance(eval_item, dict) else None
            final_reward = qwen_reward
            final_scores = scores if isinstance(scores, dict) else {}

            sampling_used = dict(generation_kwargs)
            if isinstance(extra, dict):
                for key in ("num_inference_steps", "guidance_scale", "height", "width", "seed", "negative_prompt"):
                    if key in extra:
                        sampling_used[key] = extra[key]

            if qwen_reward > best_reward:
                best_reward = qwen_reward
                best_scores = scores if isinstance(scores, dict) else {}
                best_image_path = image_path

            repair_action: Dict[str, Any] = {
                "repair_type": "none",
                "repair_reason": "passed" if passed else "not_repaired",
                "repair_source": "none",
            }

            if not passed and attempt_idx < max_retry:
                if self.task_enable_auto_repair:
                    prev_prompt = current_prompt
                    current_prompt, current_negative, sampling, repair_action = self._auto_repair(
                        current_prompt=current_prompt,
                        current_negative_prompt=current_negative,
                        current_sampling=sampling,
                        eval_item=eval_item,
                        fail_reasons=fail_reasons,
                    )
                    used_repair_types.append(str(repair_action.get("repair_type")))

                    if current_prompt != prev_prompt:
                        pending_evolution = {
                            "schema_version": self.schema_version,
                            "run_id": self.run_id,
                            "task_id": task_id,
                            "prompt_id": task_id,
                            "row_index": row_index,
                            "parent_prompt": prev_prompt,
                            "child_prompt": current_prompt,
                            "repair_reason": repair_action.get("repair_reason"),
                            "reward_before": qwen_reward,
                            "reward_after": None,
                            "accepted": False,
                        }
                else:
                    sampling["seed"] = random.randint(1, 2_147_483_647)
                    repair_action = {
                        "repair_type": "seed_change",
                        "repair_reason": "auto_repair_disabled",
                        "repair_source": "fallback",
                        "new_sampling": dict(sampling),
                    }
                    used_repair_types.append("seed_change")

            record = {
                "schema_version": self.schema_version,
                "run_id": self.run_id,
                "task_id": task_id,
                "prompt_id": task_id,
                "row_index": row_index,
                "attempt_index": attempt_idx,
                "original_prompt": original_prompt,
                "current_prompt": current_prompt if passed else str(repair_action.get("old_prompt") or current_prompt),
                "used_prompt": str(repair_action.get("old_prompt") or current_prompt),
                "used_negative_prompt": str(repair_action.get("old_negative_prompt") or current_negative),
                "current_negative_prompt": current_negative,
                "sampling_params": sampling_used,
                "image_path": image_path,
                "qwen_scores": scores,
                "qwen_reward": qwen_reward,
                "confidence": confidence,
                "labels": labels,
                "critique": critique,
                "prompt_optimization": prompt_optimization,
                "pass_threshold": pass_threshold,
                "threshold_source": threshold_source,
                "passed": passed,
                "fail_reasons": fail_reasons,
                "repair_action": repair_action,
                "timestamp": datetime.utcnow().isoformat() + "Z",
            }
            self._append_jsonl_record(self.task_run_records_path, record)

            self.global_step += 1

            if passed:
                final_status = "passed"
                break

        if pending_evolution is not None:
            evo = dict(pending_evolution)
            evo["reward_after"] = None
            evo["accepted"] = False
            evo["timestamp"] = datetime.utcnow().isoformat() + "Z"
            self._append_jsonl_record(self.prompt_evolution_path, evo)

        if best_reward == float("-inf"):
            best_reward = 0.0

        if self.task_save_best_only and attempt_image_paths:
            for path in attempt_image_paths:
                if not path or path == best_image_path:
                    continue
                try:
                    if os.path.exists(path):
                        os.remove(path)
                except Exception as exc:
                    logger.warning("Failed to remove non-best task image %s: %s", path, exc)

        summary = {
            "schema_version": self.schema_version,
            "run_id": self.run_id,
            "task_id": task_id,
            "prompt_id": task_id,
            "row_index": row_index,
            "original_prompt": original_prompt,
            "final_prompt": current_prompt,
            "total_attempts": attempt_count,
            "execution_count": attempt_count,
            "final_status": final_status,
            "final_reward": final_reward,
            "final_score": final_reward,
            "final_scores": final_scores,
            "best_image_path": best_image_path,
            "best_reward": best_reward,
            "best_scores": best_scores,
            "used_repair_types": sorted(set([x for x in used_repair_types if x])),
            "start_time": task_start_time,
            "end_time": datetime.utcnow().isoformat() + "Z",
            "category": task.get("category"),
            "notes": task.get("notes"),
        }

        self._append_jsonl_record(self.task_summary_path, summary)
        if final_status != "passed":
            self._append_jsonl_record(self.failed_tasks_path, summary)

        return summary

    def run_prompt_task_queue(self, tasks: List[Dict[str, Any]]) -> Dict[str, Any]:
        if self.actor_type != "z-image":
            raise RuntimeError("Task queue mode requires actor_type=z-image")

        total = len(tasks)
        passed = 0
        failed = 0
        start_time = datetime.utcnow().isoformat() + "Z"
        task_results: List[Dict[str, Any]] = []

        logger.info("Starting task queue mode. tasks=%d", total)
        for idx, task in enumerate(tasks, start=1):
            task_id = task.get("prompt_id", f"row_{task.get('row_index', idx)}")
            try:
                logger.info("[Task %d/%d] processing prompt_id=%s", idx, total, task_id)
                summary = self.process_prompt_until_pass(task)
            except Exception as exc:
                failed += 1
                logger.error("Task failed unexpectedly prompt_id=%s error=%s", task_id, exc)
                summary = {
                    "schema_version": self.schema_version,
                    "run_id": self.run_id,
                    "task_id": task_id,
                    "prompt_id": task_id,
                    "row_index": task.get("row_index"),
                    "original_prompt": task.get("prompt"),
                    "final_prompt": task.get("prompt"),
                    "total_attempts": 0,
                    "execution_count": 0,
                    "final_status": "failed",
                    "final_reward": 0.0,
                    "final_score": 0.0,
                    "final_scores": {},
                    "best_image_path": None,
                    "best_reward": 0.0,
                    "best_scores": {},
                    "used_repair_types": [],
                    "start_time": start_time,
                    "end_time": datetime.utcnow().isoformat() + "Z",
                    "error": str(exc),
                }
                self._append_jsonl_record(self.task_summary_path, summary)
                self._append_jsonl_record(self.failed_tasks_path, summary)
                task_results.append({
                    "prompt_id": summary.get("prompt_id"),
                    "final_score": summary.get("final_score", 0.0),
                    "execution_count": summary.get("execution_count", summary.get("total_attempts", 0)),
                    "final_status": summary.get("final_status"),
                })
                if not self.task_continue_on_fail:
                    break
                continue

            if summary.get("final_status") == "passed":
                passed += 1
            else:
                failed += 1
                if not self.task_continue_on_fail:
                    logger.info("Stopping task queue because task_continue_on_fail=False")
                    task_results.append({
                        "prompt_id": summary.get("prompt_id"),
                        "final_score": summary.get("final_score", summary.get("final_reward", summary.get("best_reward", 0.0))),
                        "execution_count": summary.get("execution_count", summary.get("total_attempts", 0)),
                        "final_status": summary.get("final_status"),
                    })
                    break

            task_results.append({
                "prompt_id": summary.get("prompt_id"),
                "final_score": summary.get("final_score", summary.get("final_reward", summary.get("best_reward", 0.0))),
                "execution_count": summary.get("execution_count", summary.get("total_attempts", 0)),
                "final_status": summary.get("final_status"),
            })
            logger.info(
                "Task result prompt_id=%s final_score=%.4f execution_count=%s status=%s",
                summary.get("prompt_id"),
                float(summary.get("final_score", summary.get("final_reward", summary.get("best_reward", 0.0))) or 0.0),
                summary.get("execution_count", summary.get("total_attempts", 0)),
                summary.get("final_status"),
            )

        result = {
            "schema_version": self.schema_version,
            "run_id": self.run_id,
            "total_tasks": total,
            "passed_tasks": passed,
            "failed_tasks": failed,
            "task_results": task_results,
            "start_time": start_time,
            "end_time": datetime.utcnow().isoformat() + "Z",
        }
        logger.info(
            "Task queue completed. total=%d passed=%d failed=%d",
            result["total_tasks"],
            result["passed_tasks"],
            result["failed_tasks"],
        )
        return result

    def generate_images(
        self,
        prompts: List[Dict[str, Any]],
        images: Optional[List[Image.Image]] = None,
        generation_kwargs: Optional[Dict[str, Any]] = None,
    ):
        if self.actor_type != "z-image":
            raise RuntimeError("generate_images is only valid for z-image actor")
        texts = [p.get('text', '') for p in prompts]
        return self.z_image.generate(
            texts,
            init_images=images,
            generation_kwargs=generation_kwargs,
        )

    def generate(self, prompts: List[Dict[str, Any]],
                 images: Optional[List[Image.Image]] = None) -> Tuple[torch.Tensor, torch.Tensor, List[str]]:
        if self.actor_type == "z-image":
            raise RuntimeError("generate is only valid for text actor")

        if images:
            inputs = self.processor(
                text=[p['text'] for p in prompts],
                images=images,
                return_tensors="pt",
                padding=True,
            ).to(self.device)
        else:
            inputs = self.processor(
                text=[p['text'] for p in prompts],
                return_tensors="pt",
                padding=True,
            ).to(self.device)

        with torch.no_grad():
            outputs = self.policy_model.generate(
                **inputs,
                max_new_tokens=self.config.max_new_tokens,
                temperature=self.config.temperature,
                top_p=self.config.top_p,
                do_sample=True,
                return_dict_in_generate=True,
                output_scores=True,
                use_cache=True,
            )

        sequences = outputs.sequences
        log_probs = self._compute_log_probs(sequences, inputs)
        decoded_texts = self.processor.batch_decode(sequences, skip_special_tokens=True)
        return sequences, log_probs, decoded_texts

    def _compute_log_probs(self, sequences: torch.Tensor,
                           inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        with torch.no_grad():
            outputs = self.policy_model(
                input_ids=sequences,
                attention_mask=torch.ones_like(sequences),
                return_dict=True,
            )
            logits = outputs.logits
            log_probs = F.log_softmax(logits, dim=-1)

            target_log_probs = log_probs[:, :-1, :].gather(
                2, sequences[:, 1:].unsqueeze(-1)
            ).squeeze(-1)

            input_length = inputs['input_ids'].shape[1]
            new_token_log_probs = target_log_probs[:, input_length-1:]
            sequence_log_probs = new_token_log_probs.sum(dim=1)

        return sequence_log_probs

    def compute_kl_divergence(self, sequences: torch.Tensor,
                              inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        with torch.no_grad():
            ref_outputs = self.ref_model(
                input_ids=sequences,
                attention_mask=torch.ones_like(sequences),
                return_dict=True,
            )
            ref_logits = ref_outputs.logits
            ref_log_probs = F.log_softmax(ref_logits, dim=-1)

            policy_outputs = self.policy_model(
                input_ids=sequences,
                attention_mask=torch.ones_like(sequences),
                return_dict=True,
            )
            policy_logits = policy_outputs.logits
            policy_log_probs = F.log_softmax(policy_logits, dim=-1)

            input_length = inputs['input_ids'].shape[1]
            kl_div = (policy_log_probs - ref_log_probs)[:, input_length-1:-1, :]
            kl_div = kl_div.exp() - kl_div - 1

            kl_per_sequence = kl_div.sum(dim=[1, 2])
            kl_mean = kl_per_sequence.mean()

        return kl_mean

    def ppo_update(self, memory_batch: Dict[str, Any]) -> Dict[str, float]:
        if self.actor_type == "z-image":
            raise RuntimeError("ppo_update is only valid for text actor")

        total_policy_loss = 0
        total_value_loss = 0
        total_entropy_loss = 0
        total_kl_penalty = 0
        n_updates = 0

        states = memory_batch['states']
        actions = memory_batch['actions']
        old_log_probs = torch.stack(memory_batch['log_probs'])
        advantages = torch.tensor(memory_batch['advantages'], device=self.device)
        returns = torch.tensor(memory_batch['returns'], device=self.device)

        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        current_params = self.adaptive_controller.params if self.use_adaptive else {}
        clip_epsilon = current_params.get('clip_epsilon', self.config.clip_epsilon)
        kl_coef = current_params.get('kl_coef', self.config.kl_coef)
        entropy_coef = current_params.get('entropy_coef', self.config.entropy_coef)

        for _ in range(self.config.ppo_epochs):
            indices = torch.randperm(len(states))

            for start_idx in range(0, len(states), self.config.mini_batch_size):
                end_idx = start_idx + self.config.mini_batch_size
                mb_indices = indices[start_idx:end_idx]

                mb_states = [states[i] for i in mb_indices]
                mb_actions = [actions[i] for i in mb_indices]
                mb_old_log_probs = old_log_probs[mb_indices]
                mb_advantages = advantages[mb_indices]
                mb_returns = returns[mb_indices]

                batch_input_ids = torch.stack([s['input_ids'] for s in mb_states])
                batch_attention_mask = torch.stack([s['attention_mask'] for s in mb_states])

                with torch.cuda.amp.autocast(enabled=self.config.mixed_precision):
                    policy_outputs = self.policy_model(
                        input_ids=batch_input_ids,
                        attention_mask=batch_attention_mask,
                        return_dict=True,
                    )

                    logits = policy_outputs.logits
                    log_probs = F.log_softmax(logits, dim=-1)

                    new_log_probs = []
                    for i, action in enumerate(mb_actions):
                        action_log_probs = log_probs[i, :-1, :].gather(
                            1, action.unsqueeze(-1)
                        ).squeeze(-1).sum()
                        new_log_probs.append(action_log_probs)
                    new_log_probs = torch.stack(new_log_probs)

                    ratio = torch.exp(new_log_probs - mb_old_log_probs)

                    surr1 = ratio * mb_advantages
                    surr2 = torch.clamp(
                        ratio,
                        1 - clip_epsilon,
                        1 + clip_epsilon,
                    ) * mb_advantages
                    policy_loss = -torch.min(surr1, surr2).mean()

                    value_outputs = self.value_model(
                        input_ids=batch_input_ids,
                        attention_mask=batch_attention_mask,
                    )
                    value_pred = value_outputs

                    if self.config.value_clip > 0:
                        old_values = torch.tensor(
                            [memory_batch['values'][i] for i in mb_indices],
                            device=self.device,
                        )
                        value_pred_clipped = old_values + torch.clamp(
                            value_pred - old_values,
                            -self.config.value_clip,
                            self.config.value_clip,
                        )
                        value_loss1 = F.mse_loss(value_pred, mb_returns, reduction='none')
                        value_loss2 = F.mse_loss(value_pred_clipped, mb_returns, reduction='none')
                        value_loss = torch.max(value_loss1, value_loss2).mean()
                    else:
                        value_loss = F.mse_loss(value_pred, mb_returns)

                    kl_penalty = self.compute_kl_divergence(
                        batch_input_ids, {'input_ids': batch_input_ids}
                    )

                    entropy = -(log_probs * torch.exp(log_probs)).sum(dim=-1).mean()

                    loss = (
                        policy_loss +
                        self.config.value_coef * value_loss -
                        entropy_coef * entropy +
                        kl_coef * kl_penalty
                    )

                self.policy_optimizer.zero_grad()
                self.value_optimizer.zero_grad()

                if self.scaler:
                    self.scaler.scale(loss).backward()
                    self.scaler.unscale_(self.policy_optimizer)
                    self.scaler.unscale_(self.value_optimizer)
                else:
                    loss.backward()

                torch.nn.utils.clip_grad_norm_(
                    self.policy_model.parameters(),
                    self.config.max_grad_norm,
                )
                torch.nn.utils.clip_grad_norm_(
                    self.value_model.parameters(),
                    self.config.max_grad_norm,
                )

                if self.scaler:
                    self.scaler.step(self.policy_optimizer)
                    self.scaler.step(self.value_optimizer)
                    self.scaler.update()
                else:
                    self.policy_optimizer.step()
                    self.value_optimizer.step()

                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy_loss += entropy.item()
                total_kl_penalty += kl_penalty.item()
                n_updates += 1

        self.policy_scheduler.step()
        self.value_scheduler.step()

        return {
            'policy_loss': total_policy_loss / n_updates,
            'value_loss': total_value_loss / n_updates,
            'entropy': total_entropy_loss / n_updates,
            'kl_penalty': total_kl_penalty / n_updates,
            'total_loss': (total_policy_loss + total_value_loss) / n_updates,
        }

    def train_step(self, batch: Dict[str, Any]) -> Dict[str, float]:
        prompts = batch['prompts']
        images = batch.get('images', None)

        if self.actor_type == "z-image":
            generated_images, log_probs, extra = self.generate_images(prompts, images)

            prompt_texts = [p.get('text', '') for p in prompts]

            image_tensors = extra.get("image_tensors") if isinstance(extra, dict) else None
            if image_tensors is not None and self.diff_reward_model is not None:
                rewards, reward_components = self.diff_reward_model.compute_rewards(image_tensors=image_tensors)
            else:
                with torch.no_grad():
                    rewards, reward_components = self.reward_model.compute_rewards(
                        input_ids=None,
                        attention_mask=None,
                        generated_images=generated_images,
                    )

            qwen_eval = None
            qwen_meta = None
            if self.qwen_judge is not None:
                if self.qwen_eval_every <= 0 or self.global_step % self.qwen_eval_every == 0:
                    try:
                        qwen_eval = self.qwen_judge.evaluate_batch(generated_images, prompt_texts)
                    except Exception as exc:
                        logger.warning("Qwen judge evaluation failed: %s", exc)
                        qwen_eval = None

            if qwen_eval:
                qwen_rewards, qwen_meta = self._compute_qwen_reward(qwen_eval, rewards.device, return_meta=True)
                rewards = rewards + self.qwen_reward_weight * qwen_rewards
                reward_components['qwen'] = qwen_rewards.detach()
                if qwen_meta and qwen_meta.get("confidence") is not None:
                    reward_components['qwen_confidence'] = torch.tensor(qwen_meta["confidence"], device=rewards.device)

            sampling = {}
            if isinstance(extra, dict):
                for key in ("num_inference_steps", "guidance_scale", "height", "width"):
                    if key in extra:
                        sampling[key] = extra[key]

            mean_reward = rewards.detach().mean().item()
            advantages = rewards - self.reward_baseline
            self.reward_baseline = (
                (1 - self.reward_baseline_alpha) * self.reward_baseline
                + self.reward_baseline_alpha * mean_reward
            )

            policy_metrics = self.z_image.policy_gradient_step(
                prompts=[p.get('text', '') for p in prompts],
                images=generated_images,
                log_probs=log_probs,
                advantages=advantages,
            )

            self.global_step += 1

            log_dict = {
                'mean_reward': mean_reward,
                'reward': mean_reward,
                'policy_loss': policy_metrics.get('policy_loss', 0.0),
                'value_loss': policy_metrics.get('value_loss', 0.0),
                'entropy': policy_metrics.get('entropy', 0.0),
                'kl_penalty': policy_metrics.get('kl_penalty', 0.0),
                'total_loss': policy_metrics.get('total_loss', policy_metrics.get('policy_loss', 0.0)),
                'reward_baseline': self.reward_baseline,
            }

            for k, v in reward_components.items():
                if k != 'total':
                    log_dict[f'reward_{k}'] = v.detach().mean().item()

            if self.use_adaptive:
                adaptive_params = self.adaptive_controller.update({
                    'reward': mean_reward,
                    'kl': log_dict.get('kl_penalty', 0.0),
                    'entropy': log_dict.get('entropy', 0.0),
                    'loss': log_dict.get('total_loss', 0.0),
                })
                status = self.adaptive_controller.get_status()
                log_dict.update({
                    'adaptive_lr': adaptive_params.get('learning_rate'),
                    'adaptive_kl': adaptive_params.get('kl_coef'),
                    'adaptive_clip': adaptive_params.get('clip_epsilon'),
                    'adaptive_entropy': adaptive_params.get('entropy_coef'),
                    'best_reward': status['best_reward'],
                })
                if adaptive_params.get('should_stop'):
                    log_dict['should_stop'] = True
                    log_dict['stop_reason'] = adaptive_params.get('stop_reason', '')

            self._maybe_log_eval(prompts, generated_images, qwen_eval, reward_components)
            return log_dict

        sequences, log_probs, decoded_texts = self.generate(prompts, images)

        with torch.no_grad():
            reward_inputs = {
                'input_ids': sequences,
                'attention_mask': torch.ones_like(sequences),
            }
            rewards, reward_components = self.reward_model.compute_rewards(
                **reward_inputs, generated_images=None
            )

        with torch.no_grad():
            values = self.value_model(
                input_ids=sequences,
                attention_mask=torch.ones_like(sequences),
            )

        for i in range(len(prompts)):
            state = {
                'input_ids': sequences[i],
                'attention_mask': torch.ones_like(sequences[i]),
            }
            action = sequences[i, len(prompts[i]['text']):]

            self.memory.add(
                state=state,
                action=action,
                log_prob=log_probs[i],
                reward=rewards[i].item(),
                value=values[i].item(),
                done=True,
            )

        self.memory.compute_gae(
            gamma=self.config.gamma,
            gae_lambda=self.config.gae_lambda,
        )

        if len(self.memory) >= self.config.batch_size:
            batch_data = self.memory.get_batch(list(range(len(self.memory))))
            losses = self.ppo_update(batch_data)
            self.memory.clear()
        else:
            losses = {'policy_loss': 0, 'value_loss': 0, 'entropy': 0, 'kl_penalty': 0, 'total_loss': 0}

        self.global_step += 1

        log_dict = {
            **losses,
            'mean_reward': rewards.mean().item(),
            'reward': rewards.mean().item(),
            'kl': losses['kl_penalty'],
            'loss': losses['total_loss'],
        }

        for k, v in reward_components.items():
            if k != 'total':
                log_dict[f'reward_{k}'] = v.detach().mean().item()

        if self.use_adaptive:
            adaptive_params = self.adaptive_controller.update({
                'reward': rewards.mean().item(),
                'kl': losses['kl_penalty'],
                'entropy': losses['entropy'],
                'loss': losses['total_loss'],
            })

            self.reward_optimizer.update({
                'human': reward_components.get('human', torch.zeros(1)).mean().item(),
                'aesthetic': reward_components.get('aesthetic', torch.zeros(1)).mean().item(),
            })

            new_lr = adaptive_params.get('learning_rate', self.config.learning_rate)
            for param_group in self.policy_optimizer.param_groups:
                param_group['lr'] = new_lr
            for param_group in self.value_optimizer.param_groups:
                param_group['lr'] = new_lr

            status = self.adaptive_controller.get_status()
            log_dict.update({
                'adaptive_lr': new_lr,
                'adaptive_kl': adaptive_params.get('kl_coef'),
                'adaptive_clip': adaptive_params.get('clip_epsilon'),
                'adaptive_entropy': adaptive_params.get('entropy_coef'),
                'best_reward': status['best_reward'],
            })

            if adaptive_params.get('should_stop'):
                log_dict['should_stop'] = True
                log_dict['stop_reason'] = adaptive_params.get('stop_reason', '')

        return log_dict

    def save_checkpoint(self, epoch: int, step: int):
        checkpoint_path = os.path.join(
            self.config.checkpoint_dir,
            f"checkpoint_epoch{epoch}_step{step}",
        )
        os.makedirs(checkpoint_path, exist_ok=True)

        if self.actor_type == "z-image":
            save_full = os.getenv("Z_IMAGE_SAVE_FULL_CHECKPOINTS", "").lower() in ("1", "true", "yes", "y", "on")
            if hasattr(self, "z_image") and save_full:
                self.z_image.save_checkpoint(checkpoint_path)
            elif hasattr(self, "z_image"):
                logger.info("Skipping full z-image checkpoint (set Z_IMAGE_SAVE_FULL_CHECKPOINTS=1 to enable).")
            save_dict = {
                'epoch': epoch,
                'step': step,
                'global_step': self.global_step,
                'reward_baseline': getattr(self, "reward_baseline", 0.0),
            }
            if self.use_adaptive:
                save_dict['adaptive_params'] = self.adaptive_controller.params
                save_dict['best_reward'] = self.adaptive_controller.best_reward
            torch.save(save_dict, os.path.join(checkpoint_path, "trainer_state.pt"))
            logger.info(f"Checkpoint saved to {checkpoint_path}")
            return

        self.policy_model.save_pretrained(os.path.join(checkpoint_path, "policy_model"))
        torch.save(
            self.value_model.state_dict(),
            os.path.join(checkpoint_path, "value_model.pt"),
        )

        save_dict = {
            'policy_optimizer': self.policy_optimizer.state_dict(),
            'value_optimizer': self.value_optimizer.state_dict(),
            'policy_scheduler': self.policy_scheduler.state_dict(),
            'value_scheduler': self.value_scheduler.state_dict(),
            'epoch': epoch,
            'step': step,
            'global_step': self.global_step,
        }

        if self.use_adaptive:
            save_dict['adaptive_params'] = self.adaptive_controller.params
            save_dict['best_reward'] = self.adaptive_controller.best_reward

        torch.save(save_dict, os.path.join(checkpoint_path, "optimizer.pt"))

        logger.info(f"Checkpoint saved to {checkpoint_path}")

    def load_checkpoint(self, checkpoint_path: str):
        if self.actor_type == "z-image":
            if hasattr(self, "z_image"):
                self.z_image.load_checkpoint(checkpoint_path)
            state_path = os.path.join(checkpoint_path, "trainer_state.pt")
            if os.path.exists(state_path):
                state = torch.load(state_path)
                self.global_step = state.get('global_step', 0)
                self.reward_baseline = state.get('reward_baseline', getattr(self, "reward_baseline", 0.0))
                if self.use_adaptive and 'adaptive_params' in state:
                    self.adaptive_controller.params = state['adaptive_params']
                    self.adaptive_controller.best_reward = state.get('best_reward', float('-inf'))
            logger.info(f"Checkpoint loaded from {checkpoint_path}")
            return

        self.policy_model = Qwen2VLForConditionalGeneration.from_pretrained(
            os.path.join(checkpoint_path, "policy_model"),
        )
        self.value_model.load_state_dict(
            torch.load(os.path.join(checkpoint_path, "value_model.pt")),
        )

        optimizer_state = torch.load(os.path.join(checkpoint_path, "optimizer.pt"))
        self.policy_optimizer.load_state_dict(optimizer_state['policy_optimizer'])
        self.value_optimizer.load_state_dict(optimizer_state['value_optimizer'])
        self.policy_scheduler.load_state_dict(optimizer_state['policy_scheduler'])
        self.value_scheduler.load_state_dict(optimizer_state['value_scheduler'])
        self.global_step = optimizer_state['global_step']

        if self.use_adaptive and 'adaptive_params' in optimizer_state:
            self.adaptive_controller.params = optimizer_state['adaptive_params']
            self.adaptive_controller.best_reward = optimizer_state.get('best_reward', float('-inf'))

        logger.info(f"Checkpoint loaded from {checkpoint_path}")











































