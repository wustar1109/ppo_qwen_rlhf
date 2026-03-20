"""
Core models and config.
"""

from dataclasses import dataclass, field
import os
from typing import List, Dict, Tuple, Optional, Any

import torch
import torch.nn as nn

from transformers import Qwen2VLForConditionalGeneration
from PIL import Image


@dataclass
class PPOConfig:
    model_name: str = "Qwen/Qwen2-VL-7B-Instruct"
    reward_model_path: Optional[str] = None
    use_aesthetic_reward: bool = True
    aesthetic_model_name: str = "cafeai/cafe_aesthetic"

    # Z-Image actor
    actor_type: str = "z-image"
    z_image_model_path: Optional[str] = None
    allow_zimage_stub: bool = False
    reward_baseline_alpha: float = 0.1

    # Enhanced reward weights
    use_enhanced_reward: bool = True
    gray_smoothness_weight: float = 0.15
    line_smoothness_weight: float = 0.15
    noise_artifact_weight: float = 0.15
    human_weight: float = 0.35
    aesthetic_weight: float = 0.20

    # Qwen judge
    use_qwen_judge: bool = False
    qwen_judge_model_name: Optional[str] = "Qwen/Qwen3-VL-7B-Instruct"
    qwen_judge_device: Optional[str] = None
    qwen_judge_local_files_only: Optional[bool] = None
    qwen_judge_trust_remote_code: bool = True
    qwen_judge_log_raw_output: bool = True
    qwen_judge_strict_schema: bool = False
    qwen_judge_retry_on_invalid: bool = True
    qwen_judge_max_retry: int = 1
    qwen_judge_system_prompt: str = (
        "You are a senior AIGC image visual designer and prompt engineer. "
        "Judge generated images with strict technical and aesthetic standards. "
        "Scoring semantics: noise_artifact means artifact severity where 1 is clean/artifact-free and 10 is severe artifacts/noise; higher is worse. "
        "You must diagnose defects and provide repair advice while preserving subject identity and style identity. "
        "Do NOT introduce new subjects, scenes, color themes, camera angles, or art styles. "
        "Return JSON only with fields: "
        "scores {aesthetic, gray_smoothness, noise_artifact, prompt_alignment} (1-10), "
        "confidence (0-1), labels (list), critique, prompt_optimization {"
        "protected_subject_tokens, protected_style_tokens, must_keep_phrases, "
        "rewrite_prompt_preserve_subject_style, append_constraints, forbidden_new_subject_tokens, reason"
        "}. "
        "Use labels from: noise, dirty_edge, broken_line, gray_band, local_collapse, prompt_mismatch, structure_collapse, severe_artifact."
    )
    qwen_judge_max_new_tokens: int = 512
    qwen_judge_temperature: float = 0.2
    qwen_judge_top_p: float = 0.9
    qwen_eval_every: int = 50
    qwen_reward_weight: float = 1.0
    qwen_aesthetic_weight: float = 0.4
    qwen_gray_weight: float = 0.4
    qwen_noise_weight: float = 0.2
    qwen_prompt_alignment_weight: float = 0.2
    qwen_confidence_low: float = 0.3
    qwen_confidence_high: float = 0.7
    qwen_gate_min_aesthetic: float = 3.0
    qwen_gate_min_gray: float = 3.0
    qwen_gate_min_prompt_alignment: float = 3.0
    qwen_gate_max_noise: float = 7.0
    qwen_gate_penalty: float = -1.0
    qwen_fatal_labels: str = "structure_collapse,prompt_mismatch,severe_artifact"
    qwen_pairwise_samples: int = 1
    qwen_pairwise_every: int = 50
    qwen_noise_higher_is_worse: bool = True
    human_review_every: int = 0
    human_disagreement_threshold: float = 2.0

    schema_version: str = "v1"
    qwen_judge_prompt_version: str = "v2_designer_guarded"
    run_id: Optional[str] = None

    # Logging / outputs
    enable_wandb: bool = False
    wandb_project: Optional[str] = "z-image-rl"
    wandb_entity: Optional[str] = None
    wandb_run_name: Optional[str] = None
    wandb_mode: str = "online"
    eval_output_dir: str = "./eval_output"
    save_eval_every: int = 50

    # LoRA
    lora_r: int = 64
    lora_alpha: int = 16
    lora_dropout: float = 0.1
    target_modules: List[str] = field(default_factory=lambda: ["q_proj", "v_proj", "k_proj", "o_proj"])

    # PPO
    ppo_epochs: int = 4
    clip_epsilon: float = 0.2
    value_clip: float = 0.4
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    kl_coef: float = 0.1
    max_grad_norm: float = 1.0

    gamma: float = 0.99
    gae_lambda: float = 0.95

    batch_size: int = 4
    mini_batch_size: int = 1
    num_epochs: int = 3
    learning_rate: float = 1e-5
    warmup_steps: int = 100
    max_length: int = 512
    max_new_tokens: int = 256
    temperature: float = 0.7
    top_p: float = 0.9

    train_data_path: str = "data/train.json"
    eval_data_path: Optional[str] = None

    # Task queue mode (Excel/CSV)
    task_excel_path: Optional[str] = None
    task_sheet_name: Optional[str] = None
    task_max_retry: int = 3
    task_pass_threshold: float = 0.5
    task_use_row_threshold: bool = True
    task_start_row: int = 1
    task_end_row: int = 0
    task_save_best_only: bool = False
    task_enable_auto_repair: bool = True
    task_enable_negative_repair: bool = True
    task_enable_sampling_repair: bool = True
    task_continue_on_fail: bool = True
    task_mode_only: bool = False
    task_output_dir: Optional[str] = None
    task_export_learning_data: bool = False
    task_disable_training: bool = True
    task_repair_low_score_max: float = 3.0
    task_repair_mid_score_max: float = 6.0
    task_prompt_identity_guard: bool = True
    task_prompt_identity_min_keep_ratio: float = 0.6
    task_repair_use_original_anchor: bool = True

    output_dir: str = "./ppo_qwen_output"
    checkpoint_dir: str = "./checkpoints"
    log_interval: int = 10
    eval_interval: int = 100
    save_interval: int = 500

    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    mixed_precision: bool = True
    gradient_accumulation_steps: int = 4


class AestheticRewardModel(nn.Module):
    def __init__(self, model_name: str = "cafeai/cafe_aesthetic"):
        super().__init__()
        from transformers import AutoModelForImageClassification, AutoProcessor
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model = AutoModelForImageClassification.from_pretrained(model_name)
        self.model.eval()
        self.score_scale = 1.0
        self.score_shift = 0.0

    @torch.no_grad()
    def forward(self, images: List[Image.Image]) -> torch.Tensor:
        inputs = self.processor(images=images, return_tensors="pt").to(self.model.device)
        outputs = self.model(**inputs)
        if hasattr(outputs, 'logits'):
            scores = torch.softmax(outputs.logits, dim=-1)
            aesthetic_score = torch.sum(scores * torch.arange(scores.shape[-1], device=scores.device), dim=-1)
        else:
            aesthetic_score = outputs.logits.squeeze()
        rewards = (aesthetic_score / 10.0) * self.score_scale + self.score_shift
        return rewards


class HumanPreferenceRewardModel(nn.Module):
    def __init__(self, base_model_name: str, checkpoint_path: Optional[str] = None):
        super().__init__()
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            base_model_name, torch_dtype=torch.bfloat16, device_map="auto"
        )
        hidden_size = self.model.config.hidden_size
        self.reward_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size // 2, 1)
        )
        if checkpoint_path and os.path.exists(checkpoint_path):
            self.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))

    def forward(self, input_ids, attention_mask, pixel_values=None, image_grid_thw=None):
        outputs = self.model(
            input_ids=input_ids, attention_mask=attention_mask,
            pixel_values=pixel_values, image_grid_thw=image_grid_thw,
            output_hidden_states=True, return_dict=True
        )
        last_hidden_state = outputs.hidden_states[-1]
        last_token_hidden = last_hidden_state[:, -1, :]
        rewards = self.reward_head(last_token_hidden).squeeze(-1)
        return rewards


class HybridRewardModel(nn.Module):
    def __init__(self, config: PPOConfig):
        super().__init__()
        self.config = config
        self.human_reward_model = HumanPreferenceRewardModel(config.model_name, config.reward_model_path) if config.reward_model_path else None
        self.aesthetic_model = AestheticRewardModel(config.aesthetic_model_name) if config.use_aesthetic_reward else None
        self.human_weight = 0.7
        self.aesthetic_weight = 0.3

    @torch.no_grad()
    def compute_rewards(self, input_ids, attention_mask, pixel_values=None, image_grid_thw=None, generated_images=None):
        reward_components = {}

        if input_ids is not None:
            n_samples = input_ids.shape[0]
            device = input_ids.device
        elif generated_images is not None:
            n_samples = len(generated_images)
            device = 'cpu'
        else:
            n_samples = 1
            device = 'cpu'

        total_rewards = torch.zeros(n_samples, device=device)

        if self.human_reward_model is not None and input_ids is not None:
            human_rewards = self.human_reward_model(input_ids, attention_mask, pixel_values, image_grid_thw)
            reward_components['human'] = human_rewards
            total_rewards += self.human_weight * human_rewards

        if self.aesthetic_model is not None and generated_images is not None:
            aesthetic_rewards = self.aesthetic_model(generated_images)
            reward_components['aesthetic'] = aesthetic_rewards.to(device)
            total_rewards += self.aesthetic_weight * aesthetic_rewards.to(device)

        reward_components['total'] = total_rewards
        return total_rewards, reward_components


class ValueNetwork(nn.Module):
    def __init__(self, base_model_name: str):
        super().__init__()
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            base_model_name, torch_dtype=torch.bfloat16, device_map="auto"
        )
        hidden_size = self.model.config.hidden_size
        self.value_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size // 2, 1)
        )
        for module in self.value_head.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=1.0)
                nn.init.zeros_(module.bias)

    def forward(self, input_ids, attention_mask, pixel_values=None, image_grid_thw=None):
        outputs = self.model(
            input_ids=input_ids, attention_mask=attention_mask,
            pixel_values=pixel_values, image_grid_thw=image_grid_thw,
            output_hidden_states=True, return_dict=True
        )
        last_hidden_state = outputs.hidden_states[-1]
        last_token_hidden = last_hidden_state[:, -1, :]
        values = self.value_head(last_token_hidden).squeeze(-1)
        return values


class PPOMemory:
    def __init__(self):
        self.states = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.values = []
        self.dones = []
        self.advantages = []
        self.returns = []

    def add(self, state: Dict[str, torch.Tensor], action: torch.Tensor,
            log_prob: torch.Tensor, reward: float, value: float, done: bool):
        self.states.append(state)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.rewards.append(reward)
        self.values.append(value)
        self.dones.append(done)

    def compute_gae(self, gamma: float = 0.99, gae_lambda: float = 0.95):
        advantages = []
        gae = 0
        for t in reversed(range(len(self.rewards))):
            if t == len(self.rewards) - 1:
                next_value = 0
            else:
                next_value = self.values[t + 1]
            delta = self.rewards[t] + gamma * next_value * (1 - self.dones[t]) - self.values[t]
            gae = delta + gamma * gae_lambda * (1 - self.dones[t]) * gae
            advantages.insert(0, gae)
        self.advantages = advantages
        self.returns = [adv + val for adv, val in zip(advantages, self.values)]

    def clear(self):
        self.__init__()

    def get_batch(self, indices: List[int]) -> Dict[str, Any]:
        return {
            'states': [self.states[i] for i in indices],
            'actions': [self.actions[i] for i in indices],
            'log_probs': [self.log_probs[i] for i in indices],
            'rewards': [self.rewards[i] for i in indices],
            'values': [self.values[i] for i in indices],
            'advantages': [self.advantages[i] for i in indices],
            'returns': [self.returns[i] for i in indices],
        }

    def __len__(self):
        return len(self.states)
