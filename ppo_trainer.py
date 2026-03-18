"""
PPO RLHF Training for Qwen Vision-Language Model
基于图中的强化学习架构实现
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import (
    Qwen2VLForConditionalGeneration,
    Qwen2VLProcessor,
    get_cosine_schedule_with_warmup
)
from peft import get_peft_model, LoraConfig, TaskType
import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Any
import json
import os
from tqdm import tqdm
import logging
from datetime import datetime
import wandb
from PIL import Image

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class PPOConfig:
    """PPO训练配置"""
    model_name: str = "Qwen/Qwen2-VL-7B-Instruct"
    reward_model_path: Optional[str] = None
    use_aesthetic_reward: bool = True
    aesthetic_model_name: str = "cafeai/cafe_aesthetic"
    
    lora_r: int = 64
    lora_alpha: int = 16
    lora_dropout: float = 0.1
    target_modules: List[str] = field(default_factory=lambda: ["q_proj", "v_proj", "k_proj", "o_proj"])
    
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
    
    output_dir: str = "./ppo_qwen_output"
    checkpoint_dir: str = "./checkpoints"
    log_interval: int = 10
    eval_interval: int = 100
    save_interval: int = 500
    
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    mixed_precision: bool = True
    gradient_accumulation_steps: int = 4
    
    use_wandb: bool = False
    wandb_project: str = "ppo-qwen-vl"
    wandb_run_name: Optional[str] = None


class AestheticRewardModel(nn.Module):
    """美学评分模型"""
    
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
    """人工偏好奖励模型"""
    
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
            logger.info(f"Loaded reward model from {checkpoint_path}")
    
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
    """混合奖励模型"""
    
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
        total_rewards = torch.zeros(input_ids.shape[0], device=input_ids.device)
        
        if self.human_reward_model is not None:
            human_rewards = self.human_reward_model(input_ids, attention_mask, pixel_values, image_grid_thw)
            reward_components['human'] = human_rewards
            total_rewards += self.human_weight * human_rewards
        
        if self.aesthetic_model is not None and generated_images is not None:
            aesthetic_rewards = self.aesthetic_model(generated_images)
            reward_components['aesthetic'] = aesthetic_rewards.to(input_ids.device)
            total_rewards += self.aesthetic_weight * aesthetic_rewards.to(input_ids.device)
        
        reward_components['total'] = total_rewards
        return total_rewards, reward_components


class ValueNetwork(nn.Module):
    """价值网络"""
    
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
    """PPO经验回放缓冲区"""
    
    def __init__(self):
        self.states = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.values = []
        self.dones = []
        self.advantages = []
        self.returns = []
    
    def add(self, state, action, log_prob, reward, value, done):
        self.states.append(state)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.rewards.append(reward)
        self.values.append(value)
        self.dones.append(done)
    
    def compute_gae(self, gamma=0.99, gae_lambda=0.95):
        advantages = []
        gae = 0
        for t in reversed(range(len(self.rewards))):
            next_value = 0 if t == len(self.rewards) - 1 else self.values[t + 1]
            delta = self.rewards[t] + gamma * next_value * (1 - self.dones[t]) - self.values[t]
            gae = delta + gamma * gae_lambda * (1 - self.dones[t]) * gae
            advantages.insert(0, gae)
        self.advantages = advantages
        self.returns = [adv + val for adv, val in zip(advantages, self.values)]
    
    def clear(self):
        self.__init__()
    
    def get_batch(self, indices):
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


class PPOTrainer:
    """PPO训练器"""
    
    def __init__(self, config: PPOConfig):
        self.config = config
        self.device = torch.device(config.device)
        
        self.processor = Qwen2VLProcessor.from_pretrained(config.model_name)
        
        logger.info(f"Loading policy model: {config.model_name}")
        self.policy_model = Qwen2VLForConditionalGeneration.from_pretrained(
            config.model_name, torch_dtype=torch.bfloat16, device_map="auto"
        )
        
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=config.lora_r, lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout, target_modules=config.target_modules,
        )
        self.policy_model = get_peft_model(self.policy_model, lora_config)
        self.policy_model.print_trainable_parameters()
        
        self.ref_model = Qwen2VLForConditionalGeneration.from_pretrained(
            config.model_name, torch_dtype=torch.bfloat16, device_map="auto"
        )
        self.ref_model.eval()
        for param in self.ref_model.parameters():
            param.requires_grad = False
        
        logger.info("Initializing value network")
        self.value_model = ValueNetwork(config.model_name)
        
        logger.info("Initializing reward model")
        self.reward_model = HybridRewardModel(config)
        
        self.policy_optimizer = torch.optim.AdamW(
            self.policy_model.parameters(), lr=config.learning_rate,
            betas=(0.9, 0.999), weight_decay=0.01
        )
        self.value_optimizer = torch.optim.AdamW(
            self.value_model.parameters(), lr=config.learning_rate,
            betas=(0.9, 0.999), weight_decay=0.01
        )
        
        self.policy_scheduler = get_cosine_schedule_with_warmup(
            self.policy_optimizer, num_warmup_steps=config.warmup_steps, num_training_steps=10000
        )
        self.value_scheduler = get_cosine_schedule_with_warmup(
            self.value_optimizer, num_warmup_steps=config.warmup_steps, num_training_steps=10000
        )
        
        self.scaler = torch.cuda.amp.GradScaler() if config.mixed_precision else None
        self.memory = PPOMemory()
        self.global_step = 0
        self.epoch = 0
        
        if config.use_wandb:
            wandb.init(project=config.wandb_project, name=config.wandb_run_name, config=config.__dict__)
        
        os.makedirs(config.output_dir, exist_ok=True)
        os.makedirs(config.checkpoint_dir, exist_ok=True)
    
    def generate(self, prompts, images=None):
        if images:
            inputs = self.processor(text=[p['text'] for p in prompts], images=images, return_tensors="pt", padding=True).to(self.device)
        else:
            inputs = self.processor(text=[p['text'] for p in prompts], return_tensors="pt", padding=True).to(self.device)
        
        with torch.no_grad():
            outputs = self.policy_model.generate(
                **inputs, max_new_tokens=self.config.max_new_tokens,
                temperature=self.config.temperature, top_p=self.config.top_p,
                do_sample=True, return_dict_in_generate=True, output_scores=True, use_cache=True
            )
        
        sequences = outputs.sequences
        log_probs = self._compute_log_probs(sequences, inputs)
        decoded_texts = self.processor.batch_decode(sequences, skip_special_tokens=True)
        return sequences, log_probs, decoded_texts
    
    def _compute_log_probs(self, sequences, inputs):
        with torch.no_grad():
            outputs = self.policy_model(input_ids=sequences, attention_mask=torch.ones_like(sequences), return_dict=True)
            logits = outputs.logits
            log_probs = F.log_softmax(logits, dim=-1)
            target_log_probs = log_probs[:, :-1, :].gather(2, sequences[:, 1:].unsqueeze(-1)).squeeze(-1)
            input_length = inputs['input_ids'].shape[1]
            new_token_log_probs = target_log_probs[:, input_length-1:]
            sequence_log_probs = new_token_log_probs.sum(dim=1)
        return sequence_log_probs
    
    def compute_kl_divergence(self, sequences, inputs):
        with torch.no_grad():
            ref_outputs = self.ref_model(input_ids=sequences, attention_mask=torch.ones_like(sequences), return_dict=True)
            ref_logits = ref_outputs.logits
            ref_log_probs = F.log_softmax(ref_logits, dim=-1)
            policy_outputs = self.policy_model(input_ids=sequences, attention_mask=torch.ones_like(sequences), return_dict=True)
            policy_logits = policy_outputs.logits
            policy_log_probs = F.log_softmax(policy_logits, dim=-1)
            input_length = inputs['input_ids'].shape[1]
            kl_div = (policy_log_probs - ref_log_probs)[:, input_length-1:-1, :]
            kl_div = kl_div.exp() - kl_div - 1
            kl_per_sequence = kl_div.sum(dim=[1, 2])
            kl_mean = kl_per_sequence.mean()
        return kl_mean
    
    def ppo_update(self, memory_batch):
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
                    policy_outputs = self.policy_model(input_ids=batch_input_ids, attention_mask=batch_attention_mask, return_dict=True)
                    logits = policy_outputs.logits
                    log_probs = F.log_softmax(logits, dim=-1)
                    
                    new_log_probs = []
                    for i, action in enumerate(mb_actions):
                        action_log_probs = log_probs[i, :-1, :].gather(1, action.unsqueeze(-1)).squeeze(-1).sum()
                        new_log_probs.append(action_log_probs)
                    new_log_probs = torch.stack(new_log_probs)
                    
                    ratio = torch.exp(new_log_probs - mb_old_log_probs)
                    surr1 = ratio * mb_advantages
                    surr2 = torch.clamp(ratio, 1 - self.config.clip_epsilon, 1 + self.config.clip_epsilon) * mb_advantages
                    policy_loss = -torch.min(surr1, surr2).mean()
                    
                    value_pred = self.value_model(input_ids=batch_input_ids, attention_mask=batch_attention_mask)
                    
                    if self.config.value_clip > 0:
                        old_values = torch.tensor([memory_batch['values'][i] for i in mb_indices], device=self.device)
                        value_pred_clipped = old_values + torch.clamp(value_pred - old_values, -self.config.value_clip, self.config.value_clip)
                        value_loss1 = F.mse_loss(value_pred, mb_returns, reduction='none')
                        value_loss2 = F.mse_loss(value_pred_clipped, mb_returns, reduction='none')
                        value_loss = torch.max(value_loss1, value_loss2).mean()
                    else:
                        value_loss = F.mse_loss(value_pred, mb_returns)
                    
                    kl_penalty = self.compute_kl_divergence(batch_input_ids, {'input_ids': batch_input_ids})
                    entropy = -(log_probs * torch.exp(log_probs)).sum(dim=-1).mean()
                    
                    loss = policy_loss + self.config.value_coef * value_loss - self.config.entropy_coef * entropy + self.config.kl_coef * kl_penalty
                
                self.policy_optimizer.zero_grad()
                self.value_optimizer.zero_grad()
                
                if self.scaler:
                    self.scaler.scale(loss).backward()
                    self.scaler.unscale_(self.policy_optimizer)
                    self.scaler.unscale_(self.value_optimizer)
                else:
                    loss.backward()
                
                torch.nn.utils.clip_grad_norm_(self.policy_model.parameters(), self.config.max_grad_norm)
                torch.nn.utils.clip_grad_norm_(self.value_model.parameters(), self.config.max_grad_norm)
                
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
            'total_loss': (total_policy_loss + total_value_loss) / n_updates
        }
    
    def train_step(self, batch):
        prompts = batch['prompts']
        images = batch.get('images', None)
        
        sequences, log_probs, decoded_texts = self.generate(prompts, images)
        
        with torch.no_grad():
            reward_inputs = {'input_ids': sequences, 'attention_mask': torch.ones_like(sequences)}
            rewards, reward_components = self.reward_model.compute_rewards(**reward_inputs, generated_images=None)
        
        with torch.no_grad():
            values = self.value_model(input_ids=sequences, attention_mask=torch.ones_like(sequences))
        
        for i in range(len(prompts)):
            state = {'input_ids': sequences[i], 'attention_mask': torch.ones_like(sequences[i])}
            action = sequences[i, len(prompts[i]['text']):]
            self.memory.add(state, action, log_probs[i], rewards[i].item(), values[i].item(), True)
        
        self.memory.compute_gae(gamma=self.config.gamma, gae_lambda=self.config.gae_lambda)
        
        if len(self.memory) >= self.config.batch_size:
            batch_data = self.memory.get_batch(list(range(len(self.memory))))
            losses = self.ppo_update(batch_data)
            self.memory.clear()
        else:
            losses = {'policy_loss': 0, 'value_loss': 0, 'entropy': 0, 'kl_penalty': 0, 'total_loss': 0}
        
        self.global_step += 1
        
        log_dict = {**losses, 'mean_reward': rewards.mean().item()}
        log_dict.update({f'reward_{k}': v.mean().item() for k, v in reward_components.items() if k != 'total'})
        
        if self.config.use_wandb and self.global_step % self.config.log_interval == 0:
            wandb.log(log_dict, step=self.global_step)
        
        return log_dict
    
    def save_checkpoint(self, epoch, step):
        checkpoint_path = os.path.join(self.config.checkpoint_dir, f"checkpoint_epoch{epoch}_step{step}")
        os.makedirs(checkpoint_path, exist_ok=True)
        self.policy_model.save_pretrained(os.path.join(checkpoint_path, "policy_model"))
        torch.save(self.value_model.state_dict(), os.path.join(checkpoint_path, "value_model.pt"))
        torch.save({
            'policy_optimizer': self.policy_optimizer.state_dict(),
            'value_optimizer': self.value_optimizer.state_dict(),
            'policy_scheduler': self.policy_scheduler.state_dict(),
            'value_scheduler': self.value_scheduler.state_dict(),
            'epoch': epoch, 'step': step, 'global_step': self.global_step
        }, os.path.join(checkpoint_path, "optimizer.pt"))
        logger.info(f"Checkpoint saved to {checkpoint_path}")
    
    def load_checkpoint(self, checkpoint_path):
        self.policy_model = Qwen2VLForConditionalGeneration.from_pretrained(os.path.join(checkpoint_path, "policy_model"))
        self.value_model.load_state_dict(torch.load(os.path.join(checkpoint_path, "value_model.pt")))
        optimizer_state = torch.load(os.path.join(checkpoint_path, "optimizer.pt"))
        self.policy_optimizer.load_state_dict(optimizer_state['policy_optimizer'])
        self.value_optimizer.load_state_dict(optimizer_state['value_optimizer'])
        self.policy_scheduler.load_state_dict(optimizer_state['policy_scheduler'])
        self.value_scheduler.load_state_dict(optimizer_state['value_scheduler'])
        self.global_step = optimizer_state['global_step']
        logger.info(f"Checkpoint loaded from {checkpoint_path}")


class RLHFDataset(Dataset):
    """RLHF训练数据集"""
    
    def __init__(self, data_path, processor):
        self.processor = processor
        with open(data_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        prompt = {'text': item.get('prompt', item.get('instruction', '')), 'image_path': item.get('image_path', None)}
        image = None
        if prompt['image_path'] and os.path.exists(prompt['image_path']):
            image = Image.open(prompt['image_path']).convert('RGB')
        return {'prompts': [prompt], 'images': [image] if image else None}


def collate_fn(batch):
    prompts = []
    images = []
    for item in batch:
        prompts.extend(item['prompts'])
        if item['images']:
            images.extend(item['images'])
    return {'prompts': prompts, 'images': images if images else None}
