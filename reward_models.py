"""
Image-quality reward models (enhanced).
"""

from dataclasses import dataclass
from typing import List, Tuple, Dict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image


def _cfg_get(config, name, default=None):
    if hasattr(config, "get"):
        return config.get(name, default)
    return getattr(config, name, default)


class DifferentiableImageRewardModel(nn.Module):
    """Differentiable reward on torch image tensors.

    Expects images as torch tensors in [0, 1] with shape [B, C, H, W].
    """

    def __init__(self, edge_weight: float = 0.6, tv_weight: float = 0.4, color_weight: float = 0.05):
        super().__init__()
        self.edge_weight = edge_weight
        self.tv_weight = tv_weight
        self.color_weight = color_weight

        sobel_x = torch.tensor([[[[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]]], dtype=torch.float32) / 8.0
        sobel_y = torch.tensor([[[[-1, -2, -1], [0, 0, 0], [1, 2, 1]]]], dtype=torch.float32) / 8.0
        self.register_buffer("sobel_x", sobel_x)
        self.register_buffer("sobel_y", sobel_y)

    def compute_rewards(self, image_tensors: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        if image_tensors is None:
            rewards = torch.zeros(1)
            return rewards, {"total": rewards}

        if image_tensors.dim() == 3:
            image_tensors = image_tensors.unsqueeze(0)

        image_tensors = image_tensors.clamp(0, 1)
        if image_tensors.size(1) == 1:
            gray = image_tensors
        else:
            gray = (
                0.299 * image_tensors[:, 0:1, :, :]
                + 0.587 * image_tensors[:, 1:2, :, :]
                + 0.114 * image_tensors[:, 2:3, :, :]
            )

        sobel_x = self.sobel_x.to(device=gray.device, dtype=gray.dtype)
        sobel_y = self.sobel_y.to(device=gray.device, dtype=gray.dtype)
        grad_x = F.conv2d(gray, sobel_x, padding=1)
        grad_y = F.conv2d(gray, sobel_y, padding=1)
        edge_strength = torch.sqrt(grad_x ** 2 + grad_y ** 2 + 1e-6).mean(dim=[1, 2, 3])

        tv_h = (gray[:, :, 1:, :] - gray[:, :, :-1, :]).abs().mean(dim=[1, 2, 3])
        tv_w = (gray[:, :, :, 1:] - gray[:, :, :, :-1]).abs().mean(dim=[1, 2, 3])
        total_variation = tv_h + tv_w

        if image_tensors.size(1) >= 3:
            color_var = image_tensors.var(dim=[2, 3]).mean(dim=1)
        else:
            color_var = torch.zeros_like(edge_strength)

        rewards = (
            self.edge_weight * edge_strength
            - self.tv_weight * total_variation
            + self.color_weight * color_var
        )

        components = {
            "edge": edge_strength,
            "tv": total_variation,
            "color": color_var,
            "total": rewards,
        }

        return rewards, components

    def forward(self, image_tensors: torch.Tensor) -> torch.Tensor:
        rewards, _ = self.compute_rewards(image_tensors)
        return rewards


class GraySmoothnessRewardModel(nn.Module):
    def __init__(self):
        super().__init__()

    @torch.no_grad()
    def forward(self, images: List[Image.Image]) -> torch.Tensor:
        rewards = []
        for img in images:
            if isinstance(img, Image.Image):
                gray = img.convert('L')
                arr = np.array(gray, dtype=np.float32) / 255.0
            else:
                arr = img

            tensor = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0)

            grad_x = F.conv2d(tensor, torch.tensor([[[[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]]], dtype=torch.float32) / 8.0)
            grad_y = F.conv2d(tensor, torch.tensor([[[[-1, -2, -1], [0, 0, 0], [1, 2, 1]]]], dtype=torch.float32) / 8.0)
            grad_magnitude = torch.sqrt(grad_x ** 2 + grad_y ** 2)

            grad_diff_x = F.conv2d(grad_magnitude, torch.tensor([[[[-1, 0, 1]]]], dtype=torch.float32) / 2.0)
            grad_diff_y = F.conv2d(grad_magnitude, torch.tensor([[[[-1], [0], [1]]]], dtype=torch.float32) / 2.0)

            smoothness = 1.0 / (1.0 + grad_diff_x.abs().mean() + grad_diff_y.abs().mean())
            rewards.append(smoothness.item())

        return torch.tensor(rewards, device='cpu')


class LineSmoothnessRewardModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.sobel_x = torch.tensor([[[[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]]], dtype=torch.float32) / 8.0
        self.sobel_y = torch.tensor([[[[-1, -2, -1], [0, 0, 0], [1, 2, 1]]]], dtype=torch.float32) / 8.0
        self.laplacian = torch.tensor([[[[0, 1, 0], [1, -4, 1], [0, 1, 0]]]], dtype=torch.float32)

    @torch.no_grad()
    def forward(self, images: List[Image.Image]) -> torch.Tensor:
        rewards = []
        for img in images:
            if isinstance(img, Image.Image):
                gray = img.convert('L')
                arr = np.array(gray, dtype=np.float32) / 255.0
            else:
                arr = img

            tensor = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0)

            edges_x = F.conv2d(tensor, self.sobel_x)
            edges_y = F.conv2d(tensor, self.sobel_y)
            edges = torch.sqrt(edges_x ** 2 + edges_y ** 2 + 1e-8)

            edge_threshold = edges.mean() * 0.5
            edge_binary = (edges > edge_threshold).float()
            edge_density = edge_binary.mean()

            laplacian = F.conv2d(tensor, self.laplacian)
            corner_responses = laplacian.abs()
            corner_density = (corner_responses > corner_responses.mean()).float().mean()

            edge_profile = edges.squeeze().numpy()
            profile_std = np.std(edge_profile[edge_profile > edge_threshold.item() * 0.5])

            continuity_score = 1.0 - min(corner_density.item() * 2, 1.0)
            thickness_score = 1.0 / (1.0 + profile_std * 10)
            density_score = min(edge_density.item(), 1.0)

            smoothness = 0.4 * continuity_score + 0.3 * thickness_score + 0.3 * density_score
            rewards.append(smoothness)

        return torch.tensor(rewards, device='cpu')


class NoiseArtifactRewardModel(nn.Module):
    def __init__(self):
        super().__init__()

    @torch.no_grad()
    def forward(self, images: List[Image.Image]) -> torch.Tensor:
        rewards = []
        for img in images:
            if isinstance(img, Image.Image):
                gray = img.convert('L')
                arr = np.array(gray, dtype=np.float32) / 255.0
            else:
                arr = img

            tensor = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0)

            high_pass = torch.tensor([[[[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]]]], dtype=torch.float32) / 8.0
            high_freq = F.conv2d(tensor, high_pass)
            high_energy = (high_freq ** 2).mean()

            local_mean = F.avg_pool2d(tensor, kernel_size=3, stride=1, padding=1)
            local_var = F.avg_pool2d((tensor - local_mean) ** 2, kernel_size=3, stride=1, padding=1)

            var_threshold = local_var.mean() * 2
            artifact_pixels = (local_var > var_threshold).float().mean()

            neighbor_diff_x = F.conv2d(tensor, torch.tensor([[[[0, 0, 0], [-1, 1, 0], [0, 0, 0]]]], dtype=torch.float32))
            neighbor_diff_y = F.conv2d(tensor, torch.tensor([[[[0, -1, 0], [0, 1, 0], [0, 0, 0]]]], dtype=torch.float32))
            neighbor_diff = torch.sqrt(neighbor_diff_x ** 2 + neighbor_diff_y ** 2 + 1e-8)

            isolated_points = (neighbor_diff > neighbor_diff.quantile(0.95)).float().mean()

            noise_score = 1.0 / (1.0 + high_energy.item() * 100)
            artifact_score = 1.0 - min(artifact_pixels.item(), 1.0)
            isolated_score = 1.0 - min(isolated_points.item() * 5, 1.0)

            cleanliness = 0.3 * noise_score + 0.4 * artifact_score + 0.3 * isolated_score
            rewards.append(cleanliness)

        return torch.tensor(rewards, device='cpu')


class EnhancedRewardModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.gray_smoothness_model = GraySmoothnessRewardModel()
        self.line_smoothness_model = LineSmoothnessRewardModel()
        self.noise_artifact_model = NoiseArtifactRewardModel()

        self.weights = {
            'gray_smoothness': _cfg_get(config, 'gray_smoothness_weight', 0.15),
            'line_smoothness': _cfg_get(config, 'line_smoothness_weight', 0.15),
            'noise_artifact': _cfg_get(config, 'noise_artifact_weight', 0.15),
            'human': _cfg_get(config, 'human_weight', 0.35),
            'aesthetic': _cfg_get(config, 'aesthetic_weight', 0.20),
        }

        if _cfg_get(config, 'reward_model_path'):
            from models import HumanPreferenceRewardModel
            self.human_reward_model = HumanPreferenceRewardModel(
                _cfg_get(config, 'model_name'),
                _cfg_get(config, 'reward_model_path'),
            )
        else:
            self.human_reward_model = None

        if _cfg_get(config, 'use_aesthetic_reward', True):
            from models import AestheticRewardModel
            self.aesthetic_model = AestheticRewardModel(
                _cfg_get(config, 'aesthetic_model_name', 'cafeai/cafe_aesthetic')
            )
        else:
            self.aesthetic_model = None

    @torch.no_grad()
    def compute_rewards(
        self,
        input_ids: torch.Tensor = None,
        attention_mask: torch.Tensor = None,
        pixel_values: torch.Tensor = None,
        generated_images: List[Image.Image] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        reward_components = {}
        total_rewards = torch.zeros(len(generated_images) if generated_images else 1)

        if generated_images:
            n_samples = len(generated_images)
        else:
            n_samples = input_ids.shape[0] if input_ids is not None else 1
            generated_images = [None] * n_samples

        device = 'cpu'
        if input_ids is not None:
            device = input_ids.device

        if generated_images and any(img is not None for img in generated_images):
            valid_images = [img if img is not None else Image.new('L', (256, 256), 128) for img in generated_images]
            gray_rewards = self.gray_smoothness_model(valid_images)
            reward_components['gray_smoothness'] = gray_rewards.to(device)
            total_rewards += self.weights['gray_smoothness'] * gray_rewards.to(device)

        if generated_images and any(img is not None for img in generated_images):
            valid_images = [img if img is not None else Image.new('L', (256, 256), 128) for img in generated_images]
            line_rewards = self.line_smoothness_model(valid_images)
            reward_components['line_smoothness'] = line_rewards.to(device)
            total_rewards += self.weights['line_smoothness'] * line_rewards.to(device)

        if generated_images and any(img is not None for img in generated_images):
            valid_images = [img if img is not None else Image.new('L', (256, 256), 128) for img in generated_images]
            noise_rewards = self.noise_artifact_model(valid_images)
            reward_components['noise_artifact'] = noise_rewards.to(device)
            total_rewards += self.weights['noise_artifact'] * noise_rewards.to(device)

        if self.human_reward_model is not None and input_ids is not None:
            human_rewards = self.human_reward_model(
                input_ids, attention_mask, pixel_values
            )
            reward_components['human'] = human_rewards
            total_rewards += self.weights['human'] * human_rewards

        if self.aesthetic_model is not None and generated_images:
            valid_images = [img for img in generated_images if img is not None]
            if valid_images:
                aesthetic_rewards = self.aesthetic_model(valid_images)
                if len(aesthetic_rewards) < n_samples:
                    aesthetic_rewards = torch.cat([
                        aesthetic_rewards,
                        torch.zeros(n_samples - len(aesthetic_rewards), device=aesthetic_rewards.device)
                    ])
                reward_components['aesthetic'] = aesthetic_rewards.to(device)
                total_rewards += self.weights['aesthetic'] * aesthetic_rewards.to(device)

        reward_components['total'] = total_rewards
        return total_rewards, reward_components

    def set_weights(self, weights: Dict[str, float]):
        self.weights.update(weights)

    def get_weights(self) -> Dict[str, float]:
        return self.weights.copy()


@dataclass
class EnhancedRewardConfig:
    model_name: str = "Qwen/Qwen2-VL-7B-Instruct"
    reward_model_path: str = None
    use_aesthetic_reward: bool = True
    aesthetic_model_name: str = "cafeai/cafe_aesthetic"

    gray_smoothness_weight: float = 0.15
    line_smoothness_weight: float = 0.15
    noise_artifact_weight: float = 0.15

    human_weight: float = 0.35
    aesthetic_weight: float = 0.20

    def to_dict(self):
        return {
            'gray_smoothness_weight': self.gray_smoothness_weight,
            'line_smoothness_weight': self.line_smoothness_weight,
            'noise_artifact_weight': self.noise_artifact_weight,
            'human_weight': self.human_weight,
            'aesthetic_weight': self.aesthetic_weight,
            'model_name': self.model_name,
            'reward_model_path': self.reward_model_path,
            'use_aesthetic_reward': self.use_aesthetic_reward,
            'aesthetic_model_name': self.aesthetic_model_name,
        }

