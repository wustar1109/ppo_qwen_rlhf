"""
自适应训练控制器
根据训练进展自动调整超参数
"""

import numpy as np
from typing import Dict, List, Optional, Callable
from dataclasses import dataclass, field
import logging

logger = logging.getLogger(__name__)


@dataclass
class AdaptiveConfig:
    """自适应训练配置"""
    # 奖励阈值
    reward_threshold: float = 0.8  # 达到此奖励后降低探索
    early_stop_reward: float = 0.95  # 早停奖励阈值
    early_stop_patience: int = 5  # 早停耐心值
    
    # KL自适应
    kl_target: float = 0.02  # 目标KL散度
    kl_adapt_rate: float = 0.1  # KL调整速率
    kl_min: float = 0.01  # 最小KL系数
    kl_max: float = 0.5  # 最大KL系数
    
    # 学习率自适应
    use_lr_schedule: bool = True
    lr_decay_factor: float = 0.95  # 奖励不提升时的学习率衰减
    lr_min: float = 1e-7  # 最小学习率
    lr_patience: int = 3  # 学习率调整耐心值
    
    # Clip自适应
    clip_min: float = 0.1
    clip_max: float = 0.3
    clip_adapt_threshold: float = 0.1  # 基于策略梯度方差调整
    
    # 奖励权重自适应
    adjust_reward_weights: bool = True
    reward_weight_patience: int = 5  # 调整奖励权重的耐心值
    
    # 熵正则化
    entropy_min: float = 0.001  # 最小熵（不继续探索）
    entropy_target: float = 0.05  # 目标熵
    entropy_boost: float = 1.5  # 熵不足时的boost系数


class AdaptiveController:
    """自适应训练控制器"""
    
    def __init__(self, config: AdaptiveConfig, initial_params: Dict):
        self.config = config
        self.params = initial_params
        
        # 历史记录
        self.reward_history: List[float] = []
        self.kl_history: List[float] = []
        self.entropy_history: List[float] = []
        self.loss_history: List[float] = []
        
        # 统计
        self.steps_without_improvement = 0
        self.best_reward = float('-inf')
        self.no_improvement_count = 0
        
        # 学习率调度
        self.current_lr = initial_params.get('learning_rate', 1e-5)
        
    def update(self, metrics: Dict[str, float]) -> Dict[str, float]:
        """
        更新超参数
        Returns: 调整后的参数字典
        """
        # 记录历史
        self.reward_history.append(metrics.get('reward', 0))
        self.kl_history.append(metrics.get('kl', 0))
        self.entropy_history.append(metrics.get('entropy', 0))
        self.loss_history.append(metrics.get('loss', 0))
        
        # 限制历史长度
        max_history = 100
        if len(self.reward_history) > max_history:
            self.reward_history = self.reward_history[-max_history:]
            self.kl_history = self.kl_history[-max_history:]
            self.entropy_history = self.entropy_history[-max_history:]
            self.loss_history = self.loss_history[-max_history:]
        
        # 自适应调整
        adjusted_params = self.params.copy()
        
        # 1. KL散度自适应
        adjusted_params['kl_coef'] = self._adapt_kl()
        
        # 2. 学习率自适应
        adjusted_params['learning_rate'] = self._adapt_learning_rate()
        
        # 3. Clip自适应
        adjusted_params['clip_epsilon'] = self._adapt_clip()
        
        # 4. 熵正则化自适应
        adjusted_params['entropy_coef'] = self._adapt_entropy()
        
        # 5. 早停检查
        should_stop, reason = self._check_early_stop()
        
        if should_stop:
            logger.info(f"早停触发: {reason}")
            adjusted_params['should_stop'] = True
            adjusted_params['stop_reason'] = reason
        
        # 更新历史最佳
        current_reward = metrics.get('reward', 0)
        if current_reward > self.best_reward:
            self.best_reward = current_reward
            self.no_improvement_count = 0
        else:
            self.no_improvement_count += 1
        
        self.params = adjusted_params
        return adjusted_params
    
    def _adapt_kl(self) -> float:
        """根据KL历史自适应调整KL系数"""
        if len(self.kl_history) < 10:
            return self.params.get('kl_coef', 0.1)
        
        recent_kl = np.mean(self.kl_history[-10:])
        kl_target = self.config.kl_target
        
        # 计算调整
        current_kl = self.params.get('kl_coef', 0.1)
        
        if recent_kl > kl_target * 1.5:
            # KL太大，增加惩罚
            new_kl = current_kl * (1 + self.config.kl_adapt_rate)
        elif recent_kl < kl_target * 0.5:
            # KL太小，减少惩罚
            new_kl = current_kl * (1 - self.config.kl_adapt_rate)
        else:
            new_kl = current_kl
        
        # 限制范围
        new_kl = np.clip(new_kl, self.config.kl_min, self.config.kl_max)
        
        return new_kl
    
    def _adapt_learning_rate(self) -> float:
        """根据奖励进展自适应调整学习率"""
        if len(self.reward_history) < self.config.lr_patience:
            return self.current_lr
        
        # 检查最近几次是否有提升
        recent_rewards = self.reward_history[-self.config.lr_patience:]
        
        # 计算滑动平均
        if len(recent_rewards) >= 2:
            improvement = recent_rewards[-1] - recent_rewards[0]
            
            if improvement <= 0:
                # 没有提升，降低学习率
                self.current_lr *= self.config.lr_decay_factor
            elif improvement > 0.05:
                # 提升显著，可以适当提高学习率
                self.current_lr *= 1.1
        
        # 限制范围
        self.current_lr = max(self.current_lr, self.config.lr_min)
        
        return self.current_lr
    
    def _adapt_clip(self) -> float:
        """根据策略梯度方差自适应调整clip"""
        if len(self.loss_history) < 10:
            return self.params.get('clip_epsilon', 0.2)
        
        # 计算loss的变化率（作为策略稳定性的近似）
        recent_losses = self.loss_history[-10:]
        loss_std = np.std(recent_losses)
        
        current_clip = self.params.get('clip_epsilon', 0.2)
        
        # 如果loss变化剧烈，降低clip以稳定训练
        if loss_std > self.config.clip_adapt_threshold:
            new_clip = current_clip * 0.9
        elif loss_std < self.config.clip_adapt_threshold * 0.3:
            # loss很稳定，可以增加clip尝试更大更新
            new_clip = current_clip * 1.1
        else:
            new_clip = current_clip
        
        return np.clip(new_clip, self.config.clip_min, self.config.clip_max)
    
    def _adapt_entropy(self) -> float:
        """根据熵自适应调整探索"""
        if len(self.entropy_history) < 5:
            return self.params.get('entropy_coef', 0.01)
        
        recent_entropy = np.mean(self.entropy_history[-5:])
        current_entropy_coef = self.params.get('entropy_coef', 0.01)
        
        if recent_entropy < self.config.entropy_min:
            # 熵太低，增加探索
            new_entropy_coef = current_entropy_coef * self.config.entropy_boost
        elif recent_entropy > self.config.entropy_target * 1.5:
            # 熵太高，减少探索
            new_entropy_coef = current_entropy_coef / self.config.entropy_boost
        else:
            new_entropy_coef = current_entropy_coef
        
        return np.clip(new_entropy_coef, 0.001, 0.1)
    
    def _check_early_stop(self) -> tuple:
        """检查是否应该早停"""
        # 检查1: 达到目标奖励
        if self.best_reward >= self.config.early_stop_reward:
            return True, f"达到目标奖励 {self.best_reward:.4f} >= {self.config.early_stop_reward}"
        
        # 检查2: 长时间无提升
        if self.no_improvement_count >= self.config.early_stop_patience:
            recent = self.reward_history[-self.config.early_stop_patience:]
            if len(set([round(r, 3) for r in recent])) == 1:
                return True, f"连续{self.no_improvement_count}次无提升"
        
        # 检查3: KL散度过大（训练不稳定）
        if len(self.kl_history) >= 5:
            recent_kl = np.mean(self.kl_history[-5:])
            if recent_kl > 1.0:  # KL超过1说明分布偏离太大
                return True, f"KL散度过大 {recent_kl:.4f}"
        
        # 检查4: 熵崩溃
        if len(self.entropy_history) >= 10:
            recent_entropy = np.mean(self.entropy_history[-10:])
            if recent_entropy < 0.0001:
                return True, "熵崩溃，模型已收敛到确定性策略"
        
        return False, ""
    
    def get_status(self) -> Dict:
        """获取当前状态"""
        return {
            'best_reward': self.best_reward,
            'current_lr': self.current_lr,
            'kl_coef': self.params.get('kl_coef'),
            'clip_epsilon': self.params.get('clip_epsilon'),
            'entropy_coef': self.params.get('entropy_coef'),
            'steps_without_improvement': self.no_improvement_count,
            'recent_avg_reward': np.mean(self.reward_history[-10:]) if self.reward_history else 0,
            'recent_avg_kl': np.mean(self.kl_history[-10:]) if self.kl_history else 0,
        }


class RewardWeightOptimizer:
    """奖励权重优化器 - 自动调整人工偏好和美学评分的权重"""
    
    def __init__(self, human_weight: float = 0.7, aesthetic_weight: float = 0.3):
        self.human_weight = human_weight
        self.aesthetic_weight = aesthetic_weight
        self.history = []
        
    def update(self, reward_components: Dict[str, float]) -> Dict[str, float]:
        """根据各奖励组件的表现调整权重"""
        self.history.append(reward_components)
        
        if len(self.history) < 10:
            return {'human_weight': self.human_weight, 'aesthetic_weight': self.aesthetic_weight}
        
        # 分析最近的奖励
        recent = self.history[-10:]
        
        human_rewards = [r.get('human', 0) for r in recent]
        aesthetic_rewards = [r.get('aesthetic', 0) for r in recent]
        
        # 计算方差 - 方差大的组件说明不太稳定，降低其权重
        human_std = np.std(human_rewards) if len(human_rewards) > 1 else 1
        aesthetic_std = np.std(aesthetic_rewards) if len(aesthetic_rewards) > 1 else 1
        
        # 基于稳定性调整权重
        total_std = human_std + aesthetic_std
        if total_std > 0:
            # 更稳定的组件获得更高权重
            new_human_weight = (1 / (human_std + 1e-6)) / (1 / (human_std + 1e-6) + 1 / (aesthetic_std + 1e-6))
            new_aesthetic_weight = 1 - new_human_weight
            
            # 平滑过渡
            alpha = 0.1
            self.human_weight = alpha * new_human_weight + (1 - alpha) * self.human_weight
            self.aesthetic_weight = alpha * new_aesthetic_weight + (1 - alpha) * self.aesthetic_weight
        
        return {'human_weight': self.human_weight, 'aesthetic_weight': self.aesthetic_weight}