# PPO RLHF for Qwen Vision-Language Model

基于PPO算法的Qwen多模态模型强化学习训练框架。

## 架构说明

```
Prompt → Image Model (Actor) → Generated Response
                                    ↓
Reward Model ← Human Preference + Aesthetic Score
                                    ↓
PPO Trainer → Update Image Model (with KL penalty)
```

## 核心组件

1. **Image Model (Actor)** - 策略模型，使用LoRA微调
2. **Value Network (Critic)** - 价值网络，估计状态价值
3. **Reward Model** - 混合奖励模型：
   - 人工偏好奖励 (Human Preference)
   - 美学评分奖励 (Aesthetic Score)
4. **PPO Trainer** - PPO训练器，包含GAE优势估计

## 安装依赖

```bash
pip install torch transformers peft accelerate wandb pillow tqdm
```

## 数据格式

```json
[
  {
    "prompt": "描述这张图片",
    "image_path": "path/to/image.jpg"
  }
]
```

## 使用方法

```python
from ppo_trainer import PPOTrainer, PPOConfig

config = PPOConfig(
    model_name="Qwen/Qwen2-VL-7B-Instruct",
    reward_model_path="path/to/your/reward_model",
    use_aesthetic_reward=True,
    batch_size=4,
    learning_rate=1e-5
)

trainer = PPOTrainer(config)

# 训练循环
for epoch in range(config.num_epochs):
    for batch in dataloader:
        log_dict = trainer.train_step(batch)
        print(f"Reward: {log_dict['mean_reward']:.4f}")
```

## 配置参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `model_name` | Qwen2-VL-7B-Instruct | 基础模型 |
| `lora_r` | 64 | LoRA秩 |
| `ppo_epochs` | 4 | PPO更新轮数 |
| `clip_epsilon` | 0.2 | PPO裁剪参数 |
| `kl_coef` | 0.1 | KL散度惩罚系数 |
| `gamma` | 0.99 | GAE折扣因子 |
| `gae_lambda` | 0.95 | GAE lambda |

## 文件结构

- `ppo_trainer.py` - 主训练代码
- `train.py` - 训练脚本入口
- `data/train.json` - 训练数据
- `checkpoints/` - 模型检查点
- `logs/` - 训练日志
