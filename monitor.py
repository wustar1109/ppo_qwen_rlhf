"""
训练监控脚本
实时监控训练进度，并通过 QQ 发送通知
"""

import os
import time
import json
import subprocess
from datetime import datetime
from pathlib import Path


class TrainingMonitor:
    """训练过程监控器"""
    
    def __init__(self, log_file: str, notify_interval: int = 300):
        self.log_file = log_file
        self.notify_interval = notify_interval  # 通知间隔（秒）
        self.last_notify_time = time.time()
        self.last_position = 0
        self.best_reward = float('-inf')
        
    def get_latest_stats(self) -> dict:
        """从日志中获取最新统计"""
        if not os.path.exists(self.log_file):
            return {}
        
        with open(self.log_file, 'r') as f:
            # 跳到上次位置
            f.seek(self.last_position)
            new_lines = f.readlines()
            self.last_position = f.tell()
        
        stats = {
            'epoch': None,
            'step': None,
            'reward': None,
            'policy_loss': None,
            'value_loss': None,
            'kl': None
        }
        
        for line in new_lines:
            # 解析训练日志
            if 'Reward:' in line:
                try:
                    parts = line.split('|')
                    for part in parts:
                        if 'Reward:' in part:
                            stats['reward'] = float(part.split(':')[1].strip())
                        if 'Step' in part and '/' in part:
                            stats['step'] = part.split('Step')[1].split('/')[0].strip()
                        if 'Epoch' in part:
                            stats['epoch'] = part.split('Epoch')[1].split('/')[0].strip()
                except:
                    pass
            
            # 解析完成日志
            if 'Epoch' in line and 'completed' in line or '完成' in line:
                try:
                    # 提取平均奖励
                    if 'Avg Reward:' in line:
                        reward = float(line.split('Avg Reward:')[1].strip().split()[0])
                        stats['reward'] = reward
                        if reward > self.best_reward:
                            self.best_reward = reward
                            stats['new_best'] = True
                except:
                    pass
        
        return stats
    
    def should_notify(self) -> bool:
        """判断是否应该发送通知"""
        return time.time() - self.last_notify_time >= self.notify_interval
    
    def send_notification(self, stats: dict, message: str = None):
        """发送训练通知"""
        self.last_notify_time = time.time()
        
        # 构建通知消息
        if message is None:
            msg = "📊 训练进度报告\n"
            msg += f"时间: {datetime.now().strftime('%H:%M:%S')}\n"
            
            if stats.get('epoch'):
                msg += f"Epoch: {stats['epoch']}\n"
            if stats.get('step'):
                msg += f"Step: {stats['step']}\n"
            if stats.get('reward'):
                msg += f"当前奖励: {stats['reward']:.4f}\n"
            
            if stats.get('new_best'):
                msg += f"🎉 新最佳奖励: {self.best_reward:.4f}!"
            
            if self.best_reward != float('-inf'):
                msg += f"\n最佳奖励: {self.best_reward:.4f}"
        else:
            msg = message
        
        print(msg)
        return msg
    
    def monitor(self, check_interval: int = 30):
        """持续监控训练过程"""
        print(f"开始监控训练日志: {self.log_file}")
        print(f"检查间隔: {check_interval}秒")
        
        while True:
            # 检查训练是否结束
            if not os.path.exists(self.log_file):
                time.sleep(check_interval)
                continue
            
            stats = self.get_latest_stats()
            
            # 检查是否完成
            if stats.get('reward') and 'completed' in str(stats):
                self.send_notification(stats, "🎉 训练已完成!")
                break
            
            # 定期发送进度通知
            if self.should_notify() and stats.get('reward'):
                self.send_notification(stats)
            
            time.sleep(check_interval)


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="训练监控器")
    parser.add_argument("--log_file", type=str, 
                        default="/root/.openclaw/workspace/ppo_qwen_rlhf/train.log",
                        help="训练日志文件路径")
    parser.add_argument("--interval", type=int, default=30,
                        help="检查间隔（秒）")
    parser.add_argument("--notify_interval", type=int, default=600,
                        help="通知间隔（秒），默认10分钟")
    
    args = parser.parse_args()
    
    monitor = TrainingMonitor(args.log_file, args.notify_interval)
    monitor.monitor(args.interval)


if __name__ == "__main__":
    main()