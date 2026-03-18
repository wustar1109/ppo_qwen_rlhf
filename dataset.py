from torch.utils.data import Dataset
from transformers import Qwen2VLProcessor
from PIL import Image
import json
import os
from typing import List, Dict, Any


class RLHFDataset(Dataset):
    """RLHF训练数据集"""
    
    def __init__(self, data_path: str, processor: Qwen2VLProcessor):
        self.processor = processor
        
        with open(data_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx) -> Dict[str, Any]:
        item = self.data[idx]
        
        prompt = {
            'text': item.get('prompt', item.get('instruction', '')),
            'image_path': item.get('image_path', None)
        }
        
        image = None
        if prompt['image_path'] and os.path.exists(prompt['image_path']):
            image = Image.open(prompt['image_path']).convert('RGB')
        
        return {
            'prompts': [prompt],
            'images': [image] if image else None
        }


def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """自定义collate函数"""
    prompts = []
    images = []
    
    for item in batch:
        prompts.extend(item['prompts'])
        if item['images']:
            images.extend(item['images'])
    
    return {
        'prompts': prompts,
        'images': images if images else None
    }
