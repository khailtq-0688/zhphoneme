"""
Utility functions for pretraining pipeline
"""

import logging
import torch
from pathlib import Path
from typing import Dict, List, Any
import json

logger = logging.getLogger(__name__)


def set_seed(seed: int = 42):
    """Set random seed for reproducibility"""
    import random
    import numpy as np
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    
    logger.info(f"Random seed set to {seed}")


def get_device():
    """Get available device"""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        logger.info("Using CPU")
    
    return device


def count_parameters(model) -> int:
    """Count total trainable parameters"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def save_model(model, path: str):
    """Save model state dict"""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path)
    logger.info(f"Model saved to {path}")


def load_model(model, path: str):
    """Load model state dict"""
    if not Path(path).exists():
        raise FileNotFoundError(f"Model file not found: {path}")
    
    model.load_state_dict(torch.load(path, map_location='cpu'))
    logger.info(f"Model loaded from {path}")
    return model


def save_config(config: Dict[str, Any], path: str):
    """Save configuration as JSON"""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    logger.info(f"Config saved to {path}")


def load_config(path: str) -> Dict[str, Any]:
    """Load configuration from JSON"""
    if not Path(path).exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    
    with open(path, 'r', encoding='utf-8') as f:
        config = json.load(f)
    logger.info(f"Config loaded from {path}")
    return config


def estimate_tokens(text_file: str) -> int:
    """Estimate number of tokens in a text file"""
    count = 0
    with open(text_file, 'r', encoding='utf-8') as f:
        for line in f:
            # Simple heuristic: split by whitespace
            count += len(line.split())
    
    return count


def print_model_config(model_config: Dict[str, Any]):
    """Print model configuration nicely"""
    logger.info("="*50)
    logger.info("Model Configuration:")
    logger.info("="*50)
    for key, value in model_config.items():
        logger.info(f"{key}: {value}")
    logger.info("="*50)


def prepare_input(text: str, tokenizer, max_len: int = 512) -> Dict[str, torch.Tensor]:
    """Prepare input for model inference"""
    tokens = tokenizer.encode(text)
    
    if len(tokens) > max_len - 2:
        tokens = tokens[:max_len - 2]
    
    input_ids = [1] + tokens + [2]  # [BOS] + tokens + [EOS]
    
    if len(input_ids) < max_len:
        input_ids += [3] * (max_len - len(input_ids))  # PAD
    
    return {
        'input_ids': torch.tensor(input_ids, dtype=torch.long).unsqueeze(0),
    }


class MetricsTracker:
    """Track training metrics"""
    
    def __init__(self):
        self.metrics = {}
    
    def update(self, name: str, value: float):
        """Update metric"""
        if name not in self.metrics:
            self.metrics[name] = []
        self.metrics[name].append(value)
    
    def get_average(self, name: str) -> float:
        """Get average value"""
        if name not in self.metrics or len(self.metrics[name]) == 0:
            return 0.0
        return sum(self.metrics[name]) / len(self.metrics[name])
    
    def get_last(self, name: str) -> float:
        """Get last value"""
        if name not in self.metrics or len(self.metrics[name]) == 0:
            return 0.0
        return self.metrics[name][-1]
    
    def reset(self):
        """Reset metrics"""
        self.metrics = {}
    
    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary of averages"""
        return {name: self.get_average(name) for name in self.metrics}


if __name__ == "__main__":
    # Test utilities
    set_seed(42)
    device = get_device()
    print(f"Device: {device}")
