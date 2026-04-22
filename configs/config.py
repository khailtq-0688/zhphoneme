"""
Configuration for ViWordFormer Pretraining
Follows ViWordFormer builder pattern
"""

import json
import yaml
from typing import Dict, Any
from pathlib import Path
from dataclasses import dataclass, asdict


@dataclass
class TokenizerConfig:
    """Tokenizer Configuration"""
    vocab_size: int = 30000
    model_type: str = 'unigram'
    character_coverage: float = 0.9995
    normalization_rule: str = 'identity'
    unk_piece: str = '<unk>'
    bos_piece: str = '<s>'
    eos_piece: str = '</s>'
    pad_piece: str = '<pad>'
    mask_piece: str = '<mask>'


@dataclass
class ModelConfig:
    """Model Architecture Configuration"""
    # Vocabulary
    vocab_size: int = 30000
    pad_idx: int = 3
    
    # Architecture
    d_model: int = 768
    d_ff: int = 3072
    nlayers: int = 12
    head: int = 12
    d_q: int = 64
    d_kv: int = 64
    
    # Training
    dropout: float = 0.1
    label_smoothing: float = 0.1
    max_seq_len: int = 512
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]):
        return cls(**config_dict)


@dataclass
class DataConfig:
    """Data Configuration"""
    # Dataset
    type: str = 'SubsetDataset'
    corpus_dir: str = "."
    lines_per_file: int = 1000
    
    # Processing
    max_seq_len: int = 512
    mlm_probability: float = 0.15  # MLM masking probability
    preprocessing_workers: int = 8
    chunk_size: int = 10000


@dataclass
class TrainingConfig:
    """Training Configuration"""
    # Training parameters
    batch_size: int = 64
    gradient_accumulation_steps: int = 1
    num_epochs: int = 5
    learning_rate: float = 5e-5
    weight_decay: float = 0.01
    # warmup_steps will be calculated dynamically as 5% of total_steps
    max_steps: int = -1  # -1 means no limit
    
    # Optimization
    optimizer: str = 'adamw'
    scheduler: str = 'lambda'
    gradient_clip_norm: float = 1.0
    
    # AdamW specific
    betas: tuple = (0.9, 0.999)
    eps: float = 1e-6
    
    # Checkpointing and logging
    save_steps: int = 1000
    eval_steps: int = 2000
    log_steps: int = 100
    save_total_limit: int = 3
    
    # Hardware
    device: str = 'cuda'
    fp16: bool = False
    num_workers: int = 4
    
    # Paths
    output_dir: str = "./outputs"
    checkpoint_dir: str = "./checkpoints"
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]):
        return cls(**config_dict)


class PretrainingConfig:
    """Complete pretraining configuration"""
    
    def __init__(self):
        self.tokenizer = TokenizerConfig()
        self.model = ModelConfig()
        self.data = DataConfig()
        self.training = TrainingConfig()
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'tokenizer': asdict(self.tokenizer),
            'model': asdict(self.model),
            'data': asdict(self.data),
            'training': asdict(self.training),
        }
    
    def save(self, path: str):
        """Save configuration to JSON file"""
        config_dict = self.to_dict()
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(config_dict, f, indent=2)
    
    @classmethod
    def load(cls, path: str):
        """Load configuration from JSON file"""
        with open(path, 'r', encoding='utf-8') as f:
            config_dict = json.load(f)
        
        config = cls()
        
        if 'tokenizer' in config_dict:
            config.tokenizer = TokenizerConfig(**config_dict['tokenizer'])
        if 'model' in config_dict:
            config.model = ModelConfig(**config_dict['model'])
        if 'data' in config_dict:
            config.data = DataConfig(**config_dict['data'])
        if 'training' in config_dict:
            config.training = TrainingConfig(**config_dict['training'])
        
        return config
    
    def update_from_dict(self, config_dict: Dict[str, Any]):
        """Update configuration from dictionary"""
        if 'tokenizer' in config_dict:
            self.tokenizer = TokenizerConfig(**config_dict['tokenizer'])
        if 'model' in config_dict:
            self.model = ModelConfig(**config_dict['model'])
        if 'data' in config_dict:
            self.data = DataConfig(**config_dict['data'])
        if 'training' in config_dict:
            self.training = TrainingConfig(**config_dict['training'])


def get_default_config() -> PretrainingConfig:
    """Get default configuration"""
    return PretrainingConfig()


def load_config_from_yaml(yaml_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file"""
    with open(yaml_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config if config is not None else {}


class DotDict(dict):
    """Dictionary that supports dot notation access"""
    def __getattr__(self, name):
        if name in self:
            return self[name]
        raise AttributeError(f"No attribute '{name}'")
    
    def __setattr__(self, name, value):
        self[name] = value
    
    def __delattr__(self, name):
        if name in self:
            del self[name]
        else:
            raise AttributeError(f"No attribute '{name}'")


def dict_to_dotdict(d: dict) -> DotDict:
    """Convert nested dict to DotDict for dot notation access"""
    if isinstance(d, dict):
        return DotDict({k: dict_to_dotdict(v) for k, v in d.items()})
    elif isinstance(d, list):
        return [dict_to_dotdict(item) for item in d]
    else:
        return d


if __name__ == "__main__":
    # Create and save default configuration
    config = get_default_config()
    
    # Customize as needed
    config.model.vocab_size = 30000
    config.model.d_model = 768
    config.training.batch_size = 64
    config.training.num_epochs = 5
    
    # Save configuration
    config.save("./config.json")
    print("Configuration saved to ./config.json")
