"""
Model builder for pretraining
Instantiates models from configuration
"""

import torch
from .registry import META_ARCHITECTURE
from configs.config import ModelConfig


def build_model(config: ModelConfig, vocab_size: int):
    """
    Build a model from configuration
    
    Args:
        config: ModelConfig instance
        vocab_size: Vocabulary size from tokenizer
        
    Returns:
        Model instance on specified device
    """
    # Get model class from registry
    model_class = META_ARCHITECTURE.get(config.__class__.__name__ if hasattr(config, 'architecture') else 'ViWordFormer')
    
    # Create model
    model = model_class(
        vocab_size=vocab_size,
        d_model=config.model.d_model,          
        nlayers=config.model.nlayers,           
        head=config.model.head,                 
        d_q=config.model.d_q,                   
        d_kv=config.model.d_kv,                 
        d_ff=config.model.d_ff,                 
        dropout=config.model.dropout,           
        pad_idx=config.model.pad_idx,           
        max_seq_len=config.model.max_seq_len,   
        label_smoothing=config.model.label_smoothing, 
    )
    
    # Move to device
    device = torch.device(config.__dict__.get('device', 'cuda' if torch.cuda.is_available() else 'cpu'))
    model = model.to(device)
    
    return model
