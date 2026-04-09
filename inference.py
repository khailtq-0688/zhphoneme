#!/usr/bin/env python3
"""
Inference script for using trained ViWordFormer model
"""

import sys
from pathlib import Path
from typing import Dict

import torch

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from models.viwordformer import Model
from tokenizer.unigram_tokenizer import UnigramTokenizer
from configs.config import PretrainingConfig
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ViWordFormerInference:
    """Inference class for ViWordFormer"""
    
    def __init__(self, config_path: str, model_path: str, tokenizer_path: str):
        """
        Initialize inference
        
        Args:
            config_path: Path to config.json
            model_path: Path to model.pt
            tokenizer_path: Path to tokenizer model
        """
        # Load configuration
        self.config = PretrainingConfig.load(config_path)
        logger.info(f"Loaded config from {config_path}")
        
        # Initialize device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
        
        # Load tokenizer
        self.tokenizer = UnigramTokenizer(
            model_prefix=tokenizer_path,
            vocab_size=self.config.tokenizer.vocab_size
        )
        self.tokenizer.load(tokenizer_path)
        logger.info(f"Loaded tokenizer from {tokenizer_path}")
        
        # Load model
        self.model = Model(
            vocab_size=self.config.model.vocab_size,
            d_model=self.config.model.d_model,
            nlayers=self.config.model.nlayers,
            head=self.config.model.head,
            d_q=self.config.model.d_q,
            d_kv=self.config.model.d_kv,
            d_ff=self.config.model.d_ff,
            dropout=self.config.model.dropout,
            pad_idx=self.config.model.pad_idx,
            max_seq_len=self.config.model.max_seq_len,
        )
        
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()
        logger.info(f"Loaded model from {model_path}")
    
    def prepare_input(self, text: str) -> Dict[str, torch.Tensor]:
        """Prepare input text for model"""
        # Tokenize
        tokens = self.tokenizer.encode(text)
        
        # Truncate
        max_len = self.config.model.max_seq_len
        if len(tokens) > max_len - 2:
            tokens = tokens[:max_len - 2]
        
        # Add special tokens
        input_ids = [1] + tokens + [2]  # [BOS] + tokens + [EOS]
        
        # Pad
        if len(input_ids) < max_len:
            input_ids += [3] * (max_len - len(input_ids))  # PAD
        
        input_ids = torch.tensor(input_ids, dtype=torch.long).unsqueeze(0)
        
        return {'input_ids': input_ids.to(self.device)}
    
    @torch.no_grad()
    def predict(self, text: str):
        """
        Get model predictions
        
        Args:
            text: Input text
            
        Returns:
            logits, attention weights
        """
        inputs = self.prepare_input(text)
        logits, _, attentions = self.model(inputs['input_ids'])
        
        return {
            'logits': logits,
            'attentions': attentions,
        }
    
    @torch.no_grad()
    def get_embeddings(self, text: str) -> torch.Tensor:
        """Get token embeddings"""
        inputs = self.prepare_input(text)
        
        with torch.no_grad():
            embeddings = self.model.embedding(inputs['input_ids'])
        
        return embeddings


def main():
    """Main inference example"""
    
    # Configuration paths - UPDATE THESE
    config_path = "./checkpoints/best/config.json"
    model_path = "./checkpoints/best/model.pt"
    tokenizer_path = "./outputs/unigram_tokenizer"
    
    # Check if files exist
    if not Path(config_path).exists():
        logger.error(f"Config not found: {config_path}")
        logger.info("Please train model first and ensure checkpoint exists")
        return
    
    # Initialize inference
    try:
        inf = ViWordFormerInference(
            config_path=config_path,
            model_path=model_path,
            tokenizer_path=tokenizer_path
        )
    except Exception as e:
        logger.error(f"Failed to initialize inference: {e}")
        return
    
    # Test inference
    logger.info("\n" + "="*60)
    logger.info("Model Inference Demo")
    logger.info("="*60)
    
    # Vietnamese text
    vi_text = "xin chào thế giới"
    logger.info(f"\nVietnamese: {vi_text}")
    try:
        result = inf.predict(vi_text)
        logger.info(f"Logits shape: {result['logits'].shape}")
    except Exception as e:
        logger.error(f"Inference failed: {e}")
    
    # Chinese text
    zh_text = "你好世界"
    logger.info(f"\nChinese: {zh_text}")
    try:
        result = inf.predict(zh_text)
        logger.info(f"Logits shape: {result['logits'].shape}")
    except Exception as e:
        logger.error(f"Inference failed: {e}")


if __name__ == "__main__":
    main()
