#!/usr/bin/env python3
"""
Vietnamese Language-Specific Fine-tuning Script
Loads pretrained multilingual model and fine-tunes on Vietnamese dataset
"""

import sys
import os
import logging
from pathlib import Path
from argparse import ArgumentParser
import yaml
import torch

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from tasks.base_pretraining_task import MLMPretrainingTask
from configs.config import PretrainingConfig, load_config_from_yaml, dict_to_dotdict
from data_processing.data_processor import VietnameseProcessor, DataMerger
from tokenizer.unigram_tokenizer import UnigramTokenizer
from builders.registry import META_ARCHITECTURE
from builders.model_builder import build_model

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def register_model():
    """Register models with architecture registry"""
    from models.viwordformer import Model
    logger.info("Registering Model...")
    META_ARCHITECTURE.register(Model)


def load_config(config_path: str):
    """Load configuration from YAML or JSON"""
    config_path = Path(config_path)
    
    if config_path.suffix == '.yaml' or config_path.suffix == '.yml':
        logger.info(f"Loading YAML config from {config_path}")
        with open(config_path, 'r', encoding='utf-8') as f:
            config_dict = yaml.safe_load(f)
        return dict_to_dotdict(config_dict)
    else:
        logger.info(f"Loading JSON config from {config_path}")
        config = PretrainingConfig.load(str(config_path))
        return config


def process_vietnamese_dataset(config):
    """Process Vietnamese dataset only (no merging)"""
    data_config = config.get('data_paths', {})
    vietnamese_config = data_config.get('vietnamese', {})
    preprocessing_config = config.get('preprocessing', {})
    
    output_dir = preprocessing_config.get('output_dir', './processed_data/vietnamese_finetune')
    
    # Check if processed file already exists
    processed_file = Path(output_dir) / "vietnamese_processed.txt"
    if processed_file.exists():
        logger.info(f"Vietnamese processed corpus already exists at {processed_file}")
        return output_dir
    
    logger.info("="*60)
    logger.info("Processing Vietnamese Dataset for Fine-tuning")
    logger.info("="*60)
    
    # Vietnamese processing
    vi_path = vietnamese_config.get('path')
    if not vi_path or not Path(vi_path).exists():
        raise FileNotFoundError(f"Vietnamese dataset path not found: {vi_path}")
    
    logger.info(f"Processing Vietnamese dataset from {vi_path}")
    vi_processor = VietnameseProcessor(
        input_dir=vi_path,
        output_dir=output_dir
    )
    vi_processor.process()
    logger.info("✓ Vietnamese processing complete")
    
    return output_dir


def load_pretrained_model(model_config, checkpoint_path: str, device='cuda'):
    """Load pretrained model from checkpoint"""
    logger.info("="*60)
    logger.info("Loading Pretrained Model")
    logger.info("="*60)
    
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Pretrained model checkpoint not found: {checkpoint_path}")
    
    logger.info(f"Loading checkpoint from {checkpoint_path}")
    
    # Load model architecture
    model = build_model(model_config, vocab_size=30000)
    
    # Load state dict
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    logger.info("✓ Pretrained model loaded successfully")
    return model


def main():
    parser = ArgumentParser(description='Vietnamese Fine-tuning for ViWordFormer')
    parser.add_argument(
        '--config',
        type=str,
        default='./configs/viwordformer_pretrain.yaml',
        help='Path to config file (YAML or JSON)'
    )
    parser.add_argument(
        '--pretrained-model',
        type=str,
        required=True,
        help='Path to pretrained model checkpoint'
    )
    parser.add_argument(
        '--resume',
        type=str,
        default=None,
        help='Path to checkpoint to resume from'
    )
    parser.add_argument(
        '--no-process',
        action='store_true',
        help='Skip dataset processing if already done'
    )
    
    args = parser.parse_args()
    
    # Register models
    register_model()
    
    # Load config
    logger.info(f"Loading configuration from {args.config}")
    config = load_config(args.config)
    
    # Device setup
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"Using device: {device}")
    
    # Process dataset
    if not args.no_process:
        processed_dir = process_vietnamese_dataset(config)
    else:
        processed_dir = config.get('preprocessing', {}).get('output_dir', './processed_data/vietnamese_finetune')
    
    # Load tokenizer
    logger.info("="*60)
    logger.info("Loading Tokenizer")
    logger.info("="*60)
    
    tokenizer_path = Path('./tokenizers/unigram_tokenizer.model')
    if not tokenizer_path.exists():
        logger.error("Tokenizer not found. Please run pretraining first or train tokenizer separately.")
        sys.exit(1)
    
    tokenizer = UnigramTokenizer()
    tokenizer.load(str(tokenizer_path))
    logger.info(f"✓ Tokenizer loaded from {tokenizer_path}")
    
    # Load pretrained model
    model = load_pretrained_model(
        config.get('model', {}),
        args.pretrained_model,
        device=device
    )
    
    # Create task and fine-tune
    logger.info("="*60)
    logger.info("Starting Vietnamese Fine-tuning")
    logger.info("="*60)
    
    # Update config for fine-tuning
    finetune_config = dict_to_dotdict({
        'device': device,
        'tokenizer': tokenizer,
        'model': model,
        'data_path': Path(processed_dir) / "vietnamese_processed.txt",
        'checkpoint_dir': Path('./checkpoints/vietnamese_finetune'),
        'optimizer': config.get('training', {}).get('optimizer', 'adamw'),
        'learning_rate': config.get('training', {}).get('learning_rate', 0.00005),  # Lower LR for fine-tuning
        'batch_size': config.get('training', {}).get('batch_size', 16),
        'num_epochs': config.get('training', {}).get('num_epochs', 1),
        'max_seq_len': config.get('dataset', {}).get('max_seq_len', 512),
        'mlm_probability': config.get('dataset', {}).get('mlm_probability', 0.15),
    })
    
    task = MLMPretrainingTask(finetune_config)
    
    # Resume from checkpoint if specified
    if args.resume:
        logger.info(f"Resuming from checkpoint: {args.resume}")
        task.load_checkpoint(args.resume)
    
    # Train
    task.train(num_epochs=finetune_config.num_epochs)
    
    logger.info("="*60)
    logger.info("Vietnamese Fine-tuning Complete!")
    logger.info(f"Checkpoints saved to {finetune_config.checkpoint_dir}")
    logger.info("="*60)


if __name__ == "__main__":
    main()
