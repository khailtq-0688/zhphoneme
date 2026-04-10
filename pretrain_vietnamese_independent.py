#!/usr/bin/env python3
"""
Vietnamese Independent Pretraining Script
Trains a ViWordFormer model from scratch on Vietnamese Curated dataset only
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
from data_processing.data_processor import VietnameseProcessor
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
    from models.viwordformer import ViWordFormer
    logger.info("Registering ViWordFormer model...")
    META_ARCHITECTURE.register(ViWordFormer)


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
    """Process Vietnamese dataset without merging"""
    data_config = config.get('data_paths', {})
    vietnamese_config = data_config.get('vietnamese', {})
    preprocessing_config = config.get('preprocessing', {})
    
    output_dir = preprocessing_config.get('output_dir', './processed_data/vietnamese_pretrain')
    
    # Check if processed file already exists
    processed_file = Path(output_dir) / "vietnamese_processed.txt"
    if processed_file.exists():
        logger.info(f"Vietnamese processed corpus already exists at {processed_file}")
        return output_dir
    
    logger.info("="*60)
    logger.info("Processing Vietnamese Dataset (Independent)")
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


def train_tokenizer_vietnamese(config):
    """Train tokenizer on Vietnamese data only"""
    tokenizer_config = config.get('tokenizer', {})
    model_prefix = tokenizer_config.get('model_prefix', './tokenizers/unigram_tokenizer_vietnamese')
    
    model_file = f"{model_prefix}.model"
    if Path(model_file).exists():
        logger.info(f"Vietnamese tokenizer already exists at {model_file}")
        return model_prefix
    
    logger.info("="*60)
    logger.info("Training Vietnamese Tokenizer")
    logger.info("="*60)
    
    vocab_size = tokenizer_config.get('vocab_size', 30000)
    output_dir = './processed_data/vietnamese_pretrain'
    training_file = Path(output_dir) / 'vietnamese_processed.txt'
    
    if not training_file.exists():
        raise FileNotFoundError(f"Vietnamese training data not found: {training_file}")
    
    logger.info(f"Training Vietnamese tokenizer (vocab_size={vocab_size})")
    logger.info(f"Training data: {training_file}")
    
    tokenizer = UnigramTokenizer(
        model_prefix=model_prefix,
        vocab_size=vocab_size
    )
    
    tokenizer.train(
        corpus_path=str(training_file),
        vocab_size=vocab_size,
        character_coverage=0.9999,
        model_type='unigram'
    )
    
    logger.info(f"✓ Vietnamese tokenizer saved to {model_prefix}")
    return model_prefix


def main():
    parser = ArgumentParser(description='Vietnamese Independent Pretraining for ViWordFormer')
    parser.add_argument(
        '--config',
        type=str,
        default='./configs/viwordformer_pretrain_vietnamese.yaml',
        help='Path to config file (YAML or JSON)'
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
    parser.add_argument(
        '--no-tokenizer',
        action='store_true',
        help='Skip tokenizer training if already done'
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
        processed_dir = config.get('preprocessing', {}).get('output_dir', './processed_data/vietnamese_pretrain')
    
    # Train or load tokenizer
    if not args.no_tokenizer:
        tokenizer_prefix = train_tokenizer_vietnamese(config)
        tokenizer_path = f"{tokenizer_prefix}.model"
    else:
        tokenizer_path = Path('./tokenizers/unigram_tokenizer_vietnamese.model')
        if not tokenizer_path.exists():
            raise FileNotFoundError(f"Tokenizer not found: {tokenizer_path}")
    
    logger.info("="*60)
    logger.info("Loading Tokenizer")
    logger.info("="*60)
    
    tokenizer = UnigramTokenizer()
    tokenizer.load(str(tokenizer_path))
    logger.info(f"✓ Tokenizer loaded from {tokenizer_path}")
    
    # Create task and train
    logger.info("="*60)
    logger.info("Starting Vietnamese Independent Pretraining")
    logger.info("="*60)
    
    # Create config for training
    training_config = dict_to_dotdict({
        'device': device,
        'tokenizer': tokenizer,
        'data_path': Path(processed_dir) / "vietnamese_processed.txt",
        'checkpoint_dir': Path('./checkpoints/vietnamese_pretrain'),
        'optimizer': config.get('training', {}).get('optimizer', 'adamw'),
        'learning_rate': config.get('training', {}).get('learning_rate', 0.0001),
        'batch_size': config.get('training', {}).get('batch_size', 32),
        'num_epochs': config.get('training', {}).get('num_epochs', 3),
        'max_seq_len': config.get('dataset', {}).get('max_seq_len', 512),
        'mlm_probability': config.get('dataset', {}).get('mlm_probability', 0.15),
    })
    
    task = MLMPretrainingTask(training_config)
    
    # Resume from checkpoint if specified
    if args.resume:
        logger.info(f"Resuming from checkpoint: {args.resume}")
        task.load_checkpoint(args.resume)
    
    # Train
    task.train(num_epochs=training_config.num_epochs)
    
    logger.info("="*60)
    logger.info("Vietnamese Independent Pretraining Complete!")
    logger.info(f"Model saved to {training_config.checkpoint_dir}")
    logger.info("="*60)


if __name__ == "__main__":
    main()
