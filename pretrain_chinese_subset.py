#!/usr/bin/env python3
"""
Chinese Independent Pretraining Script (Subset Format)
Trains a ViWordFormer model from scratch on all Chinese Baidu Baike subset files
Treats subset_*.txt files as one continuous corpus transparently
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
from tokenizer.unigram_tokenizer import UnigramTokenizer
from builders.registry import META_ARCHITECTURE
from builders.model_builder import build_model
from builders.dataset_builder import SubsetDataset, collate_fn
from torch.utils.data import DataLoader

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


def train_tokenizer_on_subset_files(config, corpus_dir):
    """Train tokenizer on all subset files"""
    tokenizer_config = config.get('tokenizer', {})
    model_prefix = tokenizer_config.get('model_prefix', './tokenizers/unigram_tokenizer_chinese_subset')
    
    model_file = f"{model_prefix}.model"
    if Path(model_file).exists():
        logger.info(f"Tokenizer already exists at {model_file}")
        return model_prefix
    
    logger.info("="*60)
    logger.info("Training Tokenizer on All Subset Files")
    logger.info("="*60)
    
    vocab_size = tokenizer_config.get('vocab_size', 30000)
    
    # Create temporary merged file for tokenizer training
    temp_training_file = Path(corpus_dir) / 'tokenizer_training.txt'
    logger.info(f"Merging all subset files for tokenizer training...")
    
    with open(temp_training_file, 'w', encoding='utf-8') as out_f:
        for filename in sorted(os.listdir(corpus_dir)):
            if filename.startswith('subset_') and filename.endswith('.txt'):
                filepath = os.path.join(corpus_dir, filename)
                with open(filepath, 'r', encoding='utf-8', errors='ignore') as in_f:
                    for line in in_f:
                        out_f.write(line)
    
    logger.info(f"Training tokenizer (vocab_size={vocab_size})")
    
    tokenizer = UnigramTokenizer(
        model_prefix=model_prefix,
        vocab_size=vocab_size
    )
    
    tokenizer.train(
        corpus_path=str(temp_training_file),
        vocab_size=vocab_size,
        character_coverage=0.9999,
        model_type='unigram'
    )
    
    # Clean up temporary file
    temp_training_file.unlink()
    
    logger.info(f"✓ Tokenizer saved to {model_prefix}")
    return model_prefix


def main():
    parser = ArgumentParser(description='Chinese Pretraining on Subset Format Corpus')
    parser.add_argument(
        '--config',
        type=str,
        default='./configs/viwordformer_pretrain_chinese_subset.yaml',
        help='Path to config file'
    )
    parser.add_argument(
        '--corpus-dir',
        type=str,
        default='../../baidubaike_chinese',
        help='Path to corpus directory with subset_*.txt files'
    )
    parser.add_argument(
        '--resume',
        type=str,
        default=None,
        help='Path to checkpoint to resume from'
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
    
    # Resolve corpus directory
    corpus_dir = Path(args.corpus_dir)
    if not corpus_dir.is_absolute():
        corpus_dir = Path(__file__).parent / corpus_dir
    
    if not corpus_dir.exists():
        raise FileNotFoundError(f"Corpus directory not found: {corpus_dir}")
    
    subset_files = list(corpus_dir.glob('subset_*.txt'))
    logger.info(f"Corpus directory: {corpus_dir}")
    logger.info(f"Found {len(subset_files)} subset files")
    
    # Train or load tokenizer
    if not args.no_tokenizer:
        tokenizer_prefix = train_tokenizer_on_subset_files(config, corpus_dir)
        tokenizer_path = f"{tokenizer_prefix}.model"
    else:
        tokenizer_path = Path('./tokenizers/unigram_tokenizer_chinese_subset.model')
        if not tokenizer_path.exists():
            raise FileNotFoundError(f"Tokenizer not found: {tokenizer_path}")
    
    logger.info("="*60)
    logger.info("Loading Tokenizer")
    logger.info("="*60)
    
    tokenizer = UnigramTokenizer()
    tokenizer.load(str(tokenizer_path))
    logger.info(f"✓ Tokenizer loaded from {tokenizer_path}")
    
    # Create dataset from all subset files
    logger.info("="*60)
    logger.info("Creating SubsetDataset from All Files")
    logger.info("="*60)
    
    dataset = SubsetDataset(
        corpus_dir=str(corpus_dir),
        tokenizer=tokenizer,
        max_seq_len=config.get('dataset', {}).get('max_seq_len', 512),
        mlm_probability=config.get('dataset', {}).get('mlm_probability', 0.15),
        lines_per_file=config.get('dataset', {}).get('lines_per_file', 1000)
    )
    
    logger.info(f"✓ SubsetDataset created with {len(dataset)} total samples")
    
    # Create dataloader
    batch_size = config.get('training', {}).get('batch_size', 32)
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        collate_fn=collate_fn,
        pin_memory=True if device == 'cuda' else False
    )
    logger.info(f"✓ DataLoader created: {len(dataloader)} batches per epoch")
    
    # Create training task and train
    logger.info("="*60)
    logger.info("Starting Pretraining on All Subset Files")
    logger.info("="*60)
    
    training_config = dict_to_dotdict({
        'device': device,
        'tokenizer': tokenizer,
        'checkpoint_dir': Path('./checkpoints/chinese_subset_pretrain'),
        'optimizer': config.get('training', {}).get('optimizer', 'adamw'),
        'learning_rate': config.get('training', {}).get('learning_rate', 6e-4),
        'weight_decay': config.get('training', {}).get('weight_decay', 0.01),
        'betas': tuple(config.get('training', {}).get('betas', [0.9, 0.98])),
        'eps': config.get('training', {}).get('eps', 1e-6),
        'batch_size': batch_size,
        'num_epochs': config.get('training', {}).get('num_epochs', 5),
        'max_seq_len': config.get('dataset', {}).get('max_seq_len', 512),
        'mlm_probability': config.get('dataset', {}).get('mlm_probability', 0.15),
    })
    
    task = MLMPretrainingTask(training_config)
    
    if args.resume:
        logger.info(f"Resuming from checkpoint: {args.resume}")
        task.load_checkpoint(args.resume)
    
    # Train with the dataloader
    task.train(
        num_epochs=training_config.num_epochs,
        train_dataloader=dataloader
    )
    
    logger.info("="*60)
    logger.info("Pretraining on All Subset Files Complete!")
    logger.info(f"Model saved to {training_config.checkpoint_dir}")
    logger.info("="*60)


if __name__ == "__main__":
    main()

