#!/usr/bin/env python3
"""
Vietnamese Independent Pretraining Script (Subset Format)
Trains a ViWordFormer model from scratch on all Vietnamese Curated subset files
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
from tokenizer.bpe_tokenizer import BpeTokenizer
from builders.registry import META_ARCHITECTURE
from builders.model_builder import build_model
from builders.dataset_builder import SubsetDataset, collate_fn
from torch.utils.data import DataLoader, random_split

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
    model_prefix = tokenizer_config.get('model_prefix', './tokenizers/unigram_tokenizer_vietnamese_subset')
    
    tokenizer_wrapper =BpeTokenizer(model_prefix)
    
    tokenizer_wrapper.train(corpus_dir)

    return tokenizer_wrapper.get_tokenizer()

def main():
    parser = ArgumentParser(description='Vietnamese Pretraining on Subset Format Corpus')
    parser.add_argument(
        '--config',
        type=str,
        default='./configs/viwordformer_pretrain_vietnamese_subset.yaml',
        help='Path to config file'
    )
    parser.add_argument(
        '--corpus-dir',
        type=str,
        default='../../vietnamese_curated',
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
    corpus_dir = args.corpus_dir
    if not os.path.isdir(corpus_dir):
        corpus_dir = os.path.join(Path(__file__).parent, corpus_dir)
    
    if not os.path.isdir(corpus_dir):
        raise FileNotFoundError(f"Corpus directory not found: {corpus_dir}")
    
    # Train or load tokenizer
    logger.info(f"Loading tokenizer")
    tokenizer = train_tokenizer_on_subset_files(config, corpus_dir)
    
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

    val_size = int(0.1 * len(dataset))
    train_size = len(dataset) - val_size

    train_dataset, val_dataset = random_split(
        dataset, 
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42) # Cố định seed để dễ tái lập
    )

    logger.info(f"✓ Split dataset: {train_size} train samples, {val_size} val samples")
    
    # Create dataloader
    batch_size = config.get('training', {}).get('batch_size', 32)
    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        collate_fn=collate_fn,
        pin_memory=True if device == 'cuda' else False
    )

    val_dataloader = DataLoader(
        dataset=val_dataset,
        batch_size=batch_size,
        shuffle=False, # Val không cần shuffle
        num_workers=4,
        collate_fn=collate_fn,
        pin_memory=True if device == 'cuda' else False
    )

    logger.info(f"✓ Train DataLoader created: {len(train_dataloader)} batches per epoch")
    
    # Create training task and train
    logger.info("="*60)
    logger.info("Starting Pretraining on All Subset Files")
    logger.info("="*60)
    
    training_config = dict_to_dotdict({
        'model': config.model,
        'device': device,
        'tokenizer': tokenizer,
        'checkpoint_dir': Path('./checkpoints/vietnamese_subset_pretrain'),
        'optimizer': config.get('training', {}).get('optimizer', 'adamw'),
        'learning_rate': config.get('training', {}).get('learning_rate', 5e-5),
        'weight_decay': config.get('training', {}).get('weight_decay', 0.01),
        'betas': tuple(config.get('training', {}).get('betas', [0.9, 0.999])),
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
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader
    )
    
    logger.info("="*60)
    logger.info("Pretraining on All Subset Files Complete!")
    logger.info(f"Model saved to {training_config.checkpoint_dir}")
    logger.info("="*60)


if __name__ == "__main__":
    main()
