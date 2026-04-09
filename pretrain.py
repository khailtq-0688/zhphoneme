#!/usr/bin/env python3
"""
Main pretraining script for ViWordFormer
Using registry-based architecture similar to ViWordFormer
"""

import sys
import os
import logging
from pathlib import Path
from argparse import ArgumentParser
import yaml

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from tasks.base_pretraining_task import MLMPretrainingTask
from configs.config import PretrainingConfig, load_config_from_yaml, dict_to_dotdict
from data_processing.data_processor import VietnameseProcessor, ChineseProcessor, DataMerger
from tokenizer.unigram_tokenizer import UnigramTokenizer
from builders.registry import META_ARCHITECTURE

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


def process_datasets(config):
    """Process Vietnamese and Chinese datasets"""
    data_config = config.get('data_paths', {})
    vietnamese_config = data_config.get('vietnamese', {})
    chinese_config = data_config.get('chinese', {})
    preprocessing_config = config.get('preprocessing', {})
    
    output_dir = preprocessing_config.get('output_dir', './processed_data')
    
    # Check if merged corpus already exists
    merged_file = Path(output_dir) / preprocessing_config.get('merge_output', 'merged_corpus.txt')
    if merged_file.exists():
        logger.info(f"Merged corpus already exists at {merged_file}")
        return
    
    logger.info("="*60)
    logger.info("Processing Datasets")
    logger.info("="*60)
    
    # Vietnamese processing
    vi_path = vietnamese_config.get('path')
    if vi_path and Path(vi_path).exists():
        logger.info(f"Processing Vietnamese dataset from {vi_path}")
        try:
            vi_processor = VietnameseProcessor(
                input_dir=vi_path,
                output_dir=output_dir
            )
            vi_processor.process()
            logger.info("✓ Vietnamese processing complete")
        except Exception as e:
            logger.warning(f"Vietnamese processing failed: {e}")
    else:
        logger.warning(f"Vietnamese dataset path not found: {vi_path}")
    
    # Chinese processing
    zh_path = chinese_config.get('path')
    if zh_path and Path(zh_path).exists():
        logger.info(f"Processing Chinese dataset from {zh_path}")
        try:
            zh_processor = ChineseProcessor(
                input_dir=zh_path,
                output_dir=output_dir
            )
            zh_processor.process()
            logger.info("✓ Chinese processing complete")
        except Exception as e:
            logger.warning(f"Chinese processing failed: {e}")
    else:
        logger.warning(f"Chinese dataset path not found: {zh_path}")
    
    # Merge datasets
    logger.info("Merging processed datasets...")
    try:
        merger = DataMerger(output_dir=output_dir)
        merger.merge_all(output_name=preprocessing_config.get('merge_output', 'merged_corpus.txt'))
        logger.info("✓ Dataset merging complete")
    except Exception as e:
        logger.warning(f"Dataset merging failed: {e}")


def train_tokenizer(config):
    """Train tokenizer if not exists"""
    tokenizer_config = config.tokenizer if hasattr(config, 'tokenizer') else config.get('tokenizer', {})
    model_prefix = tokenizer_config.get('model_prefix', './outputs/unigram_tokenizer') if isinstance(tokenizer_config, dict) else tokenizer_config.model_prefix
    
    model_file = f"{model_prefix}.model"
    if Path(model_file).exists():
        logger.info(f"Tokenizer already exists at {model_file}")
        return
    
    logger.info("="*60)
    logger.info("Training Tokenizer")
    logger.info("="*60)
    
    vocab_size = tokenizer_config.get('vocab_size', 30000) if isinstance(tokenizer_config, dict) else tokenizer_config.vocab_size
    
    tokenizer = UnigramTokenizer(
        model_prefix=model_prefix,
        vocab_size=vocab_size
    )
    
    # Get training files
    output_dir = './processed_data'
    vi_file = Path(output_dir) / 'vietnamese_processed.txt'
    zh_file = Path(output_dir) / 'chinese_processed.txt'
    
    training_files = []
    if vi_file.exists():
        training_files.append(str(vi_file))
    if zh_file.exists():
        training_files.append(str(zh_file))
    
    if not training_files:
        logger.warning("No training files found for tokenizer")
        return
    
    logger.info(f"Training tokenizer with {len(training_files)} files")
    tokenizer.train(training_files=training_files, vocab_size=vocab_size)
    logger.info("✓ Tokenizer training complete")


def main():
    """Main pretraining pipeline"""
    parser = ArgumentParser(description='ViWordFormer Pretraining')
    parser.add_argument('--config', type=str, required=True, help='Path to config file (YAML or JSON)')
    parser.add_argument('--no-process', action='store_true', help='Skip data processing')
    parser.add_argument('--no-tokenizer', action='store_true', help='Skip tokenizer training')
    parser.add_argument('--resume', type=str, help='Path to checkpoint to resume from')
    
    args = parser.parse_args()
    
    # Load configuration
    if not Path(args.config).exists():
        logger.error(f"Config file not found: {args.config}")
        sys.exit(1)
    
    config = load_config(args.config)
    logger.info(f"Loaded configuration from {args.config}")
    
    # Register models
    register_model()
    
    # Process datasets
    if not args.no_process:
        process_datasets(config)
    
    # Train tokenizer
    if not args.no_tokenizer:
        train_tokenizer(config)
    
    # Start pretraining
    logger.info("\n" + "="*60)
    logger.info("Starting MLM Pretraining")
    logger.info("="*60)
    
    try:
        task = MLMPretrainingTask(config)
        
        # Load checkpoint if specified
        if args.resume:
            task.load_checkpoint(args.resume)
        
        # Train
        num_epochs = config.training.get('num_epochs', 3) if isinstance(config.training, dict) else config.training.num_epochs
        task.train(num_epochs=num_epochs)
        
        logger.info("\n✓ Pretraining complete!")
        
    except Exception as e:
        logger.error(f"Training failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
