#!/usr/bin/env python3
"""
ViWordFormer Pretraining
Train ViWordFormer models on Chinese and Vietnamese corpora
"""

import sys
from pathlib import Path
from argparse import ArgumentParser

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from pretrain_chinese_subset import main as train_chinese


def main():
    parser = ArgumentParser(description='ViWordFormer Pretraining')
    parser.add_argument(
        'command',
        choices=['train-chinese', 'help'],
        help='Command to run'
    )
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
    
    if args.command == 'train-chinese':
        # Convert args to sys.argv format for pretrain_chinese_subset.main()
        sys.argv = [
            'pretrain_chinese_subset.py',
            '--config', args.config,
            '--corpus-dir', args.corpus_dir,
        ]
        if args.resume:
            sys.argv.extend(['--resume', args.resume])
        if args.no_tokenizer:
            sys.argv.append('--no-tokenizer')
        
        train_chinese()
    
    elif args.command == 'help':
        print("""
ViWordFormer Pretraining

Usage:
    python main.py train-chinese [OPTIONS]
    
Options:
    --config PATH           Config file path (default: configs/viwordformer_pretrain_chinese_subset.yaml)
    --corpus-dir PATH       Corpus directory with subset_*.txt files (default: ../../baidubaike_chinese)
    --resume PATH           Resume from checkpoint
    --no-tokenizer          Skip tokenizer training if already done

Examples:
    # Train on Chinese corpus
    python main.py train-chinese
    
    # Train with custom config
    python main.py train-chinese --config configs/my_config.yaml
    
    # Resume from checkpoint
    python main.py train-chinese --resume checkpoints/chinese_subset_pretrain/best/model.pt
        """)


if __name__ == "__main__":
    main()
