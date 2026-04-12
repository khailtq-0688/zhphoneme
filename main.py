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
from pretrain_vietnamese_subset import main as train_vietnamese


def main():
    parser = ArgumentParser(description='ViWordFormer Pretraining')
    parser.add_argument(
        'command',
        choices=['train-chinese', 'train-vietnamese', 'help'],
        help='Command to run'
    )
    parser.add_argument(
        '--config',
        type=str,
        default=None,
        help='Path to config file (auto-selected based on language if not specified)'
    )
    parser.add_argument(
        '--corpus-dir',
        type=str,
        default=None,
        help='Path to corpus directory with subset_*.txt files (auto-selected based on language if not specified)'
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
        config = args.config or './configs/viwordformer_pretrain_chinese_subset.yaml'
        corpus_dir = args.corpus_dir or '../../baidubaike_chinese'
        
        sys.argv = [
            'pretrain_chinese_subset.py',
            '--config', config,
            '--corpus-dir', corpus_dir,
        ]
        if args.resume:
            sys.argv.extend(['--resume', args.resume])
        if args.no_tokenizer:
            sys.argv.append('--no-tokenizer')
        
        train_chinese()
    
    elif args.command == 'train-vietnamese':
        config = args.config or './configs/viwordformer_pretrain_vietnamese_subset.yaml'
        corpus_dir = args.corpus_dir or '../../vietnamese_curated'
        
        sys.argv = [
            'pretrain_vietnamese_subset.py',
            '--config', config,
            '--corpus-dir', corpus_dir,
        ]
        if args.resume:
            sys.argv.extend(['--resume', args.resume])
        if args.no_tokenizer:
            sys.argv.append('--no-tokenizer')
        
        train_vietnamese()
    
    elif args.command == 'help':
        print("""
ViWordFormer Pretraining - Train on Chinese and Vietnamese

Usage:
    python main.py train-chinese [OPTIONS]
    python main.py train-vietnamese [OPTIONS]
    python main.py help
    
Languages:
    train-chinese       Train on Baidu Baike Chinese corpus (subset format)
    train-vietnamese    Train on Vietnamese Curated corpus (subset format)
    help               Show this help message
    
Options:
    --config PATH           Config file path
    --corpus-dir PATH       Corpus directory with subset_*.txt files
    --resume PATH           Resume from checkpoint
    --no-tokenizer          Skip tokenizer training if already done

Examples:
    # Train on Chinese corpus
    python main.py train-chinese
    
    # Train on Vietnamese corpus
    python main.py train-vietnamese
    
    # Train with custom corpus directory
    python main.py train-chinese --corpus-dir /path/to/corpus
    
    # Resume from checkpoint
    python main.py train-chinese --resume checkpoints/chinese_subset_pretrain/best/model.pt
    
    # Skip tokenizer training (use existing)
    python main.py train-vietnamese --no-tokenizer

Direct usage (bypassing main.py):
    python pretrain_chinese_subset.py --corpus-dir ../../baidubaike_chinese
    python pretrain_vietnamese_subset.py --corpus-dir ../../vietnamese_curated
        """)


if __name__ == "__main__":
    main()

