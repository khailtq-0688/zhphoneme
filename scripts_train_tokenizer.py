#!/usr/bin/env python3
"""
Quick start example for tokenizer training
Run this after preparing your dataset
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from tokenizer.unigram_tokenizer import UnigramTokenizer, TokenizerConfig
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    """Train tokenizer on Vietnamese and Chinese corpus"""
    
    # Configuration
    config = TokenizerConfig()
    config.vocab_size = 30000
    
    # Initialize tokenizer
    tokenizer = UnigramTokenizer(
        model_prefix="./outputs/unigram_tokenizer",
        vocab_size=config.vocab_size
    )
    
    # Training files - UPDATE THESE PATHS
    training_files = [
        # Vietnamese Curated Dataset
        "/path/to/vietnamese/curated/dataset/corpus.txt",
        # Chinese CommonCrawl Dataset
        "/path/to/chinese/commoncrawl/dataset/corpus.txt",
    ]
    
    logger.info(f"Training Unigram tokenizer with {len(training_files)} files")
    logger.info(f"Vocabulary size: {config.vocab_size}")
    
    try:
        tokenizer.train(
            training_files=training_files,
            vocab_size=config.vocab_size
        )
        
        # Save configuration
        config.save("./outputs/tokenizer_config.json")
        logger.info("Tokenizer training complete!")
        
        # Test tokenizer
        logger.info("\n=== Testing Tokenizer ===")
        test_vi = "xin chào thế giới"
        test_zh = "你好世界"
        
        print(f"\nVietnamese: {test_vi}")
        print(f"Encoded: {tokenizer.encode(test_vi)}")
        print(f"Pieces: {tokenizer.encode_pieces(test_vi)}")
        
        print(f"\nChinese: {test_zh}")
        print(f"Encoded: {tokenizer.encode(test_zh)}")
        print(f"Pieces: {tokenizer.encode_pieces(test_zh)}")
        
        print(f"\nVocabulary size: {tokenizer.get_vocab_size()}")
        
    except Exception as e:
        logger.error(f"Error training tokenizer: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
