#!/usr/bin/env python3
"""
Quick start example for data processing
Run this after placing your dataset files
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from data_processing.data_processor import (
    VietnameseProcessor,
    ChineseProcessor,
    DataMerger
)
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    """Process Vietnamese and Chinese datasets"""
    
    output_dir = "./processed_data"
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Vietnamese Processing
    logger.info("="*60)
    logger.info("Processing Vietnamese (Curated) Dataset")
    logger.info("="*60)
    
    vi_input_dir = "/path/to/vietnamese/curated/dataset"
    
    try:
        vi_processor = VietnameseProcessor(
            input_dir=vi_input_dir,
            output_dir=output_dir
        )
        vi_processor.process()
        logger.info("✓ Vietnamese processing complete")
    except Exception as e:
        logger.error(f"Vietnamese processing failed: {e}")
    
    # Chinese Processing
    logger.info("\n" + "="*60)
    logger.info("Processing Chinese (CommonCrawl) Dataset")
    logger.info("="*60)
    
    zh_input_dir = "/path/to/chinese/commoncrawl/dataset"
    
    try:
        zh_processor = ChineseProcessor(
            input_dir=zh_input_dir,
            output_dir=output_dir
        )
        zh_processor.process()
        logger.info("✓ Chinese processing complete")
    except Exception as e:
        logger.error(f"Chinese processing failed: {e}")
    
    # Merge Datasets
    logger.info("\n" + "="*60)
    logger.info("Merging Datasets")
    logger.info("="*60)
    
    try:
        merger = DataMerger(output_dir=output_dir)
        merger.merge_all(output_name="merged_corpus.txt")
        logger.info("✓ Dataset merging complete")
        
        # Print statistics
        merged_file = Path(output_dir) / "merged_corpus.txt"
        if merged_file.exists():
            with open(merged_file) as f:
                line_count = sum(1 for _ in f)
            logger.info(f"\nFinal corpus: {line_count:,} lines")
            logger.info(f"Location: {merged_file}")
        
    except Exception as e:
        logger.error(f"Dataset merging failed: {e}")


if __name__ == "__main__":
    main()
