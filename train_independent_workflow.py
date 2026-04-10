#!/usr/bin/env python3
"""
Complete Independent Training Workflow
Two separate models pretrained from scratch on each language:
1. Vietnamese model trained on Curated dataset
2. Chinese model trained on CommonCrawl dataset
"""

import subprocess
import sys
from pathlib import Path
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def run_command(cmd, description):
    """Run a shell command and handle errors"""
    logger.info("="*70)
    logger.info(f"STAGE: {description}")
    logger.info("="*70)
    logger.info(f"Command: {' '.join(cmd)}")
    logger.info("-"*70)
    
    result = subprocess.run(cmd, cwd=Path(__file__).parent)
    
    if result.returncode != 0:
        logger.error(f"✗ {description} FAILED")
        sys.exit(1)
    
    logger.info(f"✓ {description} COMPLETE")
    logger.info("="*70)
    logger.info("")


def main():
    logger.info("""
    ╔════════════════════════════════════════════════════════════════════╗
    ║  ViWordFormer Independent Pretraining Workflow                    ║
    ║  Two Separate Models Trained from Scratch                         ║
    ║                                                                    ║
    ║  Stage 1: Vietnamese Pretraining (Curated Dataset)                ║
    ║  Stage 2: Chinese Pretraining (CommonCrawl Dataset)               ║
    ╚════════════════════════════════════════════════════════════════════╝
    """)
    
    # Stage 1: Vietnamese Pretraining
    logger.info("Starting Stage 1: Vietnamese Independent Pretraining")
    logger.info("This trains a ViWordFormer model from scratch on Vietnamese data")
    logger.info("")
    
    run_command(
        ['python', 'pretrain_vietnamese_independent.py',
         '--config', 'configs/viwordformer_pretrain_vietnamese.yaml'],
        "Stage 1 - Vietnamese Independent Pretraining"
    )
    
    logger.info("✓ Vietnamese model saved to: checkpoints/vietnamese_pretrain/best/model.pt")
    logger.info("✓ Vietnamese tokenizer saved to: tokenizers/unigram_tokenizer_vietnamese.model")
    logger.info("")
    
    # Stage 2: Chinese Pretraining
    logger.info("Starting Stage 2: Chinese Independent Pretraining")
    logger.info("This trains a ViWordFormer model from scratch on Chinese data")
    logger.info("")
    
    run_command(
        ['python', 'pretrain_chinese_independent.py',
         '--config', 'configs/viwordformer_pretrain_chinese.yaml'],
        "Stage 2 - Chinese Independent Pretraining"
    )
    
    logger.info("✓ Chinese model saved to: checkpoints/chinese_pretrain/best/model.pt")
    logger.info("✓ Chinese tokenizer saved to: tokenizers/unigram_tokenizer_chinese.model")
    logger.info("")
    
    # Summary
    logger.info("""
    ╔════════════════════════════════════════════════════════════════════╗
    ║  INDEPENDENT PRETRAINING COMPLETE! You now have 2 models:        ║
    ╠════════════════════════════════════════════════════════════════════╣
    ║                                                                    ║
    ║  1. Vietnamese Pretrained Model                                   ║
    ║     → Model: checkpoints/vietnamese_pretrain/best/model.pt        ║
    ║     → Tokenizer: tokenizers/unigram_tokenizer_vietnamese.model    ║
    ║     → Dataset: Vietnamese Curated                                 ║
    ║     → Training objective: Masked Language Modeling (MLM)          ║
    ║     → Use case: Vietnamese language processing tasks              ║
    ║                                                                    ║
    ║  2. Chinese Pretrained Model                                      ║
    ║     → Model: checkpoints/chinese_pretrain/best/model.pt           ║
    ║     → Tokenizer: tokenizers/unigram_tokenizer_chinese.model       ║
    ║     → Dataset: Chinese CommonCrawl                                ║
    ║     → Training objective: Masked Language Modeling (MLM)          ║
    ║     → Use case: Chinese language processing tasks                 ║
    ║                                                                    ║
    ║  Key Differences from Transfer Learning Approach:                 ║
    ║  - Each model trained independently from scratch                  ║
    ║  - Language-specific tokenizers                                   ║
    ║  - No knowledge transfer between languages                        ║
    ║  - Optimized for individual language characteristics              ║
    ║                                                                    ║
    ║  Model Architecture (Both Models):                                ║
    ║  - Type: ViWordFormer                                             ║
    ║  - Dimensions: 768                                                ║
    ║  - Layers: 12                                                     ║
    ║  - Attention Heads: 12                                            ║
    ║  - Vocabulary Size: 30,000                                        ║
    ║  - Max Sequence Length: 512                                       ║
    ║  - MLM Masking Rate: 15%                                          ║
    ║                                                                    ║
    ║  Next Steps:                                                       ║
    ║  1. Evaluate both models on language-specific benchmarks          ║
    ║  2. Fine-tune on downstream tasks (classification, NER, etc.)     ║
    ║  3. Compare performance between models                            ║
    ║  4. Use preferred model for production deployment                 ║
    ╚════════════════════════════════════════════════════════════════════╝
    """)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
