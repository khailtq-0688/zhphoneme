#!/usr/bin/env python3
"""
Complete Training Workflow: Multilingual Pretraining + Language-Specific Fine-tuning
This script demonstrates the full 3-stage training process
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
    ║  ViWordFormer Complete Training Workflow                          ║
    ║  Stage 1: Multilingual Pretraining (Vietnamese + Chinese)         ║
    ║  Stage 2: Vietnamese-Specific Fine-tuning                         ║
    ║  Stage 3: Chinese-Specific Fine-tuning                            ║
    ╚════════════════════════════════════════════════════════════════════╝
    """)
    
    # Stage 1: Multilingual Pretraining
    logger.info("Starting Stage 1: Multilingual Pretraining")
    logger.info("This will train a single model on merged Vietnamese + Chinese data")
    logger.info("")
    
    run_command(
        ['python', 'pretrain.py', '--config', 'configs/viwordformer_pretrain.yaml'],
        "Stage 1 - Multilingual Pretraining"
    )
    
    logger.info("✓ Multilingual model saved to: checkpoints/best/model.pt")
    logger.info("")
    
    # Stage 2: Vietnamese Fine-tuning
    logger.info("Starting Stage 2: Vietnamese Fine-tuning")
    logger.info("This will load the pretrained model and fine-tune on Vietnamese data only")
    logger.info("")
    
    pretrained_model = "checkpoints/best/model.pt"
    
    run_command(
        ['python', 'finetune_vietnamese.py',
         '--config', 'configs/viwordformer_finetune_vietnamese.yaml',
         '--pretrained-model', pretrained_model],
        "Stage 2 - Vietnamese Fine-tuning"
    )
    
    logger.info("✓ Vietnamese model saved to: checkpoints/vietnamese_finetune/best/model.pt")
    logger.info("")
    
    # Stage 3: Chinese Fine-tuning
    logger.info("Starting Stage 3: Chinese Fine-tuning")
    logger.info("This will load the pretrained model and fine-tune on Chinese data only")
    logger.info("")
    
    run_command(
        ['python', 'finetune_chinese.py',
         '--config', 'configs/viwordformer_finetune_chinese.yaml',
         '--pretrained-model', pretrained_model],
        "Stage 3 - Chinese Fine-tuning"
    )
    
    logger.info("✓ Chinese model saved to: checkpoints/chinese_finetune/best/model.pt")
    logger.info("")
    
    # Summary
    logger.info("""
    ╔════════════════════════════════════════════════════════════════════╗
    ║  TRAINING COMPLETE! You now have 3 models:                        ║
    ╠════════════════════════════════════════════════════════════════════╣
    ║  1. Multilingual Model                                             ║
    ║     → checkpoints/best/model.pt                                    ║
    ║     → Trained on Vietnamese + Chinese combined                     ║
    ║     → Best for cross-lingual tasks                                 ║
    ║                                                                    ║
    ║  2. Vietnamese-Specific Model                                      ║
    ║     → checkpoints/vietnamese_finetune/best/model.pt                ║
    ║     → Fine-tuned on Vietnamese data from pretrained model          ║
    ║     → Best for Vietnamese-specific tasks                           ║
    ║                                                                    ║
    ║  3. Chinese-Specific Model                                         ║
    ║     → checkpoints/chinese_finetune/best/model.pt                   ║
    ║     → Fine-tuned on Chinese data from pretrained model             ║
    ║     → Best for Chinese-specific tasks                              ║
    ╚════════════════════════════════════════════════════════════════════╝
    """)


if __name__ == "__main__":
    main()
