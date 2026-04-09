#!/usr/bin/env python3
"""
Example: Training ViWordFormer with New Architecture
Shows how to use the registry-based training system
"""

import sys
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent))

import yaml
import logging
from tasks.base_pretraining_task import MLMPretrainingTask
from configs.config import dict_to_dotdict
from models import ViWordFormer  # This imports and registers the model
from tasks import MLMPretrainingTask  # This imports and registers the task

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def example_1_basic_training():
    """Example 1: Basic training with configuration file"""
    logger.info("\n" + "="*60)
    logger.info("Example 1: Basic Training with YAML Config")
    logger.info("="*60)
    
    # Load config
    with open('configs/viwordformer_pretrain.yaml', 'r') as f:
        config_dict = yaml.safe_load(f)
    
    config = dict_to_dotdict(config_dict)
    
    # Create task
    task = MLMPretrainingTask(config)
    
    # Train (commented out to avoid actual training in example)
    # task.train(num_epochs=1)
    
    logger.info("✓ Task created successfully")
    logger.info(f"Model: {task.model.__class__.__name__}")
    logger.info(f"Device: {task.device}")


def example_2_programmatic_config():
    """Example 2: Programmatic configuration"""
    logger.info("\n" + "="*60)
    logger.info("Example 2: Programmatic Configuration")
    logger.info("="*60)
    
    from configs.config import PretrainingConfig
    
    # Create config programmatically
    config = PretrainingConfig()
    config.model.d_model = 512  # Smaller for testing
    config.model.nlayers = 6
    config.training.batch_size = 16
    config.training.num_epochs = 1
    
    # Create task
    task = MLMPretrainingTask(config)
    
    logger.info("✓ Created config programmatically")
    logger.info(f"Hidden dim: {config.model.d_model}")
    logger.info(f"Layers: {config.model.nlayers}")


def example_3_custom_component():
    """Example 3: Registering custom component"""
    logger.info("\n" + "="*60)
    logger.info("Example 3: Custom Component Registration")
    logger.info("="*60)
    
    from builders.registry import META_ARCHITECTURE
    import torch.nn as nn
    
    # Define custom model
    @META_ARCHITECTURE.register()
    class TinyViWordFormer(nn.Module):
        """Tiny model for testing"""
        def __init__(self, vocab_size, d_model=256, **kwargs):
            super().__init__()
            self.embedding = nn.Embedding(vocab_size, d_model)
            self.linear = nn.Linear(d_model, vocab_size)
            self.loss_fn = nn.CrossEntropyLoss()
        
        def forward(self, input_ids, labels=None):
            x = self.embedding(input_ids)
            logits = self.linear(x[:, 0])  # Use CLS token
            loss = None
            if labels is not None:
                loss = self.loss_fn(logits, labels.squeeze(-1))
            return logits, loss, None
    
    # Verify registration
    model_class = META_ARCHITECTURE.get('TinyViWordFormer')
    logger.info(f"✓ Registered custom model: {model_class.__name__}")
    logger.info(f"Available models: {META_ARCHITECTURE.list_names()}")


def example_4_inspect_registry():
    """Example 4: Inspect all registered components"""
    logger.info("\n" + "="*60)
    logger.info("Example 4: Component Registry Inspection")
    logger.info("="*60)
    
    from builders.registry import (
        META_ARCHITECTURE, META_DATASET, META_PRETRAIN_TASK,
        META_TOKENIZER, META_OPTIMIZER, META_SCHEDULER
    )
    
    registries = {
        'ARCHITECTURES': META_ARCHITECTURE,
        'DATASETS': META_DATASET,
        'PRETRAIN_TASKS': META_PRETRAIN_TASK,
        'TOKENIZERS': META_TOKENIZER,
        'OPTIMIZERS': META_OPTIMIZER,
        'SCHEDULERS': META_SCHEDULER,
    }
    
    for reg_name, registry in registries.items():
        names = registry.list_names()
        logger.info(f"{reg_name}: {names if names else 'empty'}")


def example_5_load_checkpoint():
    """Example 5: Load and resume from checkpoint"""
    logger.info("\n" + "="*60)
    logger.info("Example 5: Checkpoint Management")
    logger.info("="*60)
    
    from configs.config import PretrainingConfig
    
    config = PretrainingConfig()
    task = MLMPretrainingTask(config)
    
    # Save checkpoint
    task.save_checkpoint(tag='demo_checkpoint')
    logger.info("✓ Saved checkpoint to demo_checkpoint")
    
    # Load checkpoint
    checkpoint_path = Path('checkpoints/demo_checkpoint')
    if checkpoint_path.exists():
        task.load_checkpoint(str(checkpoint_path))
        logger.info("✓ Loaded checkpoint from demo_checkpoint")


def example_6_configuration_variations():
    """Example 6: Different configuration variations"""
    logger.info("\n" + "="*60)
    logger.info("Example 6: Configuration Variations")
    logger.info("="*60)
    
    from configs.config import PretrainingConfig
    
    # Small model for testing
    small_config = PretrainingConfig()
    small_config.model.d_model = 256
    small_config.model.nlayers = 2
    small_config.training.batch_size = 8
    logger.info("✓ Small model config: d_model=256, nlayers=2")
    
    # Medium model for development
    medium_config = PretrainingConfig()
    medium_config.model.d_model = 512
    medium_config.model.nlayers = 6
    medium_config.training.batch_size = 32
    logger.info("✓ Medium model config: d_model=512, nlayers=6")
    
    # Large model for production
    large_config = PretrainingConfig()
    large_config.model.d_model = 768
    large_config.model.nlayers = 12
    large_config.training.batch_size = 64
    logger.info("✓ Large model config: d_model=768, nlayers=12")


def run_all_examples():
    """Run all examples"""
    logger.info("ViWordFormer Pretraining Architecture Examples")
    logger.info("=" * 60)
    
    try:
        example_1_basic_training()
    except Exception as e:
        logger.error(f"Example 1 failed: {e}")
    
    try:
        example_2_programmatic_config()
    except Exception as e:
        logger.error(f"Example 2 failed: {e}")
    
    try:
        example_3_custom_component()
    except Exception as e:
        logger.error(f"Example 3 failed: {e}")
    
    try:
        example_4_inspect_registry()
    except Exception as e:
        logger.error(f"Example 4 failed: {e}")
    
    try:
        example_5_load_checkpoint()
    except Exception as e:
        logger.error(f"Example 5 failed: {e}")
    
    try:
        example_6_configuration_variations()
    except Exception as e:
        logger.error(f"Example 6 failed: {e}")
    
    logger.info("\n" + "="*60)
    logger.info("All examples completed!")
    logger.info("="*60)


if __name__ == "__main__":
    run_all_examples()
