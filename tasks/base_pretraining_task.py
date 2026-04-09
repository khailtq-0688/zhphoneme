"""
Base Pretraining Task
Similar to BaseTask in ViWordFormer but for pretraining objectives
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, RandomSampler
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR
from pathlib import Path
import os
import json
import pickle
import numpy as np
import random
from tqdm import tqdm
from typing import Optional, Dict, Any
import logging

from builders.model_builder import build_model
from builders.dataset_builder import build_dataset
from builders.tokenizer_builder import build_tokenizer
from tokenizer.unigram_tokenizer import UnigramTokenizer


class BasePretrainingTask:
    """Base class for pretraining tasks"""
    
    def __init__(self, config):
        """Initialize pretraining task"""
        # Setup logging
        self.logger = self._setup_logger()
        self.config = config
        self.device = torch.device(
            config.training.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
        )
        
        self.logger.info(f"Using device: {self.device}")
        
        # Setup checkpoint directory
        self.checkpoint_dir = Path(config.training.get('checkpoint_dir', './checkpoints'))
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize tokenizer
        self.logger.info("Initializing tokenizer...")
        self.tokenizer = self._load_tokenizer(config)
        
        # Initialize model
        self.logger.info("Building model...")
        self.model = build_model(config.model, vocab_size=self.tokenizer.get_vocab_size())
        self.model.to(self.device)
        
        # Log model info
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        self.logger.info(f"Model parameters - Total: {total_params:,}, Trainable: {trainable_params:,}")
        
        # Initialize optimizer and scheduler
        self.logger.info("Setting up optimizer and scheduler...")
        self._setup_optimizer(config)
        
        # Training state
        self.global_step = 0
        self.best_loss = float('inf')
        self.epoch = 0
        
    def _setup_logger(self):
        """Setup logging"""
        logger = logging.getLogger(__name__)
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        return logger
    
    def _load_tokenizer(self, config) -> UnigramTokenizer:
        """Load or initialize tokenizer"""
        tokenizer = build_tokenizer(config.tokenizer.to_dict() if hasattr(config.tokenizer, 'to_dict') else config.tokenizer)
        return tokenizer
    
    def _setup_optimizer(self, config):
        """Setup optimizer and learning rate scheduler"""
        optimizer_type = config.training.get('optimizer', 'adamw').lower()
        learning_rate = config.training.get('learning_rate', 1e-4)
        
        if optimizer_type == 'adamw':
            self.optimizer = AdamW(
                self.model.parameters(),
                lr=learning_rate,
                weight_decay=config.training.get('weight_decay', 0.01)
            )
        else:
            self.optimizer = Adam(
                self.model.parameters(),
                lr=learning_rate
            )
        
        # Setup scheduler
        scheduler_type = config.training.get('scheduler', 'linear').lower()
        num_epochs = config.training.get('num_epochs', 3)
        
        # Approximate total steps (will be updated in training loop)
        total_steps = 10000  # Placeholder, will be calculated
        
        if scheduler_type == 'linear':
            self.scheduler = LinearLR(
                self.optimizer,
                start_factor=1.0,
                end_factor=0.0,
                total_iters=total_steps
            )
        else:
            self.scheduler = CosineAnnealingLR(
                self.optimizer,
                T_max=total_steps
            )
    
    def load_dataset(self, config) -> DataLoader:
        """Load dataset and create dataloader"""
        dataset = build_dataset(
            config.training.get('dataset', {}),
            self.tokenizer,
            split='train'
        )
        
        dataloader = DataLoader(
            dataset,
            batch_size=config.training.get('batch_size', 32),
            sampler=RandomSampler(dataset),
            num_workers=config.training.get('num_workers', 4),
            pin_memory=True if self.device.type == 'cuda' else False,
        )
        
        return dataloader
    
    def train(self):
        """Training loop - to be implemented by subclasses"""
        raise NotImplementedError("Subclasses must implement train()")
    
    def evaluate(self, dataloader: DataLoader) -> float:
        """Evaluate on dataset"""
        self.model.eval()
        total_loss = 0.0
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc='Evaluating', leave=False):
                input_ids = batch['input_ids'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                _, loss, _ = self.model(input_ids, labels)
                total_loss += loss.item()
        
        return total_loss / len(dataloader)
    
    def save_checkpoint(self, tag: str = 'latest'):
        """Save model checkpoint"""
        checkpoint_path = self.checkpoint_dir / tag
        checkpoint_path.mkdir(parents=True, exist_ok=True)
        
        model_file = checkpoint_path / 'model.pt'
        optimizer_file = checkpoint_path / 'optimizer.pt'
        config_file = checkpoint_path / 'config.json'
        
        # Save model
        torch.save(self.model.state_dict(), model_file)
        self.logger.info(f"Saved model to {model_file}")
        
        # Save optimizer
        torch.save(self.optimizer.state_dict(), optimizer_file)
        
        # Save config
        if hasattr(self.config, 'to_dict'):
            config_dict = self.config.to_dict()
        else:
            config_dict = self.config
        
        with open(config_file, 'w') as f:
            json.dump(config_dict, f, indent=2)
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load from checkpoint"""
        if not os.path.exists(checkpoint_path):
            self.logger.warning(f"Checkpoint not found: {checkpoint_path}")
            return
        
        self.logger.info(f"Loading checkpoint from {checkpoint_path}")
        
        model_file = os.path.join(checkpoint_path, 'model.pt')
        if os.path.exists(model_file):
            self.model.load_state_dict(torch.load(model_file, map_location=self.device))
            self.logger.info(f"Loaded model from {model_file}")


class MLMPretrainingTask(BasePretrainingTask):
    """Masked Language Modeling pretraining task"""
    
    def __init__(self, config):
        super().__init__(config)
    
    def train(self, num_epochs: Optional[int] = None):
        """Training loop for MLM"""
        if num_epochs is None:
            num_epochs = self.config.training.get('num_epochs', 3)
        
        self.logger.info(f"Starting MLM pretraining for {num_epochs} epochs")
        
        # Load dataset
        self.logger.info("Loading dataset...")
        train_dataloader = self.load_dataset(self.config)
        
        # Update scheduler with actual total steps
        total_steps = len(train_dataloader) * num_epochs
        if hasattr(self.scheduler, 'total_iters'):
            self.scheduler.total_iters = total_steps
        
        for epoch in range(num_epochs):
            self.epoch = epoch
            self.logger.info(f"\n{'='*60}")
            self.logger.info(f"Epoch {epoch + 1}/{num_epochs}")
            self.logger.info(f"{'='*60}")
            
            train_loss = self._train_epoch(train_dataloader)
            self.logger.info(f"Epoch {epoch + 1} - Train Loss: {train_loss:.4f}")
            
            # Save checkpoint
            self.save_checkpoint(tag=f'epoch_{epoch + 1}')
            
            # Save best model
            if train_loss < self.best_loss:
                self.best_loss = train_loss
                self.save_checkpoint(tag='best')
                self.logger.info(f"Best loss improved to {self.best_loss:.4f}")
        
        self.logger.info("\n" + "="*60)
        self.logger.info("Training complete!")
        self.logger.info("="*60)
    
    def _train_epoch(self, dataloader: DataLoader) -> float:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0.0
        
        progress_bar = tqdm(dataloader, desc=f'Epoch {self.epoch + 1} Training', leave=True)
        
        for batch_idx, batch in enumerate(progress_bar):
            input_ids = batch['input_ids'].to(self.device)
            labels = batch['labels'].to(self.device)
            
            # Forward pass
            logits, loss, attentions = self.model(input_ids, labels)
            
            # Backward pass
            loss.backward()
            
            # Gradient accumulation
            grad_accum_steps = self.config.training.get('gradient_accumulation_steps', 1)
            if (batch_idx + 1) % grad_accum_steps == 0:
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.training.get('gradient_clip_val', 1.0)
                )
                
                # Optimizer step
                self.optimizer.step()
                self.optimizer.zero_grad()
                
                if self.scheduler is not None:
                    self.scheduler.step()
                
                self.global_step += 1
                
                # Logging
                log_steps = self.config.training.get('log_steps', 100)
                if self.global_step % log_steps == 0:
                    avg_loss = total_loss / (batch_idx + 1)
                    progress_bar.set_postfix({'loss': avg_loss, 'step': self.global_step})
            
            total_loss += loss.item()
        
        return total_loss / len(dataloader)
