"""
Base Pretraining Task
Similar to BaseTask in ViWordFormer but for pretraining objectives
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, LambdaLR
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
            config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
        )
        
        self.logger.info(f"Using device: {self.device}")
        
        # Setup checkpoint directory
        self.checkpoint_dir = Path(config.get('checkpoint_dir', './checkpoints'))
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize tokenizer
        # self.logger.info("Initializing tokenizer...")
        # self.tokenizer = self._load_tokenizer(config)
        
        # Initialize model
        self.logger.info("Building model...")
        if hasattr(config, 'model') and isinstance(config.model, nn.Module):
            self.logger.info("Using provided pretrained model for fine-tuning...")
            self.model = config.model
        else:
            self.logger.info("Building model from scratch...")
            self.model = build_model(config, vocab_size=config.tokenizer.get_vocab_size())
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
        """Setup optimizer and Noam Learning Rate Scheduler"""
        optimizer_type = config.get('optimizer', 'adamw').lower()
        learning_rate = config.get('learning_rate', 1.0) # Với Noam, LR gốc thường để 1.0 vì công thức đã tự scale
        weight_decay = config.get('weight_decay', 0.01)
        betas = config.get('betas', (0.9, 0.98)) # Theo paper ViWordFormer là (0.9, 0.98)
        eps = config.get('eps', 1e-9)
        
        # Lấy d_model từ kiến trúc để tính Noam
        d_model = self.model.d_model 
        
        if optimizer_type == 'adamw':
            self.optimizer = AdamW(
                self.model.parameters(),
                lr=learning_rate,
                weight_decay=weight_decay,
                betas=betas,
                eps=eps
            )
        else:
            self.optimizer = Adam(self.model.parameters(), lr=learning_rate)

        # Định nghĩa Noam Scheduler theo công thức trong paper
        def lr_lambda(current_step):
            step = current_step + 1
            # Công thức: (d_model^-0.5) * min(step^-0.5, step * warmup^-1.5)
            return (d_model ** -0.5) * min(step ** -0.5, step * (self.warmup_steps ** -1.5))
            
        self.scheduler = LambdaLR(self.optimizer, lr_lambda)
    
    def load_dataset(self, config) -> DataLoader:
        """Load dataset and create dataloader"""
        from builders.dataset_builder import collate_fn
        
        dataset = build_dataset(
            config.training.get('dataset', {}),
            self.tokenizer,
            split='train'
        )
        
        dataloader = DataLoader(
            dataset,
            batch_size=config.training.get('batch_size', 64),
            shuffle=True,
            num_workers=config.training.get('num_workers', 4),
            collate_fn=collate_fn,
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

                attention_mask = batch['attention_mask'].to(self.device)
                _, loss, _ = self.model(input_ids, attention_mask=attention_mask, labels=labels)
                
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
    
    def train(self, num_epochs: Optional[int] = None, train_dataloader: Optional['DataLoader'] = None, val_dataloader: Optional['DataLoader'] = None):
        """Training loop for MLM
        
        Args:
            num_epochs: Number of epochs to train
            train_dataloader: Optional pre-built DataLoader. If None, will be loaded from config
        """
        if num_epochs is None:
            num_epochs = self.config.training.get('num_epochs', 3)
        
        self.logger.info(f"Starting MLM pretraining for {num_epochs} epochs")
        
        # Load dataset if not provided
        if train_dataloader is None:
            self.logger.info("Loading dataset...")
            train_dataloader = self.load_dataset(self.config)
        else:
            self.logger.info("Using provided DataLoader")
        
        # Calculate and set actual total steps for scheduler
        self.total_steps = len(train_dataloader) * num_epochs
        warmup_ratio = 0.1
        self.warmup_steps = max(1, int(self.total_steps * warmup_ratio))
        
        self.logger.info(f"Total steps: {self.total_steps}")
        self.logger.info(f"Warmup steps: {self.warmup_steps} (10% of total)")
        self.logger.info(f"Batch size: {self.config.get('batch_size', 64)}")
        self.logger.info(f"Learning rate: {self.config.get('learning_rate', 5e-5)}")
        self.logger.info(f"Scheduler: LambdaLR with linear warmup and decay")
        
        # Khởi tạo các tham số Early Stopping
        best_val_loss = float('inf')
        patience_counter = 0
        max_patience = self.config.training.get('patience', 5) # Lấy từ config
        for epoch in range(num_epochs):
            self.epoch = epoch
            self.logger.info(f"Epoch {epoch + 1}/{num_epochs}")
            
            # Huấn luyện một epoch
            train_loss = self._train_epoch(train_dataloader)
            self.logger.info(f"Epoch {epoch + 1} - Train Loss: {train_loss:.4f}")

            # Đánh giá trên tập Validation (Nếu có)
            if val_dataloader is not None:
                val_loss = self.evaluate(val_dataloader)
                self.logger.info(f"Epoch {epoch + 1} - Val Loss: {val_loss:.4f}")

                # LOGIC EARLY STOPPING
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0 # Reset đếm nếu có cải thiện
                    self.save_checkpoint(tag='best') # Lưu model tốt nhất
                    self.logger.info(f"  --> Best model saved with Val Loss: {best_val_loss:.4f}")
                else:
                    patience_counter += 1
                    self.logger.info(f"  --> No improvement for {patience_counter} epochs.")

                # Kiểm tra nếu đạt ngưỡng ngắt sớm
                if patience_counter >= max_patience:
                    self.logger.info("="*30)
                    self.logger.info(f"EARLY STOPPING TRIGGERED tại epoch {epoch + 1}")
                    self.logger.info("="*30)
                    break 
            else:
                # Nếu không có val_set, lưu theo từng epoch như cũ
                self.save_checkpoint(tag=f'epoch_{epoch + 1}')

        self.logger.info("Training complete!")
    
    def _train_epoch(self, dataloader: DataLoader) -> float:
        self.model.train()
        total_loss = 0.0
        progress_bar = tqdm(dataloader, desc=f'Epoch {self.epoch + 1} Training', leave=True)
        
        
        for batch_idx, batch in enumerate(progress_bar):
            input_ids = batch['input_ids'].to(self.device)
            labels = batch['labels'].to(self.device)

            attention_mask = batch['attention_mask'].to(self.device)

            self.optimizer.zero_grad()
            
            _, loss, _ = self.model(input_ids, attention_mask=attention_mask, labels=labels)
            
            loss.backward()
            
            # Optimizer & Scheduler step
            self.optimizer.step()
            self.scheduler.step()

            self.global_step += 1
            total_loss += loss.item()
            
            # Update progress bar with loss
            progress_bar.set_postfix({'loss': f"{loss.item():.4f}"})
            
        avg_loss = total_loss / len(dataloader)
        return avg_loss
