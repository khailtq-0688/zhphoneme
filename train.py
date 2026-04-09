"""
Main Pretraining Script for ViWordFormer
Trains on Vietnamese (Curated) and Chinese (CommonCrawl) datasets
"""

import os
import sys
import json
import logging
from pathlib import Path
from typing import Dict, Tuple, Optional
from datetime import datetime

import torch
import torch.nn as nn
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler
from tqdm import tqdm

# Import from local modules
from configs.config import PretrainingConfig, get_default_config
from tokenizer.unigram_tokenizer import UnigramTokenizer
from models.viwordformer import ViWordFormer
from data_processing.data_processor import VietnameseProcessor, ChineseProcessor, DataMerger

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class PretrainingDataset(Dataset):
    """Dataset for pretraining with MLM"""
    
    def __init__(self, file_path: str, tokenizer: UnigramTokenizer, 
                 max_seq_len: int = 512, mlm_probability: float = 0.15):
        """
        Initialize pretraining dataset
        
        Args:
            file_path: Path to text file
            tokenizer: Tokenizer instance
            max_seq_len: Maximum sequence length
            mlm_probability: Probability of masking a token
        """
        self.file_path = file_path
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.mlm_probability = mlm_probability
        
        # Load all data
        self.texts = self._load_texts()
        logger.info(f"Loaded {len(self.texts)} documents from {file_path}")
        
    def _load_texts(self) -> list:
        """Load texts from file"""
        texts = []
        with open(self.file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    texts.append(line)
        return texts
    
    def __len__(self) -> int:
        return len(self.texts)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single item
        
        Returns:
            input_ids, labels
        """
        text = self.texts[idx]
        
        # Tokenize
        tokens = self.tokenizer.encode(text)
        
        # Truncate to max length
        if len(tokens) > self.max_seq_len - 2:
            tokens = tokens[:self.max_seq_len - 2]
        
        # Add special tokens
        input_ids = [1] + tokens + [2]  # [BOS] + tokens + [EOS]
        
        # Pad
        if len(input_ids) < self.max_seq_len:
            input_ids += [3] * (self.max_seq_len - len(input_ids))  # PAD
        
        input_ids = torch.tensor(input_ids, dtype=torch.long)
        
        # Create labels (masked language modeling)
        labels = input_ids.clone()
        
        # Randomly select tokens to mask (15% probability)
        mask_indices = torch.bernoulli(
            torch.full((self.max_seq_len,), self.mlm_probability)
        ).bool()
        
        # Don't mask special tokens
        mask_indices[0] = False  # Don't mask [BOS]
        mask_indices[-1] = False  # Don't mask [EOS]
        mask_indices[input_ids == 3] = False  # Don't mask [PAD]
        
        # Apply masking
        input_ids[mask_indices] = 0  # Mask token
        
        return {
            'input_ids': input_ids,
            'labels': labels,
        }


class Trainer:
    """Trainer for ViWordFormer pretraining"""
    
    def __init__(self, config: PretrainingConfig):
        """
        Initialize trainer
        
        Args:
            config: PretrainingConfig instance
        """
        self.config = config
        self.device = torch.device(config.training.device if torch.cuda.is_available() else 'cpu')
        
        # Create output directories
        Path(config.training.output_dir).mkdir(parents=True, exist_ok=True)
        Path(config.training.checkpoint_dir).mkdir(parents=True, exist_ok=True)
        
        # Initialize model
        self.model = ViWordFormer(
            vocab_size=config.model.vocab_size,
            d_model=config.model.d_model,
            nlayers=config.model.nlayers,
            head=config.model.head,
            d_q=config.model.d_q,
            d_kv=config.model.d_kv,
            d_ff=config.model.d_ff,
            dropout=config.model.dropout,
            pad_idx=config.model.pad_idx,
            max_seq_len=config.model.max_seq_len,
            label_smoothing=config.model.label_smoothing,
        ).to(self.device)
        
        logger.info(f"Model initialized on device: {self.device}")
        logger.info(f"Total parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
        # Initialize optimizer
        if config.training.optimizer.lower() == 'adamw':
            self.optimizer = AdamW(
                self.model.parameters(),
                lr=config.training.learning_rate,
                weight_decay=config.training.weight_decay
            )
        else:
            self.optimizer = Adam(
                self.model.parameters(),
                lr=config.training.learning_rate
            )
        
        # Initialize scheduler
        self.scheduler = None
        
        # Training state
        self.global_step = 0
        self.best_loss = float('inf')
        
    def train(self, train_dataloader: DataLoader, eval_dataloader: Optional[DataLoader] = None):
        """
        Training loop
        
        Args:
            train_dataloader: Training dataloader
            eval_dataloader: Evaluation dataloader (optional)
        """
        logger.info("Starting training...")
        
        num_epochs = self.config.training.num_epochs
        total_steps = len(train_dataloader) * num_epochs
        
        # Initialize scheduler
        if self.config.training.scheduler.lower() == 'linear':
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
        
        for epoch in range(num_epochs):
            logger.info(f"Epoch {epoch + 1}/{num_epochs}")
            
            train_loss = self.train_epoch(train_dataloader)
            logger.info(f"Train loss: {train_loss:.4f}")
            
            if eval_dataloader is not None:
                eval_loss = self.evaluate(eval_dataloader)
                logger.info(f"Eval loss: {eval_loss:.4f}")
                
                if eval_loss < self.best_loss:
                    self.best_loss = eval_loss
                    self.save_checkpoint(tag='best')
            
            self.save_checkpoint(tag=f'epoch_{epoch + 1}')
    
    def train_epoch(self, dataloader: DataLoader) -> float:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0.0
        
        progress_bar = tqdm(dataloader, desc='Training')
        
        for batch_idx, batch in enumerate(progress_bar):
            input_ids = batch['input_ids'].to(self.device)
            labels = batch['labels'].to(self.device)
            
            # Forward pass
            logits, loss, _ = self.model(input_ids, labels)
            
            # Backward pass
            loss.backward()
            
            if (batch_idx + 1) % self.config.training.gradient_accumulation_steps == 0:
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.training.gradient_clip_val
                )
                
                # Optimizer step
                self.optimizer.step()
                self.optimizer.zero_grad()
                
                if self.scheduler is not None:
                    self.scheduler.step()
                
                self.global_step += 1
                
                # Logging
                if self.global_step % self.config.training.log_steps == 0:
                    avg_loss = total_loss / (batch_idx + 1)
                    progress_bar.set_postfix({'loss': avg_loss})
                    logger.info(f"Step {self.global_step}: loss={avg_loss:.4f}")
            
            total_loss += loss.item()
        
        return total_loss / len(dataloader)
    
    def evaluate(self, dataloader: DataLoader) -> float:
        """Evaluate model"""
        self.model.eval()
        total_loss = 0.0
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc='Evaluating'):
                input_ids = batch['input_ids'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                _, loss, _ = self.model(input_ids, labels)
                total_loss += loss.item()
        
        return total_loss / len(dataloader)
    
    def save_checkpoint(self, tag: str = 'latest'):
        """Save model checkpoint"""
        checkpoint_dir = Path(self.config.training.checkpoint_dir) / tag
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        model_path = checkpoint_dir / 'model.pt'
        optimizer_path = checkpoint_dir / 'optimizer.pt'
        config_path = checkpoint_dir / 'config.json'
        
        torch.save(self.model.state_dict(), model_path)
        torch.save(self.optimizer.state_dict(), optimizer_path)
        self.config.save(str(config_path))
        
        logger.info(f"Checkpoint saved to {checkpoint_dir}")


def main():
    """Main training pipeline"""
    
    # Load or create configuration
    config_path = "config.json"
    if os.path.exists(config_path):
        config = PretrainingConfig.load(config_path)
        logger.info(f"Loaded config from {config_path}")
    else:
        config = get_default_config()
        config.save(config_path)
        logger.info(f"Created and saved default config to {config_path}")
    
    # Initialize tokenizer
    logger.info("Initializing tokenizer...")
    tokenizer = UnigramTokenizer(
        model_prefix="./outputs/unigram_tokenizer",
        vocab_size=config.tokenizer.vocab_size
    )
    
    # For this example, we assume tokenizer is already trained
    # If not, uncomment below:
    # tokenizer.train(
    #     training_files=[
    #         config.data.processed_data_dir + "/vietnamese_processed.txt",
    #         config.data.processed_data_dir + "/chinese_processed.txt",
    #     ],
    #     vocab_size=config.tokenizer.vocab_size
    # )
    
    # Load tokenizer (assuming it exists)
    try:
        tokenizer.load("./outputs/unigram_tokenizer")
    except Exception as e:
        logger.warning(f"Could not load tokenizer: {e}")
        logger.info("Please train tokenizer first using tokenizer/unigram_tokenizer.py")
        return
    
    # Create datasets
    logger.info("Creating datasets...")
    merged_corpus_path = Path(config.data.processed_data_dir) / "merged_corpus.txt"
    
    if not merged_corpus_path.exists():
        logger.warning(f"{merged_corpus_path} not found. Running data processors...")
        
        # Process Vietnamese data
        vi_processor = VietnameseProcessor(
            input_dir=config.data.vietnamese_data_path,
            output_dir=config.data.processed_data_dir
        )
        vi_processor.process()
        
        # Process Chinese data
        zh_processor = ChineseProcessor(
            input_dir=config.data.chinese_data_path,
            output_dir=config.data.processed_data_dir
        )
        zh_processor.process()
        
        # Merge datasets
        merger = DataMerger(output_dir=config.data.processed_data_dir)
        merger.merge_all()
    
    # Create dataset
    dataset = PretrainingDataset(
        file_path=str(merged_corpus_path),
        tokenizer=tokenizer,
        max_seq_len=config.model.max_seq_len,
        mlm_probability=0.15
    )
    
    # Create dataloaders
    train_dataloader = DataLoader(
        dataset,
        batch_size=config.training.batch_size,
        sampler=RandomSampler(dataset),
        num_workers=config.training.num_workers,
    )
    
    # Initialize trainer
    trainer = Trainer(config)
    
    # Train
    trainer.train(train_dataloader)
    
    logger.info("Training complete!")


if __name__ == "__main__":
    main()
