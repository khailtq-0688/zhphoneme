#!/usr/bin/env python3
"""
Simplified Training Example - Following the PinyinBert pattern
This demonstrates the clean training loop pattern
"""

import torch
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm
from pathlib import Path

# Import our components
from configs.config import dict_to_dotdict, load_config_from_yaml
from tokenizer.unigram_tokenizer import UnigramTokenizer
from builders.dataset_builder import build_dataset, collate_fn
from builders.model_builder import build_model
from data_processing.data_processor import VietnameseProcessor

# Configuration
EPOCHS = 5
BS = 512
# Warmup steps will be calculated as 5% of total_steps dynamically

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Load configuration
config_path = "./configs/viwordformer_pretrain_vietnamese.yaml"
config = load_config_from_yaml(config_path)
config = dict_to_dotdict(config)

print(f"Configuration loaded from {config_path}")
print(f"Batch size: {BS}")
print(f"Epochs: {EPOCHS}")
print(f"Learning rate: {config.training.learning_rate}")

# Initialize tokenizer
print("\nInitializing tokenizer...")
tokenizer = UnigramTokenizer()
tokenizer_path = Path('./tokenizers/unigram_tokenizer.model')
if tokenizer_path.exists():
    tokenizer.load(str(tokenizer_path))
    print(f"✓ Tokenizer loaded from {tokenizer_path}")
else:
    print(f"⚠ Tokenizer not found at {tokenizer_path}")

# Initialize dataset
print("\nInitializing dataset...")
dataset = build_dataset(
    config.get('dataset', {}),
    tokenizer,
    split='train'
)
print(f"✓ Dataset loaded: {len(dataset)} samples")

# Create DataLoader
print("\nCreating DataLoader...")
dataloader = DataLoader(
    dataset=dataset,
    batch_size=BS,
    shuffle=True,
    num_workers=4,
    collate_fn=collate_fn,
    pin_memory=True if device == 'cuda' else False
)
print(f"✓ DataLoader created: {len(dataloader)} batches")

# Initialize model
print("\nInitializing model...")
model = build_model(config.get('model', {}), vocab_size=30000)
model.to(device)
model.train()
print(f"✓ Model loaded: ViWordFormer (768d, 12L, 12H)")

# Initialize optimizer
print("\nInitializing optimizer...")
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=config.training.learning_rate,
    weight_decay=config.training.get('weight_decay', 0.01),
    betas=tuple(config.training.get('betas', [0.9, 0.98])),
    eps=config.training.get('eps', 1e-6)
)
print(f"✓ Optimizer: AdamW(lr={config.training.learning_rate}, weight_decay=0.01)")

# Calculate total steps and initialize scheduler
total_steps = len(dataloader) * EPOCHS
warmup_steps = int(total_steps * 0.05)  # 5% of total steps

print(f"\nScheduler configuration:")
print(f"  Total steps: {total_steps}")
print(f"  Warmup steps: {warmup_steps} (5% of total)")

def lr_lambda(current_step):
    """Learning rate schedule with linear warmup and linear decay"""
    if current_step < warmup_steps:
        return float(current_step) / float(max(1, warmup_steps))
    return max(0.0, float(total_steps - current_step) / float(max(1, total_steps - warmup_steps)))

lr_scheduler = LambdaLR(optimizer, lr_lambda)
print(f"✓ Scheduler: LambdaLR with warmup + linear decay")

# Training loop
print("\n" + "="*70)
print("STARTING TRAINING")
print("="*70 + "\n")

for epoch in range(1, EPOCHS + 1):
    total_loss = 0
    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch}/{EPOCHS}")
    
    for batch in progress_bar:
        input_ids = batch['input_ids'].to(device)
        labels = batch['labels'].to(device)
        
        # Forward pass
        optimizer.zero_grad()
        logits, loss, attentions = model(input_ids, labels)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        lr_scheduler.step()
        
        total_loss += loss.item()
        progress_bar.set_postfix({'loss': f"{loss.item():.4f}"})
        
    avg_loss = total_loss / len(dataloader)
    print(f"Epoch {epoch} - Average Loss: {avg_loss:.4f}\n")

print("="*70)
print("Training Complete!")
print("="*70)
