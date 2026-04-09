"""
Dataset builder for pretraining
"""

from typing import Dict, Any
import torch
from torch.utils.data import Dataset
from .registry import META_DATASET


@META_DATASET.register()
class PretrainingDataset(Dataset):
    """
    Dataset for masked language modeling pretraining
    """
    
    def __init__(self, file_path: str, tokenizer, max_seq_len: int = 512, 
                 mlm_probability: float = 0.15):
        """
        Initialize pretraining dataset
        
        Args:
            file_path: Path to text file (one sentence per line)
            tokenizer: Tokenizer instance
            max_seq_len: Maximum sequence length
            mlm_probability: Probability of masking tokens
        """
        self.file_path = file_path
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.mlm_probability = mlm_probability
        
        # Load all texts
        self.texts = self._load_texts()
        
    def _load_texts(self) -> list:
        """Load texts from file"""
        texts = []
        try:
            with open(self.file_path, 'r', encoding='utf-8', errors='ignore') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        texts.append(line)
        except FileNotFoundError:
            print(f"Warning: File not found: {self.file_path}")
        return texts
    
    def __len__(self) -> int:
        return len(self.texts)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a single item"""
        text = self.texts[idx]
        
        # Tokenize
        tokens = self.tokenizer.encode(text)
        
        # Truncate
        if len(tokens) > self.max_seq_len - 2:
            tokens = tokens[:self.max_seq_len - 2]
        
        # Add special tokens [BOS] + tokens + [EOS]
        input_ids = [1] + tokens + [2]
        
        # Pad with [PAD] (id=3)
        if len(input_ids) < self.max_seq_len:
            input_ids += [3] * (self.max_seq_len - len(input_ids))
        
        input_ids = torch.tensor(input_ids, dtype=torch.long)
        
        # Create labels (masked language modeling)
        labels = input_ids.clone()
        
        # Randomly select tokens to mask
        mask_indices = torch.bernoulli(
            torch.full((self.max_seq_len,), self.mlm_probability)
        ).bool()
        
        # Don't mask special tokens and padding
        mask_indices[0] = False          # Don't mask [BOS]
        mask_indices[input_ids == 2] = False  # Don't mask [EOS]
        mask_indices[input_ids == 3] = False  # Don't mask [PAD]
        
        # Apply masking (mask token id is 0)
        input_ids[mask_indices] = 0
        
        return {
            'input_ids': input_ids,
            'labels': labels,
        }


def build_dataset(config: Dict[str, Any], tokenizer, split: str = 'train'):
    """
    Build a dataset from configuration
    
    Args:
        config: Dataset configuration
        tokenizer: Tokenizer instance
        split: Data split ('train', 'val', 'test')
        
    Returns:
        Dataset instance
    """
    dataset_type = config.get('type', 'PretrainingDataset')
    
    if dataset_type == 'PretrainingDataset':
        return PretrainingDataset(
            file_path=config.get('file_path', './processed_data/merged_corpus.txt'),
            tokenizer=tokenizer,
            max_seq_len=config.get('max_seq_len', 512),
            mlm_probability=config.get('mlm_probability', 0.15)
        )
    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}")
