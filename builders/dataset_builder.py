"""
Dataset builder for pretraining
"""

from typing import Dict, Any
import os
import torch
from torch.utils.data import Dataset
from .registry import META_DATASET
from collections import OrderedDict


@META_DATASET.register()
class SubsetDataset(Dataset):
    """
    Dataset for masked language modeling pretraining from subset_*.txt files
    Structure: corpus_dir/subset_0.txt, subset_1.txt, ..., subset_N.txt
    Each file has ~1000 lines (configurable via lines_per_file)
    """
    
    def __init__(self, corpus_dir: str, tokenizer, max_seq_len: int = 512, 
                 mlm_probability: float = 0.15, lines_per_file: int = 1000,
                 max_cache_files: int = 4000):
        """
        Initialize subset dataset
        
        Args:
            corpus_dir: Directory containing subset_*.txt files
            tokenizer: Tokenizer instance
            max_seq_len: Maximum sequence length
            mlm_probability: Probability of masking tokens
            lines_per_file: Lines per subset file (default 1000)
        """
        self.corpus_dir = corpus_dir
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.mlm_probability = mlm_probability
        self.lines_per_file = lines_per_file

        # Dictionary dùng làm Cache để chống I/O bottleneck
        self.max_cache_files = max_cache_files
        self._file_cache = OrderedDict()
        
        # Count total lines across all subset files
        self.total_lines = self._count_total_lines()
        
    def _count_total_lines(self) -> int:
        """Count total lines across all subset files"""
        total = 0
        for filename in sorted(os.listdir(self.corpus_dir)):
            if filename.startswith('subset_') and filename.endswith('.txt'):
                filepath = os.path.join(self.corpus_dir, filename)
                try:
                    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                        total += sum(1 for line in f if line.strip())
                except Exception as e:
                    print(f"Warning: Error reading {filename}: {e}")
        return total
    
    def __len__(self) -> int:
        return self.total_lines
    
    def _get_line_from_file(self, filepath: str, line_idx: int) -> str:
        """Hàm phụ trợ: Lấy dòng text có sử dụng Cache"""
        """Hàm phụ trợ: Lấy dòng text có sử dụng LRU Cache"""
        
        # Nếu file đã có trong cache, di chuyển nó xuống cuối để đánh dấu là "Vừa mới sử dụng"
        if filepath in self._file_cache:
            self._file_cache.move_to_end(filepath)
        else:
            # Nếu cache đầy, xóa phần tử ở đầu (Least Recently Used - Ít sử dụng nhất)
            if len(self._file_cache) >= self.max_cache_files:
                self._file_cache.popitem(last=False)
            
            try:
                with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                    self._file_cache[filepath] = f.readlines()
            except Exception as e:
                print(f"Error reading {filepath}: {e}")
                self._file_cache[filepath] = []
        
        # Trích xuất dòng
        lines = self._file_cache[filepath]
        if line_idx < len(lines):
            return lines[line_idx].strip()
        return ""
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a single item"""
        # Calculate which file and which line
        subset_idx, line_idx = divmod(idx, self.lines_per_file)
        
        # Read line from file
        filepath = os.path.join(self.corpus_dir, f"subset_{subset_idx}.txt")

        text = self._get_line_from_file(filepath, line_idx)
        
        if not text:
            # Return empty sample if text is empty
            return self._empty_sample()
        
        # Tokenize
        encode_result = self.tokenizer.encode(text)

        tokens = encode_result.ids if hasattr(encode_result, 'ids') else encode_result
        
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
        
        # Apply masking (mask token id is 4)
        input_ids[mask_indices] = 4

        labels[~mask_indices] = -100
        
        return {
            'input_ids': input_ids,
            'labels': labels,
        }
    
    def _empty_sample(self) -> Dict[str, torch.Tensor]:
        """Return empty sample filled with padding"""
        input_ids = torch.full((self.max_seq_len,), 3, dtype=torch.long)  # All PAD
        labels = torch.full((self.max_seq_len,), -100, dtype=torch.long)  # Bỏ qua hoàn toàn
        return {
            'input_ids': input_ids,
            'labels': labels,
        }


@META_DATASET.register()
class PretrainingDataset(Dataset):
    """
    Dataset for masked language modeling pretraining
    Supports single merged corpus file format
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
        
        UNK_ID = 0
        BOS_ID = 1
        EOS_ID = 2
        PAD_ID = 3
        MASK_ID = 4  
        
        text = self.texts[idx]
        
        # Tokenize
        tokens = self.tokenizer.encode(text)
        
        # Truncate
        if len(tokens) > self.max_seq_len - 2:
            tokens = tokens[:self.max_seq_len - 2]
        
        # Add special tokens [BOS] + tokens + [EOS]
        input_ids = [BOS_ID] + tokens + [EOS_ID]
        
        # Pad with [PAD]
        if len(input_ids) < self.max_seq_len:
            input_ids += [PAD_ID] * (self.max_seq_len - len(input_ids))
        
        input_ids = torch.tensor(input_ids, dtype=torch.long)
        
        # Create labels (masked language modeling)
        labels = input_ids.clone()
        
        # Randomly select tokens to mask
        mask_indices = torch.bernoulli(
            torch.full((self.max_seq_len,), self.mlm_probability)
        ).bool()
        
        mask_indices[0] = False                   # Don't mask [BOS] ở vị trí đầu
        mask_indices[input_ids == EOS_ID] = False # Don't mask [EOS]
        mask_indices[input_ids == PAD_ID] = False # Don't mask [PAD]
        
        input_ids[mask_indices] = MASK_ID

        labels[~mask_indices] = -100
        
        return {
            'input_ids': input_ids,
            'labels': labels,
        }


def build_dataset(config: Dict[str, Any], tokenizer, split: str = 'train'):
    """
    Build a dataset from configuration
    Supports both subset_*.txt format and merged corpus format
    
    Args:
        config: Dataset configuration
        tokenizer: Tokenizer instance
        split: Data split ('train', 'val', 'test')
        
    Returns:
        Dataset instance
    """
    dataset_type = config.get('type', 'PretrainingDataset')
    
    if dataset_type == 'SubsetDataset':
        # For corpus with subset_*.txt files structure
        return SubsetDataset(
            corpus_dir=config.get('corpus_dir', './corpus'),
            tokenizer=tokenizer,
            max_seq_len=config.get('max_seq_len', 512),
            mlm_probability=config.get('mlm_probability', 0.15),
            lines_per_file=config.get('lines_per_file', 1000)
        )
    elif dataset_type == 'PretrainingDataset':
        # For merged corpus file format
        return PretrainingDataset(
            file_path=config.get('file_path', './processed_data/merged_corpus.txt'),
            tokenizer=tokenizer,
            max_seq_len=config.get('max_seq_len', 512),
            mlm_probability=config.get('mlm_probability', 0.15)
        )
    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}")


def collate_fn(batch):
    """
    Collate function for DataLoader
    Handles variable-length sequences with padding and attention masks
    
    Args:
        batch: List of samples from dataset
        
    Returns:
        Dictionary with padded input_ids, labels, and attention_mask
    """
    PAD_TOKEN_ID = 3  # Padding token id
    
    input_ids = torch.stack([sample['input_ids'] for sample in batch])
    labels = torch.stack([sample['labels'] for sample in batch])
    
    attention_mask = (input_ids != PAD_TOKEN_ID).float()
    
    return {
        'input_ids': input_ids,
        'labels': labels,
        'attention_mask': attention_mask,
    }
