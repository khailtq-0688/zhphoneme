"""
Dataset builder for pretraining
"""

from typing import Dict, Any
import os
import torch
from torch.utils.data import Dataset
from .registry import META_DATASET
from collections import OrderedDict
from torch.nn.utils.rnn import pad_sequence


@META_DATASET.register()
class SubsetDataset(Dataset):
    """
    Dataset for masked language modeling pretraining from subset_*.txt files
    Structure: corpus_dir/subset_0.txt, subset_1.txt, ..., subset_N.txt
    Each file has ~1000 lines (configurable via lines_per_file)
    """
    
    def __init__(self, corpus_dir: str, tokenizer, max_seq_len: int = 512, 
                 mlm_probability: float = 0.15, lines_per_file: int = 1000,
                 max_cache_files: int = 4000, mask_id: int = 4):
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
        self.mask_id = mask_id

        # Dictionary dùng làm Cache để chống I/O bottleneck
        self.max_cache_files = max_cache_files
        self._file_cache = OrderedDict()
        
        # Count total lines across all subset files
        self.total_lines = self._count_total_lines()
        
    def _count_total_lines(self) -> int:
        """Đếm tổng dòng tối ưu: (Số file - 1) * 1000 + số dòng file cuối"""
        files = sorted([f for f in os.listdir(self.corpus_dir) if f.startswith('subset_') and f.endswith('.txt')],
                    key=lambda x: int(x.split('_')[1].split('.')[0]))
        
        if not files:
            return 0
        
        # Giả định các file trước đều đủ lines_per_file (theo logic split của bạn)
        total = (len(files) - 1) * self.lines_per_file
        
        # Chỉ đếm thực tế file cuối cùng
        last_file = os.path.join(self.corpus_dir, files[-1])
        with open(last_file, 'r', encoding='utf-8', errors='ignore') as f:
            last_file_lines = sum(1 for line in f if line.strip())
            
        return total + last_file_lines
    
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
        
        # Truncate (chỉ cắt bớt nếu vượt quá max_seq_len)
        if len(tokens) > self.max_seq_len - 2:
            tokens = tokens[:self.max_seq_len - 2]
            
        # Thêm special tokens [BOS] + tokens + [EOS]
        input_ids = [1] + tokens + [2]
        
        # ❌ XÓA BỎ ĐOẠN STATIC PADDING DƯỚI ĐÂY:
        # if len(input_ids) < self.max_seq_len:
        #     input_ids += [3] * (self.max_seq_len - len(input_ids))
            
        input_ids = torch.tensor(input_ids, dtype=torch.long)
        labels = input_ids.clone()
        
        # ✅ Tạo mask dựa trên ĐỘ DÀI THỰC TẾ của câu hiện tại, không dùng max_seq_len nữa
        seq_len = len(input_ids)
        mask_indices = torch.bernoulli(
            torch.full((seq_len,), self.mlm_probability)
        ).bool()
        
        # Không mask [BOS] và [EOS]
        mask_indices[0] = False 
        mask_indices[-1] = False # Phần tử cuối cùng là EOS
        
        # Apply masking (mask token id là 4)
        input_ids[mask_indices] = self.mask_id
        labels[~mask_indices] = -100
        
        return {
            'input_ids': input_ids,
            'labels': labels,
        }
    
    def _empty_sample(self) -> Dict[str, torch.Tensor]:
        """Trả về sample rỗng chỉ chứa [BOS, EOS] thay vì 512 token PAD"""
        input_ids = torch.tensor([1, 2], dtype=torch.long)  # Chỉ [BOS, EOS]
        labels = torch.tensor([-100, -100], dtype=torch.long)
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
                 mlm_probability: float = 0.15, mask_id: int = 4):
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
        self.mask_id = mask_id
        
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
        input_ids = torch.tensor(input_ids, dtype=torch.long)
        
        labels = input_ids.clone()
        seq_len = len(input_ids)
        
        # Mask động dựa trên seq_len thực tế
        mask_indices = torch.bernoulli(torch.full((seq_len,), self.mlm_probability)).bool()
        mask_indices[0] = False 
        mask_indices[-1] = False
        
        input_ids[mask_indices] = self.mask_id # MASK_ID
        labels[~mask_indices] = -100
        
        return {'input_ids': input_ids, 'labels': labels}


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

    mask_id = 4

    # Check nếu là BPE Tokenizer (HuggingFace)
    if hasattr(tokenizer, 'token_to_id'):
        mask_id = tokenizer.token_to_id('[MASK]') or 4
    # Check nếu là Unigram Tokenizer (SentencePiece wrapper)
    elif hasattr(tokenizer, 'model') and hasattr(tokenizer.model, 'PieceToId'):
        mask_id = tokenizer.model.PieceToId('<mask>')

    dataset_type = config.get('type', 'PretrainingDataset')
    
    if dataset_type == 'SubsetDataset':
        # For corpus with subset_*.txt files structure
        return SubsetDataset(
            corpus_dir=config.get('corpus_dir', './corpus'),
            tokenizer=tokenizer,
            max_seq_len=config.get('max_seq_len', 512),
            mlm_probability=config.get('mlm_probability', 0.15),
            lines_per_file=config.get('lines_per_file', 1000),
            mask_id=mask_id
        )
    elif dataset_type == 'PretrainingDataset':
        # For merged corpus file format
        return PretrainingDataset(
            file_path=config.get('file_path', './processed_data/merged_corpus.txt'),
            tokenizer=tokenizer,
            max_seq_len=config.get('max_seq_len', 512),
            mlm_probability=config.get('mlm_probability', 0.15),
            mask_id=mask_id
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
    PAD_TOKEN_ID = 3 
    
    # Lấy danh sách các tensor động
    input_ids_list = [sample['input_ids'] for sample in batch]
    labels_list = [sample['labels'] for sample in batch]
    
    # Pad động theo câu dài nhất TRONG BATCH
    input_ids = pad_sequence(input_ids_list, batch_first=True, padding_value=PAD_TOKEN_ID)
    labels = pad_sequence(labels_list, batch_first=True, padding_value=-100)
    
    attention_mask = (input_ids != PAD_TOKEN_ID).float()
    
    return {
        'input_ids': input_ids,
        'labels': labels,
        'attention_mask': attention_mask,
    }
