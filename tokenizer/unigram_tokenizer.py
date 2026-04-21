"""
Unigram Tokenizer Builder
Based on SentencePiece Unigram model
"""

from typing import List, Dict, Optional
import json
import os
from collections import Counter
import logging

try:
    import sentencepiece as spm
except ImportError:
    print("Please install sentencepiece: pip install sentencepiece")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class UnigramTokenizer:
    """
    Unigram SentencePiece Tokenizer for multilingual pretraining
    """
    
    def __init__(self, model_prefix: str, vocab_size: int = 30000):
        """
        Initialize Unigram Tokenizer
        
        Args:
            model_prefix: Path prefix for tokenizer model
            vocab_size: Target vocabulary size
        """
        self.model_prefix = model_prefix
        self.vocab_size = vocab_size
        self.model = None
        
    def train(self, corpus_dir: str, vocab_size: int = None):
        """
        Train Unigram tokenizer on corpus
        
        Args:
            training_files: List of file paths for training
            vocab_size: Optional override vocab size
        """
        if vocab_size is None:
            vocab_size = self.vocab_size
            
        logger.info(f"Training Unigram tokenizer with vocab size: {vocab_size}")
        
        def generate_text():
            import os
            for txt_file in os.listdir(corpus_dir):
                with open(os.path.join(corpus_dir, txt_file)) as file:
                    for line in file:
                        yield line
        
        # Train SentencePiece model with Unigram
        spm.SentencePieceTrainer.train(
            sentence_iterator=generate_text(),
            model_prefix=self.model_prefix,
            vocab_size=vocab_size,
            model_type='unigram',
            character_coverage=0.9995,
            normalization_rule_name='identity',
            unk_piece='<unk>',
            bos_piece='<s>',
            eos_piece='</s>',
            pad_piece='<pad>',
            unk_id=0,
            bos_id=1,
            eos_id=2,
            pad_id=3,
            num_threads=os.cpu_count(),
            train_extremely_large_corpus=True,
        )
        
        logger.info(f"Tokenizer saved to {self.model_prefix}")
        self.load(self.model_prefix)
        
    def load(self, model_prefix: str):
        """Load pretrained tokenizer"""
        self.model = spm.SentencePieceProcessor()
        self.model.Load(f"{model_prefix}")
        logger.info(f"Loaded tokenizer from {model_prefix}")
        
    def encode(self, text: str) -> List[int]:
        """Encode text to token ids"""
        if self.model is None:
            raise ValueError("Model not loaded. Call load() first.")
        return self.model.EncodeAsIds(text)
    
    def decode(self, ids: List[int]) -> str:
        """Decode token ids to text"""
        if self.model is None:
            raise ValueError("Model not loaded. Call load() first.")
        return self.model.DecodeIds(ids)
    
    def encode_pieces(self, text: str) -> List[str]:
        """Encode text to token pieces"""
        if self.model is None:
            raise ValueError("Model not loaded. Call load() first.")
        return self.model.EncodeAsPieces(text)
    
    def get_vocab_size(self) -> int:
        """Get vocabulary size"""
        if self.model is None:
            raise ValueError("Model not loaded. Call load() first.")
        return self.model.vocab_size()
    
    def get_piece_size(self) -> int:
        """Get piece size"""
        if self.model is None:
            raise ValueError("Model not loaded. Call load() first.")
        return self.model.piece_size()
    
    def save_vocab(self, output_path: str):
        """Save vocabulary to file"""
        if self.model is None:
            raise ValueError("Model not loaded. Call load() first.")
        
        vocab = {}
        for i in range(self.model.vocab_size()):
            vocab[self.model.id_to_piece(i)] = i
            
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(vocab, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Vocabulary saved to {output_path}")


class TokenizerConfig:
    """Configuration for Unigram Tokenizer"""
    
    def __init__(self):
        self.vocab_size = 30000
        self.character_coverage = 0.9995
        self.model_type = 'unigram'
        self.normalization_rule = 'identity'
        self.unk_piece = '<unk>'
        self.bos_piece = '<s>'
        self.eos_piece = '</s>'
        self.pad_piece = '<pad>'
        
    def to_dict(self) -> Dict:
        """Convert config to dictionary"""
        return {
            'vocab_size': self.vocab_size,
            'character_coverage': self.character_coverage,
            'model_type': self.model_type,
            'normalization_rule': self.normalization_rule,
            'unk_piece': self.unk_piece,
            'bos_piece': self.bos_piece,
            'eos_piece': self.eos_piece,
            'pad_piece': self.pad_piece,
        }
    
    def save(self, path: str):
        """Save config to JSON"""
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, indent=2)


if __name__ == "__main__":
    # Example usage
    config = TokenizerConfig()
    config.vocab_size = 30000
    
    tokenizer = UnigramTokenizer(
        model_prefix="./outputs/unigram_tokenizer",
        vocab_size=config.vocab_size
    )
    
    # Training files would be passed here
    # tokenizer.train(
    #     training_files=["path/to/vietnamese.txt", "path/to/chinese.txt"],
    #     vocab_size=30000
    # )
