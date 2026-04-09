"""
ViWordFormer Model for Pretraining
Adapted for multilingual Vietnamese and Chinese
"""

import torch
import torch.nn as nn
import math
from typing import Tuple, Dict, List

from .attention import ScaledDotProductAttention, PhrasalLexemeAttention


def generate_padding_mask(sequences: torch.Tensor, padding_value: int = 0) -> torch.Tensor:
    """
    Generate padding mask for sequences
    
    Args:
        sequences: (bs, seq_len) or (bs, seq_len, dim)
        padding_value: The padding value
        
    Returns:
        mask: (bs, seq_len)
    """
    if len(sequences.shape) == 2:  # (bs, seq_len)
        __seq = sequences.unsqueeze(dim=-1)  # (bs, seq_len, 1)
    else:
        __seq = sequences

    mask = (torch.sum(__seq, dim=-1) == (padding_value * __seq.shape[-1])).long()  # (bs, seq_len)
    return mask


class PositionwiseFeedForward(nn.Module):
    """Position-wise Feed Forward Network"""
    
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.proj_dff = nn.Linear(d_model, d_ff)
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.proj_dmodel = nn.Linear(d_ff, d_model)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Forward pass"""
        features = self.gelu(self.proj_dff(features))
        features = self.dropout(features)
        features = self.proj_dmodel(features)
        return features


class PositionalEncoding(nn.Module):
    """Positional Encoding (Sinusoidal)"""
    
    def __init__(self, d_model: int, max_len: int = 512):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=0.1)
        
        # Compute the positional encodings once in log space
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        
        self.register_buffer('pe', pe)
        
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Add positional encoding to features"""
        pe = self.pe[:, :features.size(1)]
        pe = pe.expand(features.size(0), -1, -1)
        features = features + pe
        return self.dropout(features)


class PhrasalLexemeEncoderLayer(nn.Module):
    """Single layer of Phrasal Lexeme Encoder"""
    
    def __init__(self, head: int, d_model: int, d_q: int, d_kv: int, d_ff: int):
        super().__init__()

        self.head = head
        self.d_q = d_q
        self.d_kv = d_kv

        self.self_attn = ScaledDotProductAttention(head, d_model, d_q, d_kv)
        self.phrasal_lexeme_attn = PhrasalLexemeAttention(head, d_model, d_q, d_kv)
        self.linear_out = nn.Linear(head * d_kv, d_model)
        
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff)
        self.norm_1 = nn.LayerNorm(d_model)
        self.norm_2 = nn.LayerNorm(d_model)

    def forward(
        self, 
        inputs: torch.Tensor, 
        attention_mask: torch.Tensor, 
        phrasal_attn: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass
        
        Args:
            inputs: (bs, nq, d_model)
            attention_mask: (bs, nq)
            phrasal_attn: (bs, head, nq, nq) - Phrasal Lexeme Attention from previous layers
            
        Returns:
            output, self_attn, phrasal_attn, combined_attn
        """
        # Performing phrasal lexeme attention
        P, phrasal_attn = self.phrasal_lexeme_attn(inputs, attention_mask, phrasal_attn)
        # Performing self-attention
        self_attn = self.self_attn(inputs, inputs, inputs, attention_mask)

        attn_scores = P * self_attn
        b_s, nq = inputs.shape[:2]
        
        v = self.linear_out(inputs).view(b_s, nq, self.head, self.d_kv).permute(0, 2, 1, 3)  # (b_s, h, nq, d_kv)
        features = torch.matmul(attn_scores, v).permute(0, 2, 1, 3).contiguous().view(b_s, nq, self.head * self.d_kv)  # (b_s, nq, h*d_kv)
        
        features = self.norm_1(features + inputs)
        out = self.norm_2(features + self.feed_forward(features))

        return out, self_attn, phrasal_attn, attn_scores


class PhrasalLexemeEncoder(nn.Module):
    """Phrasal Lexeme Encoder - Stack of encoder layers"""
    
    def __init__(self, nlayers: int, head: int, d_model: int, d_q: int, d_kv: int, d_ff: int, dropout: float = 0.1):
        super().__init__()

        self.layers = nn.ModuleList([
            PhrasalLexemeEncoderLayer(head, d_model, d_q, d_kv, d_ff)
            for _ in range(nlayers)
        ])

    def forward(
        self, 
        inputs: torch.Tensor, 
        attention_mask: torch.Tensor
    ) -> Tuple[torch.Tensor, Tuple[List, List, List]]:
        """
        Forward pass
        
        Args:
            inputs: (bs, nq, d_model)
            attention_mask: (bs, nq)
            
        Returns:
            features, (self_attns, phrasal_attns, attn_scores)
        """
        self_attns = []
        phrasal_attns = []
        attn_scores = []

        # Initially phrasal lexeme attention scores are 0
        phrasal_attn = 0.
        features = inputs
        
        for layer in self.layers:
            features, self_attn, phrasal_attn, attn_score = layer(features, attention_mask, phrasal_attn)
            self_attns.append(self_attn)
            phrasal_attns.append(phrasal_attn)
            attn_scores.append(attn_score)

        return features, (self_attns, phrasal_attns, attn_scores)


class ViWordFormer(nn.Module):
    """ViWordFormer Model for Multilingual Pretraining"""
    
    def __init__(self, vocab_size: int, d_model: int = 768, nlayers: int = 12, 
                 head: int = 12, d_q: int = 64, d_kv: int = 64, d_ff: int = 3072, 
                 dropout: float = 0.1, pad_idx: int = 0, max_seq_len: int = 512,
                 label_smoothing: float = 0.1):
        """
        Initialize ViWordFormer
        
        Args:
            vocab_size: Size of vocabulary
            d_model: Model dimension
            nlayers: Number of encoder layers
            head: Number of attention heads
            d_q: Query dimension per head
            d_kv: Key/Value dimension per head
            d_ff: Feed-forward dimension
            dropout: Dropout rate
            pad_idx: Padding index
            max_seq_len: Maximum sequence length
            label_smoothing: Label smoothing factor
        """
        super().__init__()

        self.pad_idx = pad_idx
        self.d_model = d_model
        self.vocab_size = vocab_size

        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=d_model,
            padding_idx=pad_idx
        )
        self.pe = PositionalEncoding(d_model=d_model, max_len=max_seq_len)
        self.norm = nn.LayerNorm(d_model)

        self.encoder = PhrasalLexemeEncoder(
            nlayers=nlayers,
            head=head,
            d_model=d_model,
            d_q=d_q,
            d_kv=d_kv,
            d_ff=d_ff,
            dropout=dropout
        )

        self.proj_vocab = nn.Linear(
            in_features=d_model,
            out_features=vocab_size
        )
        self.dropout = nn.Dropout(dropout)
        self.loss = nn.CrossEntropyLoss(ignore_index=self.pad_idx, label_smoothing=label_smoothing)

    def forward(self, input_ids: torch.Tensor, labels: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor, Tuple]:
        """
        Forward pass for pretraining
        
        Args:
            input_ids: (bs, seq_len)
            labels: (bs, seq_len) or None
            
        Returns:
            logits, loss, attentions
        """
        padding_mask = generate_padding_mask(input_ids, padding_value=self.pad_idx).to(input_ids.device)

        features = self.embedding(input_ids)
        features = self.pe(features)
        features = self.norm(features)

        features, attentions = self.encoder(features, padding_mask)
        
        # Use CLS token (first token) for sentence-level representation
        cls_features = features[:, 0]
        logits = self.proj_vocab(cls_features)

        loss = None
        if labels is not None:
            loss = self.loss(logits, labels.squeeze(-1))

        return logits, loss, attentions
    
    def get_embedding_weight(self) -> torch.Tensor:
        """Get embedding weights"""
        return self.embedding.weight
    
    def get_config(self) -> Dict:
        """Get model configuration"""
        return {
            'vocab_size': self.vocab_size,
            'd_model': self.d_model,
            'pad_idx': self.pad_idx,
        }
