"""
ViWordFormer Model for Pretraining
Adapted for multilingual Vietnamese and Chinese (MLM Objective)
"""

import torch
import torch.nn as nn
import math
from typing import Dict

from .attention import ScaledDotProductAttention, PhrasalLexemeAttention

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
    """Positional Encoding"""
    
    def __init__(self, d_model: int, max_len: int = 4096):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=0.1)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        
        self.register_buffer('pe', pe)
        
    def forward(self, features: torch.Tensor) -> torch.Tensor:
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

    def forward(self, inputs: torch.Tensor, attention_mask: torch.Tensor, phrasal_attn: torch.Tensor):
        # Phrasal lexeme attention
        P, phrasal_attn = self.phrasal_lexeme_attn(inputs, attention_mask, phrasal_attn)
        # Self-attention
        self_attn = self.self_attn(inputs, inputs, inputs, attention_mask)

        attn_scores = P * self_attn
        b_s, nq = inputs.shape[:2]
        
        v = self.linear_out(inputs).view(b_s, nq, self.head, self.d_kv).permute(0, 2, 1, 3) 
        features = torch.matmul(attn_scores, v).permute(0, 2, 1, 3).contiguous().view(b_s, nq, self.head * self.d_kv) 
        
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

    def forward(self, inputs: torch.Tensor, attention_mask: torch.Tensor):
        self_attns = []
        phrasal_attns = []
        attn_scores = []

        phrasal_attn = 0.
        features = inputs
        
        for layer in self.layers:
            features, self_attn, phrasal_attn, attn_score = layer(features, attention_mask, phrasal_attn)
            self_attns.append(self_attn)
            phrasal_attns.append(phrasal_attn)
            attn_scores.append(attn_score)

        return features, (self_attns, phrasal_attns, attn_scores)

class ViWordFormer(nn.Module):
    """Model for MLM"""
    
    def __init__(self, vocab_size: int, d_model: int = 768, nlayers: int = 12, 
                 head: int = 12, d_q: int = 64, d_kv: int = 64, d_ff: int = 3072, 
                 dropout: float = 0.1, pad_idx: int = 0, max_seq_len: int = 4096,
                 label_smoothing: float = 0.0): 
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
            nlayers=nlayers, head=head, d_model=d_model,
            d_q=d_q, d_kv=d_kv, d_ff=d_ff, dropout=dropout
        )

        self.proj_vocab = nn.Linear(
            in_features=d_model,
            out_features=vocab_size
        )
        self.dropout = nn.Dropout(dropout)
        
        self.loss = nn.CrossEntropyLoss(ignore_index=-100, label_smoothing=label_smoothing)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor = None, labels: torch.Tensor = None):
        if attention_mask is not None:
            padding_mask = attention_mask.to(input_ids.device)
        else:
            padding_mask = (input_ids != self.pad_idx).long().to(input_ids.device)

        features = self.embedding(input_ids)
        features = self.pe(features)
        features = self.norm(features)

        features, attentions = self.encoder(features, padding_mask)
        
        logits = self.proj_vocab(features) # (Batch_Size, Seq_Len, Vocab_Size)

        loss = None
        if labels is not None:
            active_logits = logits.view(-1, self.vocab_size)
            active_labels = labels.view(-1)
            
            loss = self.loss(active_logits, active_labels)

        return logits, loss, attentions
    
    def get_embedding_weight(self) -> torch.Tensor:
        return self.embedding.weight
    
    def get_config(self) -> Dict:
        return {
            'vocab_size': self.vocab_size,
            'd_model': self.d_model,
            'pad_idx': self.pad_idx,
        }