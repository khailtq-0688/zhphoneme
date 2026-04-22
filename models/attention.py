"""
Attention Modules for ViWordFormer
Based on Scaled Dot-Product Attention and Phrasal Lexeme Attention
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class ScaledDotProductAttention(nn.Module):
    """Scaled Dot-Product Attention Mechanism"""
    
    def __init__(self, head: int, d_model: int, d_q: int, d_kv: int):
        super(ScaledDotProductAttention, self).__init__()

        self.d_model = d_model
        self.d_q = d_q
        self.d_kv = d_kv
        self.head = head

        self.fc_q = nn.Linear(d_model, head * d_q)
        self.fc_k = nn.Linear(d_model, head * d_kv)
        self.fc_v = nn.Linear(d_model, head * d_kv)

        self.init_weights()

    def init_weights(self):
        """Initialize weights"""
        nn.init.xavier_uniform_(self.fc_q.weight)
        nn.init.xavier_uniform_(self.fc_k.weight)
        nn.init.xavier_uniform_(self.fc_v.weight)
        nn.init.constant_(self.fc_q.bias, 0)
        nn.init.constant_(self.fc_k.bias, 0)
        nn.init.constant_(self.fc_v.bias, 0)

    def forward(self, queries, keys, values, attention_mask=None, **kwargs):
        """
        Forward pass
        
        Args:
            queries: (b_s, nq, d_model)
            keys: (b_s, nk, d_model)
            values: (b_s, nk, d_model)
            attention_mask: (b_s, nk) or None
        """
        b_s, nq = queries.shape[:2]
        nk = keys.shape[1]
        
        q = self.fc_q(queries).view(b_s, nq, self.head, self.d_q).permute(0, 2, 1, 3)   # (b_s, h, nq, d_q)
        k = self.fc_k(keys).view(b_s, nk, self.head, self.d_kv).permute(0, 2, 3, 1)     # (b_s, h, d_kv, nk)
        v = self.fc_v(values).view(b_s, nk, self.head, self.d_kv).permute(0, 2, 1, 3)   # (b_s, h, nk, d_kv)

        att = torch.matmul(q, k) / np.sqrt(self.d_kv)  # (b_s, h, nq, nk)
        
        if attention_mask is not None:
            attention_mask = attention_mask.unsqueeze(1).unsqueeze(1)
            att = att.masked_fill(attention_mask == 0, -1e9)
        
        att = torch.softmax(att, dim=-1)

        return att


class PhrasalLexemeAttention(nn.Module):
    """Phrasal Lexeme Attention for capturing word-level and morpheme-level structure"""
    
    def __init__(self, head: int, d_model: int, d_q: int, d_kv: int):
        super().__init__()
        
        self.d_model = d_model
        self.head = head
        self.d_q = d_q
        self.d_kv = d_kv

        self.linear_query = nn.Linear(d_model, head * d_q)
        self.linear_key = nn.Linear(d_model, head * d_kv)

    def forward(self, context, attention_mask=None, prior_attn=0, **kwargs):
        """
        Forward pass
        
        Args:
            context: (bs, seq_len, d_model)
            attention_mask: (bs, seq_len)
            prior_attn: float or (bs, head, seq_len, seq_len)
        """
        bs, seq_len = context.size()[:2]
        device = context.device

        # Create attention masks for different attention patterns
        # Only pay attention to tokens afterward
        after_attention_mask = torch.diag(torch.ones(seq_len - 1, dtype=torch.int32, device=device), 1)
        # Only pay attention to the token itself
        self_attention_mask = torch.diag(torch.ones(seq_len, dtype=torch.int32, device=device))
        # Only pay attention to previous tokens
        prev_attention_mask = torch.diag(torch.ones(seq_len - 1, dtype=torch.int32, device=device), -1)
        # For computing P_{ij}
        summing_operator = torch.triu(torch.ones(seq_len, seq_len, dtype=torch.float32, device=device))

        if attention_mask is not None:
            attention_mask = attention_mask.unsqueeze(1).unsqueeze(1)
            attention_mask = (attention_mask.bool() & (after_attention_mask.bool() | prev_attention_mask.bool())).to(attention_mask.dtype)
        else:
            attention_mask = after_attention_mask + prev_attention_mask
            attention_mask = attention_mask.unsqueeze(0).unsqueeze(1)

        key = self.linear_key(context).reshape((bs, seq_len, self.head, self.d_q)).permute((0, 2, 1, 3))  # (bs, head, seq_len, d_q)
        query = self.linear_query(context).reshape((bs, seq_len, self.head, self.d_q)).permute((0, 2, 1, 3))  # (bs, head, seq_len, d_q)
        
        phrasal_scores = torch.matmul(query, key.transpose(-2, -1)) / self.d_model  # (bs, head, seq_len, seq_len)
        
        phrasal_scores = phrasal_scores.masked_fill(attention_mask == 0, -1e9)
        phrasal_scores = F.softmax(phrasal_scores, dim=-1)
        
        # Phrasal attention - attending to words
        phrasal_attn = torch.sqrt(phrasal_scores * phrasal_scores.transpose(-2, -1) + 1e-9)
        # Co-text module - forming phrasal lexemes
        phrasal_attn = prior_attn + (1 - prior_attn) * phrasal_attn

        # Compute P_{ij}
        p = torch.log(phrasal_attn + 1e-9).masked_fill(after_attention_mask == 0, 0).matmul(summing_operator)
        attn = summing_operator.matmul(p).exp().masked_fill((summing_operator.int() - self_attention_mask) == 0, 0)
        
        # Fill upper triangle and apply residual
        attn = attn + attn.transpose(-2, -1) + phrasal_attn.masked_fill(self_attention_mask == 0, 1e-9)
        
        return attn, phrasal_attn
