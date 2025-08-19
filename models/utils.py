import torch
from torch.nn import functional as F
import math
from torch import nn
from vocabs.vocab import Vocab


def generate_padding_mask(seq, pad_token_id=0):
    if seq.ndim == 3:
        """(bs, seq_len, 3) -> padding = [[[pad_token_id, pad_token_id, pad_token_id]]]"""
        pad_token = (
            torch.Tensor([pad_token_id, pad_token_id, pad_token_id])
            .unsqueeze(0)
            .unsqueeze(0)
            .to(seq.device)
        )
        return (seq == pad_token).all(dim=-1)
    return seq == pad_token_id


def new_generate_padding_mask(seq, pad_token_id=0):
    # TODO: update `generate_padding_mask` instead of create new ones.
    if seq.ndim == 3 and seq.shape[-1] == 5:
        """(batch_size, seq_len, 5) -> padding mask"""
        pad_token = torch.full((1, 1, 5), pad_token_id, device=seq.device)
        return (seq == pad_token).all(dim=-1)

    return seq == pad_token_id


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_length=5000, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_seq_length, d_model)
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


class ViWordEmbedder(nn.Module):
    def __init__(self, config, vocab: Vocab):
        super().__init__()
        embed_dim = config.embedder.embed_dim
        self.num_layer = config.embedder.num_layer
        self.device = config.model.device

        self.embedding = nn.Embedding(
            num_embeddings=vocab.total_tokens,
            embedding_dim=embed_dim,
            padding_idx=vocab.pad_idx
        )
        self.proj = nn.Linear(
            in_features=5*embed_dim,
            out_features=embed_dim
        )

    def forward(self, x):
        """
        (bs, seq_len, 5)
        """

        bs, seq_len = x.shape[:2]
        embedded = self.embedding(x)  # (bs, seq_len, 5, d_model)
        embedded = embedded.reshape(bs, seq_len, -1) # (bs, seq_len, 5*d_model)
        embedded = F.relu(self.proj(embedded))

        return embedded

