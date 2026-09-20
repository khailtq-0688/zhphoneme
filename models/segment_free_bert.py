from functools import lru_cache
from typing import NamedTuple

import torch
import torch.nn.functional as F
from torch import nn
from transformers import PreTrainedModel
from transformers.models.roberta.modeling_roberta import (
    RobertaEmbeddings,
    RobertaIntermediate,
    RobertaLMHead,
    RobertaOutput,
    RobertaSelfAttention,
    RobertaSelfOutput,
)
from transformers.models.xlm_roberta import XLMRobertaConfig, XLMRobertaForMaskedLM

from configs.segment_free_bert_config import SegmentFreeBertConfig

PRETRAINED_MODEL_NAME = "xlm-roberta-base"

# XLM-RoBERTa-base's own position-embedding capacity (512 usable positions + 2 for its
# padding-offset scheme) and LayerNorm epsilon, kept in step with the checkpoint's actual
# architecture so every non-vocab weight lines up shape-for-shape when transferred below.
PRETRAINED_MAX_POSITION_EMBEDDINGS = 514
PRETRAINED_LAYER_NORM_EPS = 1e-5

# Backbone tensors whose shape depends on SegmentFreeBERT's own vocabulary rather than on
# XLM-RoBERTa-base's, so they stay randomly initialized instead of being copied over. The
# input/output embeddings are kept untied here (see SegmentFreeBert below), so there is no
# tied "decoder.weight" sharing get_input_embeddings() to worry about.
VOCAB_DEPENDENT_KEYS = {
    "roberta.embeddings.word_embeddings.weight",
    "lm_head.decoder.weight",
    "lm_head.decoder.bias",
    "lm_head.bias",
}

def build_backbone_config(config: SegmentFreeBertConfig) -> XLMRobertaConfig:
    return XLMRobertaConfig(
        vocab_size=config.vocab_size,
        hidden_size=config.hidden_size,
        num_hidden_layers=config.num_hidden_layers,
        num_attention_heads=config.num_attention_heads,
        intermediate_size=config.intermediate_size,
        hidden_act=config.hidden_act,
        hidden_dropout_prob=config.hidden_dropout_prob,
        attention_probs_dropout_prob=config.attention_probs_dropout_prob,
        max_position_embeddings=PRETRAINED_MAX_POSITION_EMBEDDINGS,
        type_vocab_size=config.type_vocab_size,
        initializer_range=config.initializer_range,
        layer_norm_eps=PRETRAINED_LAYER_NORM_EPS,
        position_embedding_type=config.position_embedding_type,
        pad_token_id=config.pad_token_id,
        bos_token_id=config.cls_token_id,
        mask_token_id=config.mask_token_id,
    )

class GroupAttentionMasks(NamedTuple):
    """Constant, data-independent tensors GroupAttention needs, all shaped only by
    seq_len. Building these is O(seq_len^2) (or O(seq_len) for the band indices), so
    they are computed once per forward pass (see TreeEncoder.forward) and shared across
    every layer instead of being rebuilt inside each layer's GroupAttention, and the
    per-(seq_len, device) result is cached across forward passes too since seq_len is
    often the same across consecutive batches (e.g. once padded to max_length)."""

    neighbor_band: torch.Tensor  # (L, L) bool, True at (i, i-1) and (i, i+1)
    strict_upper: torch.Tensor  # (L, L) bool, True where q > p
    eye: torch.Tensor  # (L, L) bool identity
    band_i: torch.Tensor  # (L-1,) long, 0..L-2
    band_j: torch.Tensor  # (L-1,) long, 1..L-1

@lru_cache(maxsize=8)
def _build_group_attention_masks(seq_len: int, device: torch.device) -> GroupAttentionMasks:
    band_i = torch.arange(seq_len - 1, device=device)
    band_j = band_i + 1
    superdiag = torch.zeros(seq_len, seq_len, dtype=torch.bool, device=device)
    superdiag[band_i, band_j] = True
    neighbor_band = superdiag | superdiag.t()
    strict_upper = torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool, device=device), diagonal=1)
    eye = torch.eye(seq_len, dtype=torch.bool, device=device)
    return GroupAttentionMasks(neighbor_band, strict_upper, eye, band_i, band_j)

class GroupAttention(nn.Module):
    """The constituent-prior module from "Tree Transformer: Integrating Tree Structures
    into Self-Attention" (Wang, Lee & Chen, 2019: https://arxiv.org/abs/1909.06639),
    ported from the authors' reference implementation
    (https://github.com/yaushian/Tree-Transformer/blob/master/attention.py).

    For every pair of positions it estimates how likely they belong to the same
    constituent. Each layer starts from a small "neighbor attention" (only adjacent
    positions may attend to each other) and folds in `prior` -- the previous layer's
    full pairwise estimate -- so the induced grouping is refined layer by layer without
    ever needing a gold parse tree.
    """

    def __init__(self, hidden_size):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_size)
        self.linear_key = nn.Linear(hidden_size, hidden_size)
        self.linear_query = nn.Linear(hidden_size, hidden_size)
        # The reference implementation scales by this fixed constant rather than
        # sqrt(hidden_size); kept as-is to stay faithful to the original mechanism.
        self.scale = 256.0

    def forward(self, hidden_states: torch.Tensor, valid_mask: torch.Tensor, prior, masks: GroupAttentionMasks):
        bs, seq_len, _ = hidden_states.shape

        x = self.norm(hidden_states)
        key = self.linear_key(x)
        query = self.linear_query(x)
        scores = torch.matmul(query, key.transpose(-2, -1)) / self.scale

        key_valid = valid_mask.unsqueeze(1).expand(-1, seq_len, -1)
        neighbor_mask = masks.neighbor_band.unsqueeze(0) & key_valid

        scores = scores.masked_fill(~neighbor_mask, torch.finfo(scores.dtype).min)
        neighbor_attn = F.softmax(scores, dim=-1)
        # symmetrize: combine "i attends to i+1" and "i+1 attends to i" into one
        # undirected association strength for the (i, i+1) pair
        neighbor_attn = torch.sqrt(neighbor_attn * neighbor_attn.transpose(-2, -1) + 1e-9)
        neighbor_attn = prior + (1.0 - prior) * neighbor_attn

        # group_prob[p, q] (p < q) = prod_{k=p}^{q-1} neighbor_attn[k, k+1], the
        # probability that the whole span (p, q) is one constituent. That product is a
        # range-product over the superdiagonal, i.e. exp(prefix_sum[q] - prefix_sum[p])
        # of its log -- an O(seq_len) cumsum plus an O(seq_len^2) broadcast, instead of
        # the O(seq_len^3) pair of dense triangular matmuls this originally used.
        log_superdiag = torch.log(neighbor_attn[:, masks.band_i, masks.band_j] + 1e-9)  # (bs, L-1)
        prefix_sum = F.pad(torch.cumsum(log_superdiag, dim=-1), (1, 0))  # (bs, L), prefix_sum[0] = 0
        pairwise = prefix_sum.unsqueeze(1) - prefix_sum.unsqueeze(2)  # pairwise[b,p,q] = prefix_sum[q]-prefix_sum[p]
        group_prob = torch.exp(pairwise).masked_fill(~masks.strict_upper, 0.0)
        group_prob = (
            group_prob
            + group_prob.transpose(-2, -1)
            + neighbor_attn.masked_fill(~masks.eye, 1e-9)
        )

        return group_prob, neighbor_attn

class TreeSelfAttention(RobertaSelfAttention):
    """RobertaSelfAttention with the same query/key/value parameters (so pretrained
    weights transfer directly), but with its softmax attention probabilities scaled
    elementwise by `group_prob`, as in the Tree Transformer paper's `attention()`."""

    def forward(self, hidden_states, attention_mask=None, group_prob=None, **kwargs):
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, self.num_attention_heads, self.attention_head_size)

        query_layer = self.query(hidden_states).view(*hidden_shape).transpose(1, 2)
        key_layer = self.key(hidden_states).view(*hidden_shape).transpose(1, 2)
        value_layer = self.value(hidden_states).view(*hidden_shape).transpose(1, 2)

        attn_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2)) * self.scaling
        if attention_mask is not None:
            attn_scores = attn_scores + attention_mask

        attn_probs = F.softmax(attn_scores, dim=-1)
        if group_prob is not None:
            attn_probs = attn_probs * group_prob.unsqueeze(1)
        attn_probs = self.dropout(attn_probs)

        context = torch.matmul(attn_probs, value_layer)
        context = context.transpose(1, 2).contiguous().reshape(*input_shape, -1)
        return context, attn_probs

class TreeAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.self = TreeSelfAttention(config)
        self.output = RobertaSelfOutput(config)

    def forward(self, hidden_states, attention_mask, group_prob):
        self_output, _ = self.self(hidden_states, attention_mask=attention_mask, group_prob=group_prob)
        return self.output(self_output, hidden_states)

class TreeLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attention = TreeAttention(config)
        self.intermediate = RobertaIntermediate(config)
        self.output = RobertaOutput(config)
        self.group_attn = GroupAttention(config.hidden_size)

    def forward(self, hidden_states, attention_mask, valid_mask, prior, masks: GroupAttentionMasks):
        group_prob, break_prob = self.group_attn(hidden_states, valid_mask, prior, masks)
        attention_output = self.attention(hidden_states, attention_mask, group_prob)
        layer_output = self.output(self.intermediate(attention_output), attention_output)
        return layer_output, group_prob, break_prob

class TreeEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.layer = nn.ModuleList([TreeLayer(config) for _ in range(config.num_hidden_layers)])

    def forward(self, hidden_states, attention_mask, valid_mask):
        seq_len = hidden_states.shape[1]
        # Built once here and shared by every layer below instead of being rebuilt
        # redundantly inside each layer's GroupAttention.
        masks = _build_group_attention_masks(seq_len, hidden_states.device)

        group_prob = 0.0
        break_probs = []
        for layer_module in self.layer:
            hidden_states, group_prob, break_prob = layer_module(hidden_states, attention_mask, valid_mask, group_prob, masks)
            break_probs.append(break_prob)
        return hidden_states, torch.stack(break_probs, dim=1)

class SegmentFreeRobertaModel(nn.Module):
    """XLM-RoBERTa-style encoder (embeddings + N encoder layers) whose self-attention is
    refined layer-by-layer with the Tree Transformer's constituent prior, instead of
    relying on absolute position embeddings alone to convey structure."""

    def __init__(self, config: XLMRobertaConfig):
        super().__init__()
        self.embeddings = RobertaEmbeddings(config)
        self.encoder = TreeEncoder(config)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        dtype = self.embeddings.word_embeddings.weight.dtype
        valid_mask = attention_mask.bool()
        extended_mask = (1.0 - attention_mask[:, None, None, :].to(dtype)) * torch.finfo(dtype).min

        hidden_states = self.embeddings(input_ids=input_ids)
        hidden_states, break_probs = self.encoder(hidden_states, extended_mask, valid_mask)
        return hidden_states, break_probs

class SegmentFreeBert(PreTrainedModel):
    config_class = SegmentFreeBertConfig

    def __init__(self, config: SegmentFreeBertConfig):
        super().__init__(config)
        self.config = config

        backbone_config = build_backbone_config(config)
        self.roberta = SegmentFreeRobertaModel(backbone_config)
        self.lm_head = RobertaLMHead(backbone_config)

        self.loss_fn = nn.CrossEntropyLoss(ignore_index=-100)

        self.post_init()

    @torch.no_grad()
    def _init_weights(self, module):
        super()._init_weights(module)
        if isinstance(module, RobertaLMHead):
            nn.init.zeros_(module.bias)
        elif isinstance(module, RobertaEmbeddings):
            module.position_ids.copy_(torch.arange(module.position_ids.shape[-1]).expand((1, -1)))
            module.token_type_ids.zero_()

    def load_xlm_roberta_backbone(self, pretrained_model_name: str = PRETRAINED_MODEL_NAME):
        """Copy every XLM-RoBERTa-base weight except the vocabulary-sized embedding and MLM
        head (kept random for SegmentFreeBERT's own vocabulary) and the GroupAttention
        modules (a new mechanism with no counterpart in the pretrained checkpoint, so they
        simply keep the initialization from post_init() above)."""
        pretrained = XLMRobertaForMaskedLM.from_pretrained(pretrained_model_name)
        own_state = self.state_dict()
        for key, tensor in pretrained.state_dict().items():
            if key in VOCAB_DEPENDENT_KEYS or key not in own_state:
                continue
            own_state[key] = tensor
        self.load_state_dict(own_state)
        return self

    def forward(self, input_ids, attention_mask=None, labels=None):
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)

        sequence_output, break_probs = self.roberta(input_ids, attention_mask)
        logits = self.lm_head(sequence_output)

        loss = None
        if labels is not None:
            loss = self.loss_fn(logits.view(-1, self.config.vocab_size), labels.view(-1))

        return {"loss": loss, "logits": logits, "break_probs": break_probs}
