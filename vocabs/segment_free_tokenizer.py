import torch

from configs.segment_free_bert_config import SegmentFreeBertConfig
from .viphon_tokenizer import VietnameseEncodedTokens
from .Vietnamese_utils import is_Vietnamese

import re
from typing import *
import unicodedata

class SegmentFreeTokenizer:
    def __init__(self, config: SegmentFreeBertConfig):
        self.config = config

    def create_attention_mask(self, ids: torch.Tensor):
        mask = (ids != self.config.pad_token_id).long()

        return mask

    def create_labels(self, input_ids: torch.Tensor):
        # RoBERTa/BERT dynamic masking: 15% of non-special tokens are selected; of those,
        # 80% become <mask>, 10% become a random vocab token, and 10% are left unchanged
        # (but still count towards the loss, since the model must still predict them).
        labels = input_ids.clone()

        special_ids = torch.tensor(self.config.special_ids, device=input_ids.device)
        special_tokens_mask = torch.isin(input_ids, special_ids)

        probability_matrix = torch.full(labels.shape, 0.15)
        probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
        masked_indices = torch.bernoulli(probability_matrix).bool()
        labels[~masked_indices] = -100

        indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
        input_ids[indices_replaced] = self.config.mask_token_id

        indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        random_ids = torch.randint(self.config.vocab_size, labels.shape, dtype=torch.long)
        input_ids[indices_random] = random_ids[indices_random]

        # the remaining 10% of masked_indices are left unchanged in input_ids

        return input_ids, labels

    def normalize(self, text: str):
        text = text.lower()
        text = unicodedata.normalize("NFC", text)
        text = re.sub(r"\s+", " ", text)
        special_tokens = [
            "0", "1", "2", "3", "4", "5", "6",
            "7", "8", "9", "!", "@", "#", "$",
            "%", "^", "&", "*", "(", ")", "'",
            "\"", "-", "=", "[", "]", "{", "}",
            "|", "\\", ":", ";", "<", ">", "/",
            "?", ".", ",", "_", "。", "·"
        ]
        pattern = "(" + "|".join(re.escape(t) for t in special_tokens) + ")"
        # Insert spaces around matched tokens
        text = re.sub(pattern, r" \1 ", text)
        # Normalize multiple spaces
        text = re.sub(r"\s+", " ", text).strip()

        return text

    def encode(self, sentence: str) -> torch.Tensor:
        sentence = self.normalize(sentence)
        token_ids = [self.config.cls_token_id]
        for word in sentence.split():
            is_vietnamese, _ = is_Vietnamese(word)
            if is_vietnamese:
                # whole-syllable lookup; a structurally valid Vietnamese syllable
                # that is missing from the morpheme vocab falls back to <unk>
                token_ids.append(self.config.label2id.get(word, self.config.unk_token_id))
            else:
                # character-level fallback for tokens that are not Vietnamese
                # syllables (punctuation, digits, foreign words, ...)
                for char in word:
                    token_ids.append(self.config.label2id.get(char, self.config.unk_token_id))

        vec = torch.tensor(token_ids).long()
        # truncate the input
        vec = vec[:self.config.max_length]

        return vec

    def __call__(self, sentence: str) -> VietnameseEncodedTokens:
        sentence_ids = self.encode(sentence)
        _, labels = self.create_labels(sentence_ids)
        return VietnameseEncodedTokens(
            input_ids = sentence_ids,
            labels = labels
        )
