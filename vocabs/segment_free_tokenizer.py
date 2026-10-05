import torch

from configs.segment_free_bert_config import SegmentFreeBertConfig
from .viphon_tokenizer import VietnameseEncodedTokens
from .Vietnamese_utils import is_Vietnamese
from .masking import create_roberta_mlm_labels

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
        return create_roberta_mlm_labels(input_ids, self.config)

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

    def encode_ids(self, sentence: str) -> list[int]:
        """Tokenize one sentence into vocab ids, without the leading <cls> or any
        truncation. Used directly by SegmentFreeDataset to pack several sentences into
        one training example (RoBERTa-style FULL-SENTENCES); encode() below is the
        single-sentence convenience wrapper built on top of it.
        """
        sentence = self.normalize(sentence)
        token_ids = []
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

        return token_ids

    def encode(self, sentence: str) -> torch.Tensor:
        token_ids = [self.config.cls_token_id] + self.encode_ids(sentence)
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
