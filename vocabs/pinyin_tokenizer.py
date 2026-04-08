import torch

from pypinyin import pinyin, Style
from configs.pinyin_bert_config import PinyinBertConfig
from .hanzi_processing import HanziProcessor

from typing import *
import random
from collections import OrderedDict

class PinyinEncodedTokens(OrderedDict):
    def __init__(self, **kwargs):
        super().__init__(kwargs)

    def __setattr__(self, key, value):
        self[key] = value

    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError:
            raise AttributeError(f"{key} not found")

    def get_fields(self):
        """Get current attributes/fields registered under the sample.

        Returns:
            List[str]: Attributes registered under the Sample.

        """
        return list(self.keys())

class PinyinTokenizer:
    def __init__(self, config: PinyinBertConfig):
        self.config = config
        self.processor = HanziProcessor()

    def create_attention_mask(self, ids: torch.Tensor):
        ids = ids[:, 0]
        padding_mask = (ids == self.config.pad_token_id)
        cls_mask = (ids == self.config.cls_token_id)
        mask = torch.logical_or(padding_mask, cls_mask)
        mask = 1 - mask.long() # revert the mask, 0 masking while 1 for not masking

        return mask
    
    def create_labels(self, input_ids: torch.Tensor):
        labels = torch.zeros_like(input_ids).fill_(self.config.pad_token_id).long()
        length, _ = input_ids.shape
        for idx in range(length-1):
            token_id = input_ids[idx+1, 0]
            if token_id in self.config.special_ids:
                continue
            if random.random() <= 0.3:
                    labels[idx+1, :] = input_ids[idx+1, :]
                    input_ids[idx+1, :] = self.config.mask_token_id

        return input_ids, labels

    def encode(self, sentence: str) -> torch.Tensor:
        # truncate the sentence
        sentence = sentence.strip().lower()
        sentence = sentence[:self.config.max_length]
        syllables = [
            (self.config.cls_token_id, self.config.cls_token_id, self.config.cls_token_id)
        ]
        pinyin_words = pinyin(sentence, style=Style.TONE3)
        for pinyin_word in pinyin_words:
            pinyin_word = pinyin_word[0]
            is_pinyin, components = self.processor.process_IPA(pinyin_word)
            if is_pinyin:
                initial, rhyme, tone = components[-1]
                syllables.append((
                    self.config.label2id[initial] if initial else self.config.empty_token_id,
                    self.config.label2id[rhyme],
                    self.config.label2id[tone] if tone else self.config.empty_token_id 
                ))
            else:
                for char in pinyin_word:
                    syllables.append(
                        (self.config.label2id[char], ) * 3 if char in self.config.label2id else (self.config.unk_token_id, ) * 3
                    )

        vec = torch.tensor(syllables).long()

        return vec
    
    def tokenize(self, sentence: str) -> PinyinEncodedTokens:
        sentence_ids = self.encode(sentence)
        input_ids, labels = self.create_labels(sentence_ids)
        attention_mask = self.create_attention_mask(sentence_ids)
        return PinyinEncodedTokens(
            input_ids = input_ids,
            attention_mask = attention_mask,
            labels = labels
        )
