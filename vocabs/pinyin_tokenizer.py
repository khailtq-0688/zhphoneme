import torch

from pypinyin import pinyin, Style
from configs.pinyin_bert_config import PinyinBertConfig
from .pinyin_decomposation import hanzi_to_components

from typing import *
import random

class PinyinEncodedTokens:
    def __init__(self, input_ids, labels, attention_mask):
        self.input_ids = input_ids
        self.labels = labels
        self.attention_mask = attention_mask

class PinyinTokenizer:
    def __init__(self, config: PinyinBertConfig):
        self.config = config

    def create_attention_mask(self, ids: torch.Tensor):
        ids = ids[:, :, 0]
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
            if token_id in self.config.specials:
                continue
            if random.random() <= 0.3:
                    labels[idx+1, :] = input_ids[idx+1, :]
                    input_ids[idx+1, :] = self.config.mask_token_id

        return input_ids, labels

    def encode(self, sentence: str) -> torch.Tensor:
        # truncate the sentence
        sentence = sentence[:self.config.max_length]
        syllables = [
            (self.config.cls_token_id, self.config.cls_token_id, self.config.cls_token_id)
        ]
        pinyin_words = pinyin(sentence, style=Style.TONE3)
        for words in pinyin_words:
            for word in words:
                components = hanzi_to_components(word)
            for component in components:
                if component["final"] == "":
                    for char in word:
                        syllables.append(
                            (self.config.label2id[char], ) * 3 if char in self.config.label2id else self.config.unk_token_id
                        )
                else:
                    syllables.append((
                        self.config.label2id[component["onset"]] if component["onset"] or component["onset"] != "" else self.config.empty_token_id,
                        self.config.label2id[component["final"]],
                        self.config.label2id[component["tone"]] if component["tone"] or component["tone"] != "" else self.config.empty_token_id 
                    ))

        vec = torch.tensor(syllables).long()

        return vec
    
    def __call__(self, sentence: str) -> PinyinEncodedTokens:
        sentence_ids = self.encode(sentence)
        input_ids, labels = self.create_labels(sentence_ids)
        return PinyinEncodedTokens(
            input_ids = input_ids,
            attention_mask = self.create_attention_mask(sentence_ids),
            labels = labels
        )
