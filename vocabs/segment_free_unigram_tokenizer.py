import torch

from configs.segment_free_bert_unigram_config import SegmentFreeBertUnigramConfig
from .viphon_tokenizer import VietnameseEncodedTokens
from .masking import create_roberta_mlm_labels

import unicodedata

from tokenizers import Tokenizer

class SegmentFreeUnigramTokenizer:
    """Same role as SegmentFreeTokenizer (segment_free_tokenizer.py), but delegates
    segmentation to a pretrained SentencePiece-style Unigram model (trained by
    train_unigram_tokenizer.py) instead of whole-syllable lookup + character fallback.
    Exposes the same interface (encode_ids/encode/create_labels/__call__), so it is a
    drop-in replacement wherever SegmentFreeTokenizer is used -- in particular,
    data_utils/segment_free_dataset.py's SegmentFreeDataset needs no changes to pack
    sentences tokenized this way.
    """

    def __init__(self, config: SegmentFreeBertUnigramConfig):
        self.config = config
        self.hf_tokenizer = Tokenizer.from_file(config.unigram_tokenizer_path)

    def normalize(self, text: str) -> str:
        # NFC + lowercase only: the trained tokenizer's own normalizer/pre-tokenizer
        # (Metaspace) already handles punctuation, digits and whitespace, unlike
        # SegmentFreeTokenizer's hand-written normalize(), which has to pre-split
        # punctuation itself before its whole-syllable lookup.
        return unicodedata.normalize("NFC", text.lower())

    def encode_ids(self, sentence: str) -> list[int]:
        """Tokenize one sentence into vocab ids, without the leading <cls> or any
        truncation -- used directly by SegmentFreeDataset to pack several sentences into
        one training example (RoBERTa-style FULL-SENTENCES).
        """
        return self.hf_tokenizer.encode(self.normalize(sentence)).ids

    def encode(self, sentence: str) -> torch.Tensor:
        token_ids = [self.config.cls_token_id] + self.encode_ids(sentence)
        vec = torch.tensor(token_ids).long()
        return vec[:self.config.max_length]

    def create_labels(self, input_ids: torch.Tensor):
        return create_roberta_mlm_labels(input_ids, self.config)

    def __call__(self, sentence: str) -> VietnameseEncodedTokens:
        sentence_ids = self.encode(sentence)
        _, labels = self.create_labels(sentence_ids)
        return VietnameseEncodedTokens(
            input_ids=sentence_ids,
            labels=labels
        )
