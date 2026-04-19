from datasets import load_from_disk

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

from vocabs.pinyin_tokenizer import PinyinTokenizer
from vocabs.pinyin_tokenizer import PinyinEncodedTokens

PAD_TOKEN_ID = 0

def collate_fn(samples: list[PinyinEncodedTokens], tokenizer: PinyinTokenizer):
    # Extract tensors
    input_ids_list = [torch.tensor(s.input_ids) for s in samples]   # (seq_len, 3)
    labels_list = [tokenizer.create_labels(input_ids) for input_ids in input_ids_list]
    attention_masks_list = [torch.tensor(sample.attention_mask) for sample in samples]

    # Pad sequences (batch_first=True → (bs, max_len, ...))
    input_ids = pad_sequence(
        input_ids_list,
        batch_first=True,
        padding_value=PAD_TOKEN_ID
    )

    labels = pad_sequence(
        labels_list,
        batch_first=True,
        padding_value=PAD_TOKEN_ID
    )

    attention_mask = pad_sequence(
        attention_masks_list,
        batch_first=True,
        padding_value=0
    )

    return PinyinEncodedTokens(
        input_ids=input_ids.long(),
        labels=labels.long(),
        attention_mask=attention_mask.long(),
    )

class PinyinDataset(Dataset):
    def __init__(self, tokenizer, corpus_dir, max_length=256):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.corpus = load_from_disk(corpus_dir, keep_in_memory=True)["train"]

    def __len__(self):
        return len(self.corpus)

    def __getitem__(self, idx):
        return PinyinEncodedTokens(**self.corpus[idx])
