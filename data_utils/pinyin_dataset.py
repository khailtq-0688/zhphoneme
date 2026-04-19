import torch
from torch.utils.data import Dataset

from vocabs.pinyin_tokenizer import PinyinTokenizer
from vocabs.pinyin_tokenizer import PinyinEncodedTokens

import os

PAD_TOKEN_ID = 0

def collate_fn(samples: list[PinyinEncodedTokens]):
    max_len = 0
    for sample in samples:
        length, _ = sample.input_ids.shape
        if length > max_len:
            max_len = length

    bs = len(samples)
    input_ids = torch.full((bs, max_len, 3), fill_value=PAD_TOKEN_ID, dtype=torch.long)
    labels = torch.full((bs, max_len, 3), fill_value=PAD_TOKEN_ID, dtype=torch.long)
    attention_mask = torch.zeros((bs, max_len))

    for idx, sample in enumerate(samples):
        input_len = sample.input_ids.shape[0]
        input_ids[idx, :input_len] = sample.input_ids
        labels[idx, :input_len] = sample.labels
        attention_mask[idx, :input_len] = sample.attention_mask

    return PinyinEncodedTokens(
        input_ids=input_ids.long(),
        labels=labels.long(),
        attention_mask=attention_mask.long()
    )

class PinyinDataset(Dataset):
    def __init__(self, tokenizer: PinyinTokenizer, corpus_dir, max_length=256):
        self.max_length = max_length
        self.corpus_dir = corpus_dir
        self.tokenizer = tokenizer
        self.txt_files = os.listdir(corpus_dir)
        self.total_line = 0
        for txt_file in self.txt_files:
            texts = open(os.path.join(corpus_dir, txt_file)).readlines()
            self.total_line += len(texts)

        self.LINE_PER_FILE = 1_000

    def __len__(self):
        return self.total_line

    def __getitem__(self, idx):
        # the default format for the corpus file of each line if subset_<idx>.txt
        subset_idx, line_idx = divmod(idx+1, self.LINE_PER_FILE)
        with open(os.path.join(self.corpus_dir, f"subset_{subset_idx}.txt")) as file:
            texts = file.readlines()
        
        encoded_text = self.tokenizer.tokenize(texts[line_idx-1])

        return encoded_text