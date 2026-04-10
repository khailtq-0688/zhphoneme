import torch
from torch.utils.data import Dataset

from vocabs.pinyin_tokenizer import PinyinTokenizer
from vocabs.pinyin_tokenizer import PinyinEncodedTokens

PAD_TOKEN_ID = 0

def collate_fn(samples: list[PinyinEncodedTokens]):
    max_len = 0
    for sample in samples:
        length, _ = sample.input_ids.shape
        if length > max_len:
            max_len = length

    bs = len(samples)
    input_ids = torch.zeros((bs, max_len, 3)).fill_(PAD_TOKEN_ID).long()
    labels = torch.zeros((bs, max_len, 3)).fill_(PAD_TOKEN_ID).long()
    attention_mask = torch.ones((bs, max_len))

    for idx, sample in enumerate(samples):
        input_len = sample.input_ids.shape[0]
        input_ids[idx, :input_len] = sample.input_ids
        labels[idx, :input_len] = sample.labels
        attention_mask[idx, :input_len] = sample.attention_mask

    return PinyinEncodedTokens(
        input_ids=input_ids,
        labels=labels,
        attention_mask=attention_mask
    )

class PinyinDataset(Dataset):
    def __init__(self, tokenizer: PinyinTokenizer, corpus_file, max_length=4096):
        self.max_length = max_length
        self.corpus_file = corpus_file
        self.tokenizer = tokenizer
        with open(corpus_file) as file:
            self.total_line = sum([1 for _ in file])

    def __len__(self):
        return self.total_line

    def __getitem__(self, idx):
        with open(self.corpus_file) as file:
            for irow, line in enumerate(file):
                if irow == idx:
                    text = line
                    break

        encoded_text = self.tokenizer.tokenize(text)

        return encoded_text
