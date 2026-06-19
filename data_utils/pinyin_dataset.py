import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

from vocabs.pinyin_tokenizer import PinyinTokenizer
from vocabs.pinyin_tokenizer import PinyinEncodedTokens

import os
import numpy as np
import random
from tqdm import tqdm

PAD_TOKEN_ID = -100

def collate_fn(samples: list[PinyinEncodedTokens]):
    # Extract tensors
    input_ids_list = [s.input_ids for s in samples]   # (seq_len, 3)
    labels_list = [s.labels for s in samples]

    # Pad sequences (batch_first=True → (bs, max_len, ...))
    input_ids = pad_sequence(
        input_ids_list,
        batch_first=True,
        padding_value=0
    )

    labels = pad_sequence(
        labels_list,
        batch_first=True,
        padding_value=PAD_TOKEN_ID
    )

    # Attention mask: 1 where not PAD
    attention_mask = (input_ids[..., 0] != 0).long()

    return PinyinEncodedTokens(
        input_ids=input_ids,
        labels=labels,
        attention_mask=attention_mask,
    )

class PinyinDataset(Dataset):
    def __init__(self, tokenizer: PinyinTokenizer, corpus_dir, max_length=256):
        self.max_length = max_length
        self.corpus_dir = corpus_dir
        self.tokenizer = tokenizer
        self.txt_files = os.listdir(corpus_dir)
        self.corpus = []
        for txt_file in tqdm(self.txt_files, desc="Loading data"):
            texts = open(os.path.join(corpus_dir, txt_file)).readlines()
            self.corpus.extend(texts)

    def __len__(self):
        return len(self.corpus)

    def __getitem__(self, idx):
        text = self.corpus[idx]
        input_ids, labels = self.tokenizer(self.corpus[idx])
        total_sampling = 5
        
        for i in range(total_sampling):
            # whether or not we construct the input having more than two sentences
            if np.random.binomial(1, 0.5) == 0:
                continue
            
            random_text = random.choice(self.corpus)
            random_input_ids, random_labels = self.tokenizer(random_text)

            # ignore the <cls> token
            input_ids = torch.cat([input_ids, random_input_ids[1:]], dim=0)
            labels = torch.cat([labels, random_labels[1:]], dim=0)
            
            if input_ids.shape[0] > self.max_length:
                input_ids = input_ids[:self.max_length]
                labels = labels[:self.max_length]
                break

        return PinyinEncodedTokens(
            input_ids = input_ids,
            labels = labels
        )
