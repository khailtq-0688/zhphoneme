import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

from vocabs.viphon_tokenizer import ViPhonTokenizer, VietnameseEncodedTokens

from tqdm import tqdm
import random
import numpy as np

import os

PAD_TOKEN_ID = -100

def collate_fn(samples: list[VietnameseEncodedTokens]):
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

    return VietnameseEncodedTokens(
        input_ids=input_ids,
        labels=labels,
        attention_mask=attention_mask,
    )

class ViPhonDataset(Dataset):
    def __init__(self, tokenizer: ViPhonTokenizer, corpus_dir, max_length=512):
        self.max_length = max_length
        self.corpus_dir = corpus_dir
        self.tokenizer = tokenizer
        self.txt_files = os.listdir(corpus_dir)
        self.total_line = 0
        for txt_file in tqdm(self.txt_files, desc="Loading corpus"):
            texts = open(os.path.join(corpus_dir, txt_file)).readlines()
            self.total_line += len(texts)

        self.LINE_PER_FILE = 10_000

    def __len__(self):
        return self.total_line

    def __getitem__(self, idx):
        subset_idx, line_idx = divmod(idx, self.LINE_PER_FILE)
        with open(os.path.join(self.corpus_dir, f"subset_{subset_idx}.txt")) as file:
            subset = file.readlines()
        sentence = subset[line_idx]
        input_ids, labels = self.tokenizer(sentence)
        total_sampling = 5
                
        for i in range(total_sampling):
            # whether or not we construct the input having more than two sentences
            if np.random.binomial(1, 0.5) == 1:
                continue

            random_sentence = random.choice(subset)
            random_input_ids, random_labels = self.tokenizer(random_sentence)

            # ignore the <cls> token
            input_ids = torch.cat([input_ids, random_input_ids[1:]], dim=0)
            labels = torch.cat([labels, random_labels[1:]], dim=0)

            if input_ids.shape[0] > self.max_length:
                input_ids = input_ids[:self.max_length]
                labels = labels[:self.max_length]
                break

        return VietnameseEncodedTokens(
            input_ids = input_ids,
            labels = labels
        )
