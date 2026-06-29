import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

from vocabs.viphon_tokenizer import ViPhonTokenizer, VietnameseEncodedTokens

from tqdm import tqdm
import random
import numpy as np

import os
from bisect import bisect_right

PAD_TOKEN_ID = -100

def _subset_sort_key(path):
    stem, _ = os.path.splitext(path)
    prefix, _, suffix = stem.rpartition("_")
    if prefix == "subset" and suffix.isdigit():
        return (0, int(suffix))
    return (1, path)

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
        txt_files = sorted(os.listdir(corpus_dir), key=_subset_sort_key)
        self.txt_files = []
        self.total_line = 0
        self.cumulative_lines = []
        for txt_file in tqdm(txt_files, desc="Loading corpus"):
            with open(os.path.join(corpus_dir, txt_file)) as file:
                texts = file.readlines()
            if not texts:
                continue
            self.txt_files.append(txt_file)
            self.total_line += len(texts)
            self.cumulative_lines.append(self.total_line)

    def __len__(self):
        return self.total_line

    def __getitem__(self, idx):
        file_idx = bisect_right(self.cumulative_lines, idx)
        previous_total = 0 if file_idx == 0 else self.cumulative_lines[file_idx - 1]
        line_idx = idx - previous_total
        with open(os.path.join(self.corpus_dir, self.txt_files[file_idx])) as file:
            subset = file.readlines()
        sentence = subset[line_idx]
        input_ids, labels = self.tokenizer(sentence)
        input_ids = input_ids[:self.max_length]
        labels = labels[:self.max_length]
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
