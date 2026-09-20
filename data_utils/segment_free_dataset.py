from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

from vocabs.segment_free_tokenizer import SegmentFreeTokenizer
from vocabs.viphon_tokenizer import VietnameseEncodedTokens

import bisect
import os
from tqdm import tqdm

PAD_TOKEN_ID = 0
LABEL_IGNORE_INDEX = -100

def collate_fn(samples: list[VietnameseEncodedTokens]):
    input_ids = pad_sequence(
        [s.input_ids for s in samples],
        batch_first=True,
        padding_value=PAD_TOKEN_ID
    )
    labels = pad_sequence(
        [s.labels for s in samples],
        batch_first=True,
        padding_value=LABEL_IGNORE_INDEX
    )
    attention_mask = (input_ids != PAD_TOKEN_ID).long()

    return VietnameseEncodedTokens(
        input_ids=input_ids,
        labels=labels,
        attention_mask=attention_mask,
    )

class SegmentFreeDataset(Dataset):
    def __init__(self, tokenizer: SegmentFreeTokenizer, corpus_dir: str, max_length: int = 256):
        self.max_length = max_length
        self.corpus_dir = corpus_dir
        self.tokenizer = tokenizer
        self.txt_files = sorted(os.listdir(corpus_dir))

        # Corpus shards don't all hold the same number of lines (the last shard of a split is
        # usually a remainder), so a flat sample index is mapped to (file, line) through the
        # per-file cumulative line counts below instead of assuming a fixed lines-per-file.
        self.cumulative_lines = [0]
        for txt_file in tqdm(self.txt_files, desc="Indexing corpus"):
            with open(os.path.join(corpus_dir, txt_file)) as file:
                num_lines = sum(1 for _ in file)
            self.cumulative_lines.append(self.cumulative_lines[-1] + num_lines)

    def __len__(self):
        return self.cumulative_lines[-1]

    def __getitem__(self, idx):
        file_idx = bisect.bisect_right(self.cumulative_lines, idx) - 1
        line_idx = idx - self.cumulative_lines[file_idx]

        encoded_text = None
        with open(os.path.join(self.corpus_dir, self.txt_files[file_idx])) as file:
            for line_ith, text in enumerate(file):
                if line_ith == line_idx:
                    encoded_text = self.tokenizer(text)
                    break

        # Fallback in case the shard on disk is shorter than what was indexed (e.g. it was
        # truncated after indexing), so the DataLoader never crashes mid-epoch.
        if encoded_text is None:
            encoded_text = self.tokenizer(self.tokenizer.config.pad_token)

        encoded_text.input_ids = encoded_text.input_ids[:self.max_length]
        encoded_text.labels = encoded_text.labels[:self.max_length]

        return encoded_text
