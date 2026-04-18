from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

from vocabs.pinyin_tokenizer import PinyinTokenizer
from vocabs.pinyin_tokenizer import PinyinEncodedTokens

PAD_TOKEN_ID = 0

def collate_fn(samples: list[PinyinEncodedTokens]):
    # Extract tensors
    input_ids_list = [s.input_ids for s in samples]   # (seq_len, 3)
    labels_list = [s.labels for s in samples]

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

    # Attention mask: 1 where not PAD
    attention_mask = (input_ids[..., 0] != PAD_TOKEN_ID).long()

    return PinyinEncodedTokens(
        input_ids=input_ids,
        labels=labels,
        attention_mask=attention_mask,
    )

class PinyinDataset(Dataset):
    def __init__(self, tokenizer, corpus_file, max_length=256):
        self.tokenizer = tokenizer
        self.max_length = max_length

        # Faster + cleaner file loading
        with open(corpus_file, "r", encoding="utf-8") as f:
            self.texts = [line for line in f if line.strip()]

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]

        # Single call, no redundant indexing or strip
        return self.tokenizer(text)
