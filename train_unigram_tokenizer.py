"""One-time preprocessing step for train_segment_free_bert_unigram.py: trains a
SentencePiece-style Unigram subword tokenizer on the pretraining corpus and saves it to
disk. Run this once (it is not part of, and would otherwise delay, the training script
itself -- same reasoning as caching the FULL-SENTENCES packing index in
data_utils/segment_free_dataset.py):

    python3 train_unigram_tokenizer.py \\
        --corpus_dir data/Vietnamese-curated-corpus \\
        --vocab_size 32000 \\
        --output_path vocabs/unigram_tokenizer.json

configs/segment_free_bert_unigram_config.py then loads the saved tokenizer.json directly
to build its vocabulary (label2id/id2label/vocab_size), rather than embedding a 32K-entry
subword list as a literal Python list the way the hand-curated morpheme vocab is -- a
trained subword vocabulary is naturally a serialized model artifact, not a hand-curated
closed list.
"""

import argparse
import os

from tokenizers import Tokenizer, decoders, normalizers, pre_tokenizers
from tokenizers.models import Unigram
from tokenizers.trainers import UnigramTrainer

# Must match the special-token order every other SegmentFreeBERT config uses
# (pad=0, cls=1, empty=2, mask=3, unk=4), since UnigramTrainer assigns ids to
# special_tokens in the order given, before any trained subword piece.
SPECIAL_TOKENS = ["<pad>", "<cls>", "<empty>", "<mask>", "<unk>"]

def iter_corpus_lines(corpus_dir: str):
    for name in sorted(os.listdir(corpus_dir)):
        path = os.path.join(corpus_dir, name)
        if not os.path.isfile(path):
            continue
        with open(path, encoding="utf-8") as file:
            for line in file:
                yield line

def train_unigram_tokenizer(corpus_dir: str, vocab_size: int, output_path: str):
    tokenizer = Tokenizer(Unigram())
    tokenizer.normalizer = normalizers.Sequence([normalizers.NFC(), normalizers.Lowercase()])
    tokenizer.pre_tokenizer = pre_tokenizers.Metaspace()
    tokenizer.decoder = decoders.Metaspace()

    trainer = UnigramTrainer(
        vocab_size=vocab_size,
        special_tokens=SPECIAL_TOKENS,
        unk_token="<unk>",
    )
    tokenizer.train_from_iterator(iter_corpus_lines(corpus_dir), trainer=trainer)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    tokenizer.save(output_path)
    print(f"Saved Unigram tokenizer ({tokenizer.get_vocab_size()} tokens) to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus_dir", default="data/Vietnamese-curated-corpus")
    parser.add_argument("--vocab_size", type=int, default=32000)
    parser.add_argument("--output_path", default="vocabs/unigram_tokenizer.json")
    args = parser.parse_args()

    train_unigram_tokenizer(args.corpus_dir, args.vocab_size, args.output_path)
