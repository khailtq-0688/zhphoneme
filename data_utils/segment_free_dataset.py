import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

from vocabs.segment_free_tokenizer import SegmentFreeTokenizer
from vocabs.viphon_tokenizer import VietnameseEncodedTokens

import hashlib
import multiprocessing
import os

import numpy as np
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

def _scan_file(path: str):
    """Runs in a worker process: one pass over one shard, recording each line's byte
    offset (for O(1) seeking later, instead of scanning from the start of the file on
    every read) and word count (a fast stand-in for its token count, used to decide
    packing boundaries without tokenizing the whole corpus up front). Opened in binary
    mode so the recorded offsets are true byte offsets, safe to seek() to later from a
    freshly-opened file handle -- text-mode tell()/seek() cookies are not guaranteed
    portable across separate file objects/processes.
    """
    word_counts = []
    byte_offsets = []
    with open(path, "rb") as file:
        offset = 0
        for line in file:
            byte_offsets.append(offset)
            offset += len(line)
            word_counts.append(max(1, len(line.split())))

    return (
        np.array(word_counts, dtype=np.int32),
        np.array(byte_offsets, dtype=np.int64),
    )

class SegmentFreeDataset(Dataset):
    """Packs consecutive sentences from the corpus into each training example until it
    fills max_length tokens, instead of one sentence per example -- the FULL-SENTENCES
    scheme from the RoBERTa paper (Liu et al., 2019, section 4.2), which they found to
    outperform one-sentence-per-example training. Packing is allowed to cross shard/file
    boundaries, same as RoBERTa allows crossing document boundaries: this corpus is a
    flat collection of independently-crawled sentences rather than ordered documents, so
    there is no document boundary to respect in the first place.

    Building the packing index requires one pass over every line of the corpus (tens to
    hundreds of millions of lines for the full pretraining corpus), so two things keep
    that from delaying every training run:
      - the per-file scan (_scan_file) runs in a process pool across all CPU cores
        instead of a single-threaded loop;
      - the resulting index is cached to disk (keyed by the shard file names/sizes and
        max_length) and simply loaded on every subsequent run instead of being rebuilt.
    """

    def __init__(
        self,
        tokenizer: SegmentFreeTokenizer,
        corpus_dir: str,
        max_length: int = 256,
        cache_dir: str | None = None,
        num_workers: int | None = None,
    ):
        self.max_length = max_length
        self.corpus_dir = corpus_dir
        self.tokenizer = tokenizer
        # isfile() filters out the cache dir below (it lives inside corpus_dir by
        # default) as well as any other stray subdirectory.
        self.txt_files = sorted(
            name for name in os.listdir(corpus_dir) if os.path.isfile(os.path.join(corpus_dir, name))
        )
        self.cache_dir = cache_dir or os.path.join(corpus_dir, ".segment_free_index_cache")

        cache_path = self._cache_path()
        cached = self._load_cache(cache_path)
        if cached is not None:
            self.chunk_file_idx, self.chunk_byte_offset, self.chunk_num_lines = cached
            return

        per_file_word_counts, per_file_byte_offsets = self._scan_corpus(num_workers)
        self.chunk_file_idx, self.chunk_byte_offset, self.chunk_num_lines = self._build_chunks(
            per_file_word_counts, per_file_byte_offsets
        )
        self._save_cache(cache_path)

    def _fingerprint(self) -> str:
        # Cheap to compute (just file sizes via stat, no content read) but changes if
        # shards are added/removed/resized or max_length changes, so a stale cache from
        # a different corpus or setting is never silently reused.
        parts = [f"max_length={self.max_length}"]
        for name in self.txt_files:
            size = os.path.getsize(os.path.join(self.corpus_dir, name))
            parts.append(f"{name}:{size}")
        return hashlib.md5("|".join(parts).encode()).hexdigest()

    def _cache_path(self) -> str:
        return os.path.join(self.cache_dir, f"index_{self._fingerprint()}.npz")

    def _load_cache(self, cache_path: str):
        if not os.path.isfile(cache_path):
            return None
        try:
            data = np.load(cache_path)
            return data["chunk_file_idx"], data["chunk_byte_offset"], data["chunk_num_lines"]
        except Exception as e:
            print(f"=> Could not read corpus index cache at {cache_path} ({e}), rebuilding.")
            return None

    def _save_cache(self, cache_path: str):
        try:
            os.makedirs(self.cache_dir, exist_ok=True)
            np.savez(
                cache_path,
                chunk_file_idx=self.chunk_file_idx,
                chunk_byte_offset=self.chunk_byte_offset,
                chunk_num_lines=self.chunk_num_lines,
            )
        except OSError as e:
            # e.g. corpus_dir is a read-only mount: training still works, it just has to
            # rebuild the index (in parallel, so still bounded) on every run.
            print(f"=> Could not write corpus index cache to {cache_path} ({e}), skipping.")

    def _scan_corpus(self, num_workers: int | None):
        paths = [os.path.join(self.corpus_dir, name) for name in self.txt_files]
        num_workers = num_workers or os.cpu_count() or 1

        word_counts, byte_offsets = [], []
        with multiprocessing.Pool(num_workers) as pool:
            for file_word_counts, file_byte_offsets in tqdm(
                pool.imap(_scan_file, paths), total=len(paths), desc="Indexing corpus"
            ):
                word_counts.append(file_word_counts)
                byte_offsets.append(file_byte_offsets)

        return word_counts, byte_offsets

    def _build_chunks(self, per_file_word_counts, per_file_byte_offsets):
        # Packing needs to know, for each training example, which range of consecutive
        # lines to concatenate. Decided here with a single greedy pass, using each
        # line's word count as a fast stand-in for its token count (exact tokenization --
        # Vietnamese-syllable analysis per word -- is too slow to run over the whole
        # corpus up front); actual encoding still hard-truncates to max_length, which
        # absorbs the rare case where a chunk's real token count slightly exceeds its
        # word-count estimate (e.g. a non-Vietnamese word expands into several character
        # tokens).
        chunk_file_idx, chunk_byte_offset, chunk_num_lines = [], [], []
        budget = max(1, self.max_length - 1)  # leave room for the leading <cls>

        cur_file_idx, cur_byte_offset, cur_num_lines, cur_word_count = None, None, 0, 0

        for file_idx, (word_counts, byte_offsets) in enumerate(zip(per_file_word_counts, per_file_byte_offsets)):
            for local_idx, word_count in enumerate(word_counts.tolist()):
                if cur_num_lines > 0 and cur_word_count + word_count > budget:
                    chunk_file_idx.append(cur_file_idx)
                    chunk_byte_offset.append(cur_byte_offset)
                    chunk_num_lines.append(cur_num_lines)
                    cur_num_lines, cur_word_count = 0, 0

                if cur_num_lines == 0:
                    cur_file_idx = file_idx
                    cur_byte_offset = int(byte_offsets[local_idx])

                cur_word_count += word_count
                cur_num_lines += 1

        if cur_num_lines > 0:
            chunk_file_idx.append(cur_file_idx)
            chunk_byte_offset.append(cur_byte_offset)
            chunk_num_lines.append(cur_num_lines)

        return (
            np.array(chunk_file_idx, dtype=np.int32),
            np.array(chunk_byte_offset, dtype=np.int64),
            np.array(chunk_num_lines, dtype=np.int32),
        )

    def __len__(self):
        return len(self.chunk_file_idx)

    def _read_chunk(self, file_idx: int, byte_offset: int, count: int) -> list[str]:
        lines = []
        while len(lines) < count and file_idx < len(self.txt_files):
            with open(os.path.join(self.corpus_dir, self.txt_files[file_idx]), "rb") as file:
                file.seek(byte_offset)
                for raw_line in file:
                    lines.append(raw_line.decode("utf-8"))
                    if len(lines) == count:
                        break
            file_idx += 1
            byte_offset = 0

        return lines

    def __getitem__(self, idx):
        lines = self._read_chunk(
            int(self.chunk_file_idx[idx]), int(self.chunk_byte_offset[idx]), int(self.chunk_num_lines[idx])
        )

        token_ids = [self.tokenizer.config.cls_token_id]
        for line in lines:
            token_ids.extend(self.tokenizer.encode_ids(line))

        # Fallback in case the shard(s) on disk are shorter than what was indexed (e.g.
        # truncated after indexing), so the DataLoader never crashes mid-epoch.
        if len(token_ids) == 1:
            token_ids.extend(self.tokenizer.encode_ids(self.tokenizer.config.pad_token))

        input_ids = torch.tensor(token_ids[:self.max_length]).long()
        input_ids, labels = self.tokenizer.create_labels(input_ids)

        return VietnameseEncodedTokens(input_ids=input_ids, labels=labels)
