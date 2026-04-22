from tokenizers import Tokenizer, models, trainers, pre_tokenizers

import os
import logging
import re

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BpeTokenizer:
    def __init__(self, model_prefix: str):
        self.is_loaded = False

        if os.path.isfile(os.path.join(model_prefix, "bpe.json")):
            logger.info("Found the pretrained tokenizer at {os.path.join(model_prefix, 'bpe.json'))}. Start loading ...")
            self.tokenizer = Tokenizer.from_file(os.path.join(model_prefix, "bpe.json"))
            self.is_loaded = True
        else:
            logger.info("There is no available tokenizer. Start training ...")
            # 1. Initialize BPE model
            self.tokenizer = Tokenizer(models.BPE(unk_token="[UNK]"))

            # 2. Pre-tokenization (IMPORTANT)
            self.tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()

            # 3. Trainer
            self.trainer = trainers.BpeTrainer(
                vocab_size=32000,
                min_frequency=2,
                special_tokens=["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"],
                limit_alphabet=1000
            )

            self.model_prefix = model_prefix

    def get_tokenizer(self):
        return self.tokenizer

    def train(self, corpus_dir):
        if self.is_loaded:
            logger.info("Tokenizer is already loaded from file. Skipping training phase.")
            return
        
        vi_pattern = re.compile(
            r'[^a-zA-Z0-9\s.,!?;:\'\"()\[\]{}<>_+\-=/@#%&'
            r'àáâãèéêìíòóôõùúăđĩũơưạảấpầẩẫậắằẳẵặẹẻẽếềểễệỉịọỏốồổỗộớờởỡợụủứừửữựỳỵỷỹ'
            r'ÀÁÂÃÈÉÊÌÍÒÓÔÕÙÚĂĐĨŨƠƯẠẢẤẦẨẪẬẮẰẲẴẶẸẺẼẾỀỂỄỆỈỊỌỎỐỒỔỖỘỚỜỞỠỢỤỦỨỪỬỮỰỲỴỶỸ]'
        )
        
#         def generate_text():
#             import os
#             for txt_file in os.listdir(corpus_dir):
#                 if not txt_file.endswith(".txt"):
#                     continue
#                 with open(os.path.join(corpus_dir, txt_file)) as file:
#                     for line in file:
#                         yield line

#         # 4. Streaming generator
#         def generate_text():
#             for fname in os.listdir(corpus_dir):
#                 with open(os.path.join(corpus_dir, fname), encoding="utf-8") as f:
#                     for line in f:
#                         yield line.strip()
# #             def generate_text():
# #                 for fname in os.listdir(corpus_dir):
# #                     if not fname.endswith(".txt"):
# #                         continue
# #                     with open(os.path.join(corpus_dir, fname), encoding="utf-8") as f:
# #                         for line in f:
# #                             yield line.strip()

        def generate_text():
            for fname in os.listdir(corpus_dir):
                if not fname.endswith(".txt"):
                    continue
                with open(os.path.join(corpus_dir, fname), encoding="utf-8") as f:
                    for line in f:
                        cleaned_line = vi_pattern.sub(' ', line.strip())
                        
                        cleaned_line = re.sub(r'\s+', ' ', cleaned_line).strip()
                        
                        if cleaned_line:
                            yield cleaned_line

        # 5. Train from iterator (this is the key)
        self.tokenizer.train_from_iterator(generate_text(), trainer=self.trainer)

        # 6. Save
        os.makedirs(self.model_prefix, exist_ok=True)
        self.tokenizer.save(os.path.join(self.model_prefix, "bpe.json"))
