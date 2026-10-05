from transformers.configuration_utils import PretrainedConfig

from tokenizers import Tokenizer


class SegmentFreeBertUnigramConfig(PretrainedConfig):
    """Same architecture hyperparameters as SegmentFreeBertConfig, but the vocabulary
    comes from a SentencePiece-style Unigram subword tokenizer trained on the corpus
    (see train_unigram_tokenizer.py) instead of the hand-curated whole-syllable list in
    configs/morphemes.json. A trained subword vocabulary is a serialized model artifact
    rather than a hand-curated closed list, so it is loaded from that tokenizer file at
    construction time instead of being embedded as a literal Python list.
    """

    model_type = "segmentfreebert_unigram"

    def __init__(
        self,
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        intermediate_size=3072,
        hidden_act="gelu",
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        max_position_embeddings=512,
        max_length=512,
        type_vocab_size=1,
        initializer_range=0.02,
        layer_norm_eps=1e-12,
        position_embedding_type="absolute",
        use_cache=True,
        classifier_dropout=None,
        unigram_tokenizer_path="vocabs/unigram_tokenizer.json",
        **kwargs,
    ):
        pad_token_id = kwargs.pop("pad_token_id", 0)

        # Gọi hàm super() với pad_token_id đã được xử lý
        super().__init__(pad_token_id=pad_token_id, **kwargs)

        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.hidden_act = hidden_act
        self.intermediate_size = intermediate_size
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.max_length = max_length
        self.type_vocab_size = type_vocab_size
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
        self.position_embedding_type = position_embedding_type
        self.use_cache = use_cache
        self.classifier_dropout = classifier_dropout

        self.unigram_tokenizer_path = unigram_tokenizer_path

        self.pad_token = "<pad>"
        self.cls_token = "<cls>"
        self.empty_token = "<empty>"
        self.mask_token = "<mask>"
        self.unk_token = "<unk>"

        self.specials = [self.pad_token, self.cls_token, self.empty_token, self.mask_token, self.unk_token]

        self.pad_token_id = 0
        self.cls_token_id = 1
        self.empty_token_id = 2
        self.mask_token_id = 3
        self.unk_token_id = 4

        self.special_ids = [self.pad_token_id, self.cls_token_id, self.empty_token_id, self.mask_token_id, self.unk_token_id]

        # Special tokens are passed to UnigramTrainer in this same order (see
        # train_unigram_tokenizer.py), so the trained tokenizer assigns them these exact
        # ids before any trained subword piece.
        try:
            tokenizer = Tokenizer.from_file(unigram_tokenizer_path)
            self.label2id = tokenizer.get_vocab()
        except Exception:
            # transformers' own save_pretrained() constructs a bare self.__class__()
            # with no arguments purely to diff against when deciding what to write to
            # config.json; that internal probe uses unigram_tokenizer_path's default,
            # which need not exist, so this falls back instead of crashing save_pretrained().
            self.label2id = {token: idx for idx, token in enumerate(self.specials)}

        self.id2label = {idx: token for token, idx in self.label2id.items()}
        self.vocab_size = len(self.label2id)
