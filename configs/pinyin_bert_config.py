from transformers.configuration_utils import PretrainedConfig

class PinyinBertConfig(PretrainedConfig):
    model_type = "pinyinbert"

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
        max_length=4096,
        type_vocab_size=1,
        initializer_range=0.02,
        layer_norm_eps=1e-12,
        position_embedding_type="absolute",
        use_cache=True,
        classifier_dropout=None,
        **kwargs,
    ):
        super().__init__(pad_token_id=0, **kwargs)

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

        self.pad_token = "<pad>"
        self.cls_token = "<cls>"
        self.sep_token = "<sep>"
        self.empty_token = "<empty>"
        self.mask_token = "<mask>"
        self.unk_token = "<unk>"

        self.specials = [self.pad_token, self.cls_token, self.sep_token, self.empty_token, self.mask_token, self.unk_token]

        self.pad_token_id = 0
        self.cls_token_id = 1
        self.sep_token_id = 2
        self.empty_token_id = 3
        self.mask_token_id = 4
        self.unk_token_id = 5

        self.special_ids = [self.pad_token_id, self.cls_token_id, self.sep_token_id, self.empty_token_id, self.mask_token_id, self.unk_token_id]

        self.initials = [
            "tɕ", "m", "p", "tʰ", "t", "s", "ts", "k", "ɕ", "ʈʂʰ", "kʰ", "l",
            "h", "n", "ʈʂ", "x", "f", "w", "ʂ", "tɕʰ", "ɻ", "pʰ", "tsʰ"
        ]
        self.rhymes = [
            "au̯", "y", "wən", "ɤ", "jaŋ", "je","aŋ", 
            "wai̯", "iŋ", "a", "ɹ̩", "in", "ɛ", "wan", 
            "wei̯", "yn", "o", "jɛn", "ja", "jʊŋ", "an", 
            "jou̯", "i", "ɥɛn", "ɔ", "ən", "ɥe",
            "ʊŋ", "ɻ̩", "ɚ", "əŋ", "waŋ", "u", "wo", 
            "ai̯", "wa", "jau̯", "ou̯", "ei̯"
        ]
        self.tones = [
            "˥", "˧˩˧", "˥˩", "˧˥"
        ]

        self.others = [
            "0", "1", "2", "3", "4", "5", "6", 
            "7", "8", "9", "!", "@", "#", "$", 
            "%", "^", "&", "*", "(", ")", "'",
            "\"", "-", "=", "[", "]", "{", "}",
            "|", "\\", ":", ";", "<", ">", "/",
            "?", ".", ",", "_", "。", "·"
        ]

        phonemes = self.initials + self.rhymes + self.tones + self.others
        self.id2label = {idx: phoneme for idx, phoneme in enumerate(self.specials + phonemes)}
        self.label2id = {phoneme: idx for idx, phoneme in enumerate(self.specials + phonemes)}

        self.vocab_size = len(self.label2id)
