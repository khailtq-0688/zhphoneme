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
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
        self.position_embedding_type = position_embedding_type
        self.use_cache = use_cache
        self.classifier_dropout = classifier_dropout

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

        initials = [
            "tsʰ", "tɕʰ", "tʰ", "ʈʂʰ", "tɕ", "ts", 
            "ʈʂ", "kʰ", "pʰ", "ɕ", "f", "j", "k", 
            "l",  "m", "n", "ŋ", "p", "ʐ", "s", "ʂ", 
            "t", "w", "x", "ɹ̩", "ɻ", "ɹ", "h"
        ]
        rhymes = [
            "a", "ɔ", "o", "ɤ", "i", "u", "y", "ɥe",
            "ai̯", "ei̯", "au̯", "ou̯", "iau̯", "iou̯",
            "uai̯", "uei̯", "an", "ən", "in", "uən",
            "yn", "ɤŋ", "aŋ", "əŋ", "iŋ", "ʊŋ", "ja", 
            "je", "jɛ", "wa", "wo", "ye", "yɛ", "jɛn", "jen", "jaŋ",
            "jʊŋ", "wan", "waŋ", "wəŋ", "ɿ", "ʅ", "ɻ̩", "ɥɛn"
        ]
        tones = [
            "˧˩˧", "˧˧˥", "˧˩", "˩˧", "˥˧", "˥˩", "˧˥", "˥", "˩", "˧", "˩",
        ]

        phonemes = initials + rhymes + tones
        self.id2label = {idx: phoneme for idx, phoneme in enumerate(self.specials + phonemes)}
        self.label2id = {phoneme: idx for idx, phoneme in enumerate(self.specials + phonemes)}

        self.vocab_size = len(self.label2id)
