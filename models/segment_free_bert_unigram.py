from torch import nn
from transformers.models.roberta.modeling_roberta import RobertaLMHead

from configs.segment_free_bert_unigram_config import SegmentFreeBertUnigramConfig
from models.segment_free_bert import SegmentFreeBertBase, SegmentFreeRobertaModel, build_scratch_backbone_config

class SegmentFreeBertUnigram(SegmentFreeBertBase):
    """Same Tree-Transformer-augmented architecture as SegmentFreeBert (models/segment_free_bert.py),
    but tokenized with a trained SentencePiece-style Unigram subword vocabulary (see
    train_unigram_tokenizer.py / vocabs/segment_free_unigram_tokenizer.py) instead of the
    hand-curated whole-syllable vocabulary, isolating "does whole-syllable segmentation-
    free tokenization help over standard subword tokenization" as the variable being
    tested. Trained fully from random initialization: no pretrained checkpoint shares
    this vocabulary, so there is nothing to transfer.
    """

    config_class = SegmentFreeBertUnigramConfig

    def __init__(self, config: SegmentFreeBertUnigramConfig):
        super().__init__(config)
        self.config = config

        backbone_config = build_scratch_backbone_config(config)
        self.roberta = SegmentFreeRobertaModel(backbone_config)
        self.lm_head = RobertaLMHead(backbone_config)

        self.loss_fn = nn.CrossEntropyLoss(ignore_index=-100)

        self.post_init()
