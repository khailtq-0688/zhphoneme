from torch import nn
from transformers.models.roberta.modeling_roberta import RobertaLMHead

from configs.segment_free_bert_config import SegmentFreeBertConfig
from models.segment_free_bert import SegmentFreeBertBase, SegmentFreeRobertaModel, build_scratch_backbone_config

class SegmentFreeBertScratch(SegmentFreeBertBase):
    """Same Tree-Transformer-augmented architecture as SegmentFreeBert (models/segment_free_bert.py),
    but every weight -- including the backbone -- is randomly initialized instead of being
    transferred from XLM-RoBERTa-base. This isolates "does starting from XLM-R's pretrained
    weights actually help" as the single variable against the main model.
    """

    config_class = SegmentFreeBertConfig

    def __init__(self, config: SegmentFreeBertConfig):
        super().__init__(config)
        self.config = config

        backbone_config = build_scratch_backbone_config(config)
        self.roberta = SegmentFreeRobertaModel(backbone_config)
        self.lm_head = RobertaLMHead(backbone_config)

        self.loss_fn = nn.CrossEntropyLoss(ignore_index=-100)

        self.post_init()
