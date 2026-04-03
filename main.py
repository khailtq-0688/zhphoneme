import os
import random
import numpy as np
import sys
import json

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import BertModel, BertConfig, PreTrainedModel
from tqdm.auto import tqdm

from vocabs.hanzi_processing import HanziProcessor

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(f"PyTorch version: {torch.__version__}")
print(f"Device: {device}")

print("Loading dataset...")

processor = HanziProcessor()

# Khởi tạo các token đặc biệt cho BERT
SPECIAL_TOKENS = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "<EMPTY>"]
UNIFIED_VOCAB = {token: idx for idx, token in enumerate(SPECIAL_TOKENS)}



