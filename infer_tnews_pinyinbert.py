import os
import json
import logging
import argparse
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from torch.optim import AdamW
from torch.amp import autocast, GradScaler
from transformers import get_linear_schedule_with_warmup, BertModel
from safetensors.torch import load_file

from configs.pinyin_bert_config import PinyinBertConfig
from vocabs.pinyin_tokenizer import PinyinTokenizer

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

# Số lớp lấy từ số label trong labels.json
NUM_LABELS = 15

class PinyinBertForSequenceClassification(nn.Module):
    def __init__(self, config, num_labels=NUM_LABELS):
        super().__init__()
        self.num_labels = num_labels
        self.hidden_size = config.hidden_size
        self.shared_embeddings = nn.Embedding(
            config.vocab_size, 
            self.hidden_size,
            padding_idx=config.pad_token_id
        )
        self.fc_emb = nn.Linear(self.hidden_size * 3, self.hidden_size)
        self.bert = BertModel(config, add_pooling_layer=False)
        self.bert.embeddings.word_embeddings = None
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, num_labels)

        self.classifier.weight.data.normal_(mean=0.0, std=config.initializer_range)
        if self.classifier.bias is not None:
            self.classifier.bias.data.zero_()

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None):
        batch_size = input_ids.size(0)
        seq_len = input_ids.size(1)
        onset_ids = input_ids[:, :, 0]
        rhyme_ids = input_ids[:, :, 1]
        tone_ids = input_ids[:, :, 2]
        onset_emb = self.shared_embeddings(onset_ids)
        rhyme_emb = self.shared_embeddings(rhyme_ids)
        tone_emb = self.shared_embeddings(tone_ids)
        pinyin_emb = torch.cat([onset_emb, rhyme_emb, tone_emb], dim=-1)
        inputs_embeds = self.fc_emb(pinyin_emb)
        outputs = self.bert(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            token_type_ids=None, 
        )
        sequence_output = outputs.last_hidden_state
        cls_output = sequence_output[:, 0, :]
        cls_output = self.dropout(cls_output)
        logits = self.classifier(cls_output)
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits, labels)
        return (loss, logits) if loss is not None else logits

def load_label_map(label_path):
    label2id = {}

    with open(label_path, 'r', encoding='utf-8') as f:
        for idx, line in enumerate(f):
            item = json.loads(line)
            label = item["label"]
            label2id[label] = idx

    return label2id

def load_tnews_test_data(data_path):
    examples = []
    with open(data_path, 'r', encoding='utf-8') as f:
        for line in f:
            item = json.loads(line)
            # [SỬA ĐỔI]: Ghép keywords vào sau sentence
            sentence = item["sentence"]
            keywords = item.get("keywords", "")
            
            if keywords:
                # Đổi dấu phẩy thành khoảng trắng để PinyinBert dễ đọc hơn
                keywords = keywords.replace(",", " ")
                sentence = f"{sentence} {keywords}"
            
            item["sentence"] = sentence
            examples.append(item)
    return examples

def convert_examples_to_features(examples, tokenizer, max_seq_length):
    cls_token = (tokenizer.config.cls_token_id,) * 3
    pad_token = (tokenizer.config.pad_token_id,) * 3
    sep_token = (tokenizer.config.empty_token_id,) * 3
    features = []
    for example in tqdm(examples, desc="Converting features"):
        tokens = tokenizer.encode(example["sentence"]).tolist()[1:]
        # Truncate if too long
        if len(tokens) > max_seq_length - 2:
            tokens = tokens[:max_seq_length - 2]
        input_ids = [cls_token] + tokens + [sep_token]
        token_type_ids = [0] * len(input_ids)
        attention_mask = [1] * len(input_ids)
        # Padding
        padding_length = max_seq_length - len(input_ids)
        input_ids += [pad_token] * padding_length
        token_type_ids += [0] * padding_length
        attention_mask += [0] * padding_length
        features.append({
            "id": example["id"],
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids,
        })
    return features

def create_dataloader(features, batch_size, is_training=True):
    all_ids = torch.tensor([f["id"] for f in features])
    all_input_ids = torch.tensor([f["input_ids"] for f in features], dtype=torch.long)
    all_attention_mask = torch.tensor([f["attention_mask"] for f in features], dtype=torch.long)
    all_token_type_ids = torch.tensor([f["token_type_ids"] for f in features], dtype=torch.long)
    dataset = TensorDataset(all_ids, all_input_ids, all_attention_mask, all_token_type_ids)
    sampler = RandomSampler(dataset) if is_training else SequentialSampler(dataset)
    return DataLoader(dataset, sampler=sampler, batch_size=batch_size)

def inference(model, dataloader, device, label2id: dict, label_path: str):
    id2label = {id: label for label, id in label2id.items()}
    label2desc = {}
    with open(label_path, 'r', encoding='utf-8') as file:
        for line in file:
            label_inf = json.loads(line)
            label2desc[label_inf["label"]] = label_inf["label_desc"]

    model.eval()
    results = {}
    for batch in tqdm(dataloader, desc="Inferring"):
        batch = tuple(t.to(device) for t in batch)
        ids, input_ids, attention_mask, token_type_ids = batch
        with torch.no_grad():
            logits = model(input_ids, token_type_ids, attention_mask)
        preds = torch.argmax(logits, dim=1).tolist()
        ids = ids.tolist()
        for id, pred in zip(ids, preds):
            results[id] = label2desc[id2label[pred]]

    return results

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)

    logger.info(f"Loading Config từ: {args.init_checkpoint}")
    config_path = os.path.join(args.init_checkpoint, "config.json")
    weights_path = os.path.join(args.init_checkpoint, "model.safetensors")
    label_path = os.path.join(args.data_dir, "labels.json")
    best_checkpoint = args.best_checkpoint

    label2id = load_label_map(label_path)
    num_labels = len(label2id)

    with open(config_path, "r", encoding="utf-8") as f:
        config_dict = json.load(f)

    config = PinyinBertConfig(**config_dict)
    model = PinyinBertForSequenceClassification(config, num_labels=num_labels).to(device)

    logger.info(f"Loading Weights từ: {weights_path}")
    state_dict = torch.load(best_checkpoint)
    model.load_state_dict(state_dict, strict=True)

    tokenizer = PinyinTokenizer(config)

    logger.info("Đang đọc dữ liệu TNEWS...")
    test_examples = load_tnews_test_data(os.path.join(args.data_dir, "test.json"))

    logger.info(f"Số lượng test: {len(test_examples)}")
    test_features = convert_examples_to_features(
        test_examples, tokenizer, args.max_seq_length, label2id
    )

    test_dataloader = create_dataloader(test_features, args.eval_batch_size, False)

    results = inference(model, test_dataloader, device, label2id, label_path)
    json.dump(results, open("tnews-results.json", "w+"), ensure_ascii=False, indent=4)
    

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True, help="Thư mục chứa dữ liệu TNEWS (train.json, dev.json, labels.json)")
    parser.add_argument("--init_checkpoint", type=str, required=True, help="Thư mục weights gốc từ quá trình Pre-train (chứa config.json)")
    parser.add_argument("--best_checkpoint", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="./tnews_outputs", help="Nơi lưu checkpoint mô hình")
    parser.add_argument("--max_seq_length", type=int, default=128, help="Độ dài chuỗi tối đa")
    parser.add_argument("--train_batch_size", type=int, default=16, help="Batch size huấn luyện")
    parser.add_argument("--eval_batch_size", type=int, default=32, help="Batch size đánh giá")
    parser.add_argument("--learning_rate", type=float, default=3e-5, help="Learning rate")
    parser.add_argument("--epochs", type=int, default=5, help="Số epoch")
    args = parser.parse_args()
    train(args)

if __name__ == "__main__":
    main()
