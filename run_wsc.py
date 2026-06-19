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

NUM_LABELS = 2  # true/false

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
            token_type_ids=token_type_ids,
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

def load_wsc_data(data_path):
    label_map = {"false": 0, "true": 1}
    examples = []
    with open(data_path, 'r', encoding='utf-8') as f:
        for line in f:
            item = json.loads(line)
            text = item["text"]
            span1 = item["target"]["span1_text"]
            span2 = item["target"]["span2_text"]
            label = label_map[item["label"]]
            # Nối text + [SEP] + span1 + [SEP] + span2
            example_text = f"{text} [SEP] {span1} [SEP] {span2}"
            examples.append({
                "text": example_text,
                "label": label
            })
    return examples

def convert_examples_to_features(examples, tokenizer, max_seq_length):
    cls_token = (tokenizer.config.cls_token_id,) * 3
    pad_token = (tokenizer.config.pad_token_id,) * 3
    sep_token = (tokenizer.config.empty_token_id,) * 3
    features = []
    for example in tqdm(examples, desc="Converting features"):
        tokens = tokenizer.encode(example["text"]).tolist()[1:]
        if len(tokens) > max_seq_length - 2:
            tokens = tokens[:max_seq_length - 2]
        input_ids = [cls_token] + tokens + [sep_token]
        token_type_ids = [0] * len(input_ids)
        attention_mask = [1] * len(input_ids)
        padding_length = max_seq_length - len(input_ids)
        input_ids += [pad_token] * padding_length
        token_type_ids += [0] * padding_length
        attention_mask += [0] * padding_length
        features.append({
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids,
            "label": example["label"]
        })
    return features

def create_dataloader(features, batch_size, is_training=True):
    all_input_ids = torch.tensor([f["input_ids"] for f in features], dtype=torch.long)
    all_attention_mask = torch.tensor([f["attention_mask"] for f in features], dtype=torch.long)
    all_token_type_ids = torch.tensor([f["token_type_ids"] for f in features], dtype=torch.long)
    all_labels = torch.tensor([f["label"] for f in features], dtype=torch.long)
    dataset = TensorDataset(all_input_ids, all_attention_mask, all_token_type_ids, all_labels)
    sampler = RandomSampler(dataset) if is_training else SequentialSampler(dataset)
    return DataLoader(dataset, sampler=sampler, batch_size=batch_size)

def evaluate(model, dataloader, device):
    model.eval()
    eval_loss, eval_accuracy = 0, 0
    nb_eval_steps, nb_eval_examples = 0, 0
    for batch in dataloader:
        batch = tuple(t.to(device) for t in batch)
        input_ids, attention_mask, token_type_ids, labels = batch
        with torch.no_grad():
            loss, logits = model(input_ids, token_type_ids, attention_mask, labels)
        preds = torch.argmax(logits, dim=1)
        eval_loss += loss.item()
        eval_accuracy += (preds == labels).sum().item()
        nb_eval_examples += input_ids.size(0)
        nb_eval_steps += 1
    eval_loss = eval_loss / nb_eval_steps
    eval_accuracy = eval_accuracy / nb_eval_examples
    return eval_loss, eval_accuracy

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)
    logger.info(f"Loading Config từ: {args.init_checkpoint}")
    config_path = os.path.join(args.init_checkpoint, "config.json")
    weights_path = os.path.join(args.init_checkpoint, "model.safetensors")
    with open(config_path, "r", encoding="utf-8") as f:
        config_dict = json.load(f)
    config = PinyinBertConfig(**config_dict)
    model = PinyinBertForSequenceClassification(config, num_labels=NUM_LABELS).to(device)
    logger.info(f"Loading Weights từ: {weights_path}")
    state_dict = load_file(weights_path)
    model.load_state_dict(state_dict, strict=False)
    tokenizer = PinyinTokenizer(config)
    logger.info("Đang đọc dữ liệu CLUEWSC2020...")
    train_examples = load_wsc_data(os.path.join(args.data_dir, "train.json"))
    dev_examples = load_wsc_data(os.path.join(args.data_dir, "dev.json"))
    logger.info(f"Số lượng mẫu huấn luyện: {len(train_examples)} | Số lượng mẫu kiểm thử: {len(dev_examples)}")
    train_features = convert_examples_to_features(train_examples, tokenizer, args.max_seq_length)
    dev_features = convert_examples_to_features(dev_examples, tokenizer, args.max_seq_length)
    train_dataloader = create_dataloader(train_features, args.train_batch_size, is_training=True)
    dev_dataloader = create_dataloader(dev_features, args.eval_batch_size, is_training=False)
    t_total = len(train_dataloader) * args.epochs
    optimizer = AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.01)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(0.1 * t_total), num_training_steps=t_total)
    scaler = GradScaler('cuda')
    logger.info("***** Bắt đầu tiến trình Huấn luyện *****")
    best_acc = 0.0

    for epoch in range(args.epochs):
        model.train()

        progress_bar = tqdm(
            train_dataloader,
            desc=f"Epoch {epoch+1}/{args.epochs}"
        )

        for step, batch in enumerate(progress_bar):
            batch = tuple(t.to(device) for t in batch)
            input_ids, attention_mask, token_type_ids, labels = batch

            optimizer.zero_grad()

            with autocast(device_type='cuda'):
                loss, logits = model(
                    input_ids,
                    token_type_ids,
                    attention_mask,
                    labels
                )

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            progress_bar.set_postfix(loss=f"{loss.item():.4f}")

        # ===== EVAL =====
        eval_loss, eval_acc = evaluate(model, dev_dataloader, device)

        logger.info(
            f"Epoch {epoch+1}/{args.epochs} | "
            f"Eval Loss: {eval_loss:.4f} | "
            f"Eval Acc: {eval_acc*100:.2f}%"
        )

        # ===== SAVE BEST MODEL =====
        if eval_acc > best_acc:
            best_acc = eval_acc
            save_path = os.path.join(args.output_dir, "best_model_wsc.pt")

            torch.save(model.state_dict(), save_path)

            logger.info(
                f"✨ Đã lưu model tốt nhất (Acc: {best_acc*100:.2f}%) tại: {save_path}"
            )

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True, help="Thư mục chứa dữ liệu CLUEWSC2020")
    parser.add_argument("--init_checkpoint", type=str, required=True, help="Thư mục weights gốc từ quá trình Pre-train (chứa config.json)")
    parser.add_argument("--output_dir", type=str, default="./cluewsc2020_outputs", help="Nơi lưu checkpoint mô hình")
    parser.add_argument("--max_seq_length", type=int, default=128, help="Độ dài chuỗi tối đa")
    parser.add_argument("--train_batch_size", type=int, default=16, help="Batch size huấn luyện")
    parser.add_argument("--eval_batch_size", type=int, default=32, help="Batch size đánh giá")
    parser.add_argument("--learning_rate", type=float, default=3e-5, help="Learning rate")
    parser.add_argument("--epochs", type=int, default=10, help="Số epoch")
    args = parser.parse_args()
    train(args)

if __name__ == "__main__":
    main()
