import argparse
import json
import logging
import os
import random
import numpy as np
from tqdm import tqdm
from sklearn.metrics import f1_score, accuracy_score
from sklearn.utils.class_weight import compute_class_weight

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from torch.optim import AdamW
from torch.amp import autocast, GradScaler
from transformers import get_linear_schedule_with_warmup, BertModel
from safetensors.torch import load_file

from configs.viphon_bert_config import ViPhonBertConfig
from vocabs.viphon_tokenizer import ViPhonTokenizer

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

# ==========================================
# THIẾT LẬP 7 NHÃN CẢM XÚC CHO UIT-VSMEC
# ==========================================
LABEL_MAP = {
    "Enjoyment": 0,
    "Sadness": 1,
    "Anger": 2,
    "Fear": 3,
    "Disgust": 4,
    "Surprise": 5,
    "Other": 6
}
NUM_LABELS = len(LABEL_MAP)

def set_seed(seed: int):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    logger.info(f"👉 Đã thiết lập mã Seed cố định: {seed}")


# ==========================================
# 1. KIẾN TRÚC MÔ HÌNH (CÓ TRỌNG SỐ LOSS)
# ==========================================
class ViPhonBertForEmotionClassification(nn.Module):
    def __init__(self, config, num_labels, class_weights=None):
        super().__init__() 
        self.num_labels = num_labels
        self.hidden_size = config.hidden_size
        self.class_weights = class_weights # Trọng số cho từng class
        
        self.shared_embeddings = nn.Embedding(
            config.vocab_size, 
            self.hidden_size,
            padding_idx=config.pad_token_id
        )
        self.fc_emb = nn.Linear(self.hidden_size * 3, self.hidden_size)
        
        self.bert = BertModel(config, add_pooling_layer=False)
        self.bert.embeddings.word_embeddings = None

        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        
        # [GIẢI PHÁP 2]: Đơn giản hóa head để tránh overfit
        self.classifier = nn.Linear(config.hidden_size, num_labels)
        self.classifier.weight.data.normal_(mean=0.0, std=config.initializer_range)
        if self.classifier.bias is not None:
            self.classifier.bias.data.zero_()

    def forward(self, input_ids, attention_mask=None, labels=None):
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
        )

        sequence_output = outputs.last_hidden_state
        
        # Mean Pooling
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(sequence_output.size()).float()
        sum_embeddings = torch.sum(sequence_output * input_mask_expanded, 1)
        sum_mask = input_mask_expanded.sum(1)
        sum_mask = torch.clamp(sum_mask, min=1e-9)
        mean_pooled_output = sum_embeddings / sum_mask
        
        cls_output = self.dropout(mean_pooled_output)
        logits = self.classifier(cls_output)

        loss = None
        if labels is not None:
            # [GIẢI PHÁP 1]: Áp dụng Class Weights vào CrossEntropy
            loss_fct = nn.CrossEntropyLoss(weight=self.class_weights, label_smoothing=0.05)
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        return loss, logits

def load_vsmec_data(data_path):
    examples = []
    if not os.path.exists(data_path):
        return examples

    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        for key, val in data.items():
            sentence = val.get("sentence", "").strip()
            emotion = val.get("emotion", "").strip()
            if not sentence or not emotion: continue
                
            emotion = emotion.capitalize()
            if emotion in LABEL_MAP:
                examples.append({
                    "id": key,
                    "text": sentence,
                    "label": LABEL_MAP[emotion]
                })
    return examples

def convert_examples_to_features(examples, tokenizer, max_seq_length):
    pad_token = (tokenizer.config.pad_token_id,) * 3
    features = []
    for example in tqdm(examples, desc="Converting features"):
        input_ids = tokenizer.encode(example["text"]).tolist()[:max_seq_length]
        attention_mask = [1] * len(input_ids)
        
        padding_length = max_seq_length - len(input_ids)
        if padding_length > 0:
            input_ids += [pad_token] * padding_length
            attention_mask += [0] * padding_length

        features.append({
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "label_id": example["label"]
        })
    return features

def create_dataloader(features, batch_size, is_training=True):
    if not features: return None
    all_input_ids = torch.tensor([f["input_ids"] for f in features], dtype=torch.long)
    all_attention_mask = torch.tensor([f["attention_mask"] for f in features], dtype=torch.long)
    all_label_ids = torch.tensor([f["label_id"] for f in features], dtype=torch.long)

    dataset = TensorDataset(all_input_ids, all_attention_mask, all_label_ids)
    sampler = RandomSampler(dataset) if is_training else SequentialSampler(dataset)
    return DataLoader(dataset, sampler=sampler, batch_size=batch_size)

def evaluate(model, dataloader, device):
    if dataloader is None: return 0, 0, 0
    model.eval()
    eval_loss, nb_eval_steps = 0, 0
    all_preds, all_labels = [], []

    for batch in dataloader:
        batch = tuple(t.to(device) for t in batch)
        input_ids, attention_mask, labels = batch

        with torch.no_grad():
            loss, logits = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)

        eval_loss += loss.item()
        all_preds.extend(torch.argmax(logits, dim=-1).detach().cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        nb_eval_steps += 1

    eval_loss = eval_loss / nb_eval_steps if nb_eval_steps > 0 else 0
    acc = accuracy_score(all_labels, all_preds) if len(all_labels) > 0 else 0
    
    import warnings
    from sklearn.exceptions import UndefinedMetricWarning
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UndefinedMetricWarning)
        macro_f1 = f1_score(all_labels, all_preds, average='macro') if len(all_labels) > 0 else 0
    
    return eval_loss, acc, macro_f1


def train(args):
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)
    
    logger.info(f"Loading Config từ: {args.init_checkpoint}")
    with open(os.path.join(args.init_checkpoint, "config.json"), "r", encoding="utf-8") as f:
        config = ViPhonBertConfig(**json.load(f))
    tokenizer = ViPhonTokenizer(config) 
    
    # ---------------------------------------------
    # LOAD DỮ LIỆU VÀ TÍNH TOÁN CLASS WEIGHTS
    # ---------------------------------------------
    train_examples = load_vsmec_data(os.path.join(args.data_dir, "train.json"))
    dev_examples = load_vsmec_data(os.path.join(args.data_dir, "dev.json"))
    test_examples = load_vsmec_data(os.path.join(args.data_dir, "test.json"))

    # Tính trọng số cho từng class dựa trên tập Train
    train_labels = [ex["label"] for ex in train_examples]
    classes = np.arange(NUM_LABELS)
    cw_numpy = compute_class_weight(class_weight='balanced', classes=classes, y=train_labels)
    class_weights_tensor = torch.tensor(cw_numpy, dtype=torch.float).to(device)
    
    logger.info(f"⚖️ Đã tự động kích hoạt Balance Class Weights: {cw_numpy}")

    # Khởi tạo model với class_weights
    model = ViPhonBertForEmotionClassification(
        config, 
        num_labels=NUM_LABELS, 
        class_weights=class_weights_tensor
    ).to(device)
    
    weights_path = os.path.join(args.init_checkpoint, "model.safetensors")
    if not os.path.exists(weights_path):
        weights_path = os.path.join(args.init_checkpoint, "pytorch_model.bin")
    if os.path.exists(weights_path):
        state_dict = load_file(weights_path) if weights_path.endswith('.safetensors') else torch.load(weights_path, map_location=device)
        model.load_state_dict(state_dict, strict=False)

    train_features = convert_examples_to_features(train_examples, tokenizer, args.max_seq_length)
    dev_features = convert_examples_to_features(dev_examples, tokenizer, args.max_seq_length)
    test_features = convert_examples_to_features(test_examples, tokenizer, args.max_seq_length)

    train_dataloader = create_dataloader(train_features, args.train_batch_size, is_training=True)
    dev_dataloader = create_dataloader(dev_features, args.eval_batch_size, is_training=False)
    test_dataloader = create_dataloader(test_features, args.eval_batch_size, is_training=False)

    t_total = len(train_dataloader) * args.epochs
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(0.1 * t_total), num_training_steps=t_total)
    scaler = GradScaler('cuda') 

    logger.info("***** Bắt đầu tiến trình Huấn luyện VSMEC (BALANCED) *****")
    best_f1 = 0.0
    best_model_path = os.path.join(args.output_dir, "best_model_vsmec.pt")

    for epoch in range(args.epochs):
        model.train()
        with tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{args.epochs}") as pbar:
            for batch in pbar:
                batch = tuple(t.to(device) for t in batch)
                input_ids, attention_mask, labels = batch
                optimizer.zero_grad()
                with autocast('cuda'):
                    loss, _ = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                pbar.set_postfix({"Loss": f"{loss.item():.4f}"})

        eval_loss, eval_acc, eval_f1 = evaluate(model, dev_dataloader, device)
        logger.info(f"Epoch {epoch+1} - Loss: {eval_loss:.4f} - Acc: {eval_acc*100:.2f}% - Macro F1: {eval_f1*100:.2f}%")

        if eval_f1 > best_f1:
            best_f1 = eval_f1
            torch.save(model.state_dict(), best_model_path)
            logger.info(f"✨ LƯU KỶ LỤC TỐI ƯU MỚI: Macro F1 = {best_f1*100:.2f}% tại {best_model_path}")

    logger.info("==================================================")
    if test_dataloader is not None and os.path.exists(best_model_path):
        model.load_state_dict(torch.load(best_model_path, map_location=device))
        test_loss, test_acc, test_f1 = evaluate(model, test_dataloader, device)
        logger.info(f"🎯 KẾT QUẢ TEST CUỐI CÙNG VSMEC - Loss: {test_loss:.4f}")
        logger.info(f"🎯 Accuracy: {test_acc*100:.2f}%")
        logger.info(f"🎯 Macro F1: {test_f1*100:.2f}%")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--init_checkpoint", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="./vsmec_outputs")
    parser.add_argument("--max_seq_length", type=int, default=128)
    parser.add_argument("--train_batch_size", type=int, default=32)
    parser.add_argument("--eval_batch_size", type=int, default=32)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    # Khuyến nghị chỉ nên train 10-15 Epochs, 30 là quá overfit.
    parser.add_argument("--epochs", type=int, default=15) 
    parser.add_argument("--seed", type=int, default=42)
    
    args = parser.parse_args()
    train(args)

if __name__ == "__main__":
    main()