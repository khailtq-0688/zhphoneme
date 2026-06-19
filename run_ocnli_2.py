import argparse
import json
import logging
import os
import random  # Thêm thư viện random của Python
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from torch.optim import AdamW
from torch.amp import autocast, GradScaler
from transformers import get_linear_schedule_with_warmup, BertModel
from safetensors.torch import load_file

from configs.pinyin_bert_config import PinyinBertConfig
from vocabs.pinyin_tokenizer import PinyinTokenizer

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

# OCNLI có 3 nhãn tiêu chuẩn: entailment, neutral, contradiction
LABEL_MAP = {"entailment": 0, "neutral": 1, "contradiction": 2}
INV_LABEL_MAP = {0: "entailment", 1: "neutral", 2: "contradiction"}
NUM_LABELS = 3


def set_seed(seed: int):
    """
    Hàm cố định seed để đảm bảo tính tái lặp của kết quả thử nghiệm.
    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # Nếu dùng multi-GPU
    
    # Các thiết lập giúp đảm bảo tính đồng nhất trên cấu trúc CUDA
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    logger.info(f"👉 Đã thiết lập mã Seed cố định: {seed}")


class PinyinBertForSequenceClassification(nn.Module):
    """
    Kiến trúc PinyinBert cho bài toán Phân loại cặp câu (Text Pair Classification).
    Kế thừa cơ chế Shared Embedding cấp độ âm vị (Onset, Rhyme, Tone).
    """
    def __init__(self, config, num_labels=3):
        super().__init__()
        self.num_labels = num_labels
        self.hidden_size = config.hidden_size
        
        self.shared_embeddings = nn.Embedding(
            config.vocab_size, 
            self.hidden_size,
            padding_idx=config.pad_token_id
        )
        self.fc_emb = nn.Linear(self.hidden_size * 3, self.hidden_size)
        
        # TỐI ƯU: Bật lại add_pooling_layer=True
        self.bert = BertModel(config)
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
            token_type_ids=token_type_ids,
        )

        # TỐI ƯU: Sử dụng pooler_output thay vì tự cắt sequence_output[:, 0, :]
        pooled_output = outputs.pooler_output
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits, labels)

        return loss, logits


def load_ocnli_data(data_path, is_training=True):
    examples = []
    if not os.path.exists(data_path):
        logger.warning(f"Tệp dữ liệu không tồn tại: {data_path}")
        return examples

    with open(data_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                s1 = data.get("sentence1", "")
                s2 = data.get("sentence2", "")
                

                if is_training:
                    label_str = data.get("label", "-")
                    label = LABEL_MAP[label_str]
                else:
                    label = -1 
                    
                examples.append({
                    "sentence1": s1,
                    "sentence2": s2,
                    "label": label
                })
            except Exception:
                continue
    return examples

def convert_examples_to_features(examples, tokenizer, max_seq_length, is_training=False):
    cls_token = (tokenizer.config.cls_token_id,) * 3
    pad_token = (tokenizer.config.pad_token_id,) * 3
    sep_token = (tokenizer.config.empty_token_id,) * 3

    features = []
    for example in tqdm(examples, desc="Converting features"):
        tokens_s1 = tokenizer.encode(example["sentence1"]).tolist()[1:]
        tokens_s2 = tokenizer.encode(example["sentence2"]).tolist()[1:]

        input_ids = [cls_token] + tokens_s1 + [sep_token] + tokens_s2 + [sep_token]
        
        part1_len = 1 + len(tokens_s1) + 1 
        part2_len = len(tokens_s2) + 1
        
        token_type_ids = [0] * part1_len + [1] * part2_len
        attention_mask = [1] * len(input_ids)

        padding_length = max_seq_length - len(input_ids)
        if padding_length > 0:
            input_ids += [pad_token] * padding_length
            attention_mask += [0] * padding_length
            token_type_ids += [0] * padding_length

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
    
    # THÊM MỚI: Khởi tạo mảng lưu toàn bộ dự đoán và nhãn
    all_preds = []
    all_labels = []

    for batch in dataloader:
        batch = tuple(t.to(device) for t in batch)
        input_ids, attention_mask, token_type_ids, labels = batch

        with torch.no_grad():
            loss, logits = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                labels=labels
            )

        logits = logits.detach().cpu().numpy()
        label_ids = labels.to('cpu').numpy()
        
        preds = np.argmax(logits, axis=1)
        tmp_eval_accuracy = np.sum(preds == label_ids)

        eval_loss += loss.mean().item()
        eval_accuracy += tmp_eval_accuracy

        nb_eval_examples += input_ids.size(0)
        nb_eval_steps += 1
        
        # THÊM MỚI: Lưu lại dự đoán và nhãn của batch hiện tại
        all_preds.append(preds)
        all_labels.append(label_ids)

    eval_loss = eval_loss / nb_eval_steps
    eval_accuracy = eval_accuracy / nb_eval_examples
    
    # THÊM MỚI: Gộp danh sách các batch lại thành một mảng numpy phẳng (1 chiều)
    all_preds = np.concatenate(all_preds, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    
    # THÊM MỚI: Trả về thêm all_preds và all_labels
    return eval_loss, eval_accuracy, all_preds, all_labels


def train(args):
    # Kích hoạt seed ngay đầu hàm train
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)
    
    logger.info(f"Loading Config từ: {args.init_checkpoint}")
    config_path = os.path.join(args.init_checkpoint, "config.json")
    weights_path = os.path.join(args.init_checkpoint, "model.safetensors")
    if not os.path.exists(weights_path):
        weights_path = os.path.join(args.init_checkpoint, "pytorch_model.bin")
    
    with open(config_path, "r", encoding="utf-8") as f:
        config_dict = json.load(f)
    config_dict["type_vocab_size"] = 2
    config = PinyinBertConfig(**config_dict)
    
    model = PinyinBertForSequenceClassification(config, num_labels=NUM_LABELS).to(device)
    
    logger.info(f"Loading Weights từ: {weights_path}")
    if weights_path.endswith('.safetensors'):
        state_dict = load_file(weights_path)
    else:
        state_dict = torch.load(weights_path, map_location=device)

    target_key = "bert.embeddings.token_type_embeddings.weight"
    if target_key in state_dict:
        pretrained_weight = state_dict[target_key]
        if pretrained_weight.shape[0] == 1 and config.type_vocab_size == 2:
            logger.info("👉 Phát hiện checkpoint Pre-train có type_vocab_size = 1. Đang tự động đệm tensor lên 2...")
            
            new_weight = torch.zeros(2, config.hidden_size, dtype=pretrained_weight.dtype, device=pretrained_weight.device)
            new_weight[0] = pretrained_weight[0]
            new_weight[1].normal_(mean=0.0, std=config.initializer_range)
            state_dict[target_key] = new_weight
        
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    logger.info(f"Missing keys: {missing_keys}")
    logger.info(f"Unexpected keys: {unexpected_keys}")

    tokenizer = PinyinTokenizer(config) 

    # --- ĐOẠN ĐƯỢC THÊM: ĐỊNH NGHĨA PATH CACHE ---
    cache_train_path = os.path.join(args.data_dir, f"cached_ocnli_train_{args.max_seq_length}.pt")
    cache_dev_path = os.path.join(args.data_dir, f"cached_ocnli_dev_{args.max_seq_length}.pt")

    # --- XỬ LÝ CACHE CHO TẬP TRAIN ---
    if os.path.exists(cache_train_path):
        logger.info(f"👉 Tìm thấy cache dữ liệu OCNLI Train, đang tải từ: {cache_train_path}")
        train_features = torch.load(cache_train_path)
    else:
        logger.info("❌ Không tìm thấy cache Train. Bắt đầu convert từ file JSON gốc...")
        train_examples = load_ocnli_data(os.path.join(args.data_dir, "train.50k.json"), is_training=True)
        logger.info(f"Số lượng mẫu huấn luyện thô hợp lệ: {len(train_examples)}")
        train_features = convert_examples_to_features(train_examples, tokenizer, args.max_seq_length, is_training=True)
        logger.info(f"💾 Đang lưu dữ liệu Train đã convert vào cache: {cache_train_path}")
        torch.save(train_features, cache_train_path)

    # --- XỬ LÝ CACHE CHO TẬP DEV (ĐÃ ĐƯỢC CHỈNH SỬA TỐI ƯU) ---
    dev_file_path = os.path.join(args.data_dir, "dev.json")
    
    # GIẢI PHÁP LUÔN TẢI DỮ LIỆU THÔ: 
    # Việc đọc file JSON này rất nhanh, đảm bảo luôn có `dev_examples` cho quá trình xuất Bad Cases
    logger.info(f"📋 Đang nạp dữ liệu Dev gốc để phục vụ đối chiếu mẫu sai từ: {dev_file_path}")
    dev_examples = load_ocnli_data(dev_file_path, is_training=True)

    if os.path.exists(cache_dev_path):
        logger.info(f"👉 Tìm thấy cache dữ liệu OCNLI Dev, đang tải từ: {cache_dev_path}")
        dev_features = torch.load(cache_dev_path)
    else:
        logger.info("❌ Không tìm thấy cache Dev. Bắt đầu mã hóa từ dữ liệu thô...")
        logger.info(f"Số lượng mẫu kiểm thử thô hợp lệ: {len(dev_examples)}")
        dev_features = convert_examples_to_features(dev_examples, tokenizer, args.max_seq_length, is_training=False)
        logger.info(f"💾 Đang lưu dữ liệu Dev đã convert vào cache: {cache_dev_path}")
        torch.save(dev_features, cache_dev_path)

    logger.info(f"Số lượng mẫu features Huấn luyện: {len(train_features)} | Kiểm thử: {len(dev_features)}")

    train_dataloader = create_dataloader(train_features, args.train_batch_size, is_training=True)
    dev_dataloader = create_dataloader(dev_features, args.eval_batch_size, is_training=False)

    t_total = len(train_dataloader) * args.epochs
    
    # TỐI ƯU: Phân tách nhóm tham số cho AdamW
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {
            'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            'weight_decay': 0.01
        },
        {
            'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            'weight_decay': 0.0
        }
    ]
    
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(0.1 * t_total), num_training_steps=t_total)
    scaler = GradScaler('cuda') 

    logger.info("***** Bắt đầu tiến trình Huấn luyện OCNLI *****")
    best_acc = 0.0

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        with tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{args.epochs}") as pbar:
            for batch in pbar:
                batch = tuple(t.to(device) for t in batch)
                input_ids, attention_mask, token_type_ids, labels = batch

                optimizer.zero_grad()

                with autocast('cuda'):
                    loss, logits = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        token_type_ids=token_type_ids,
                        labels=labels
                    )

                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()

                total_loss += loss.item()
                pbar.set_postfix({"Loss": f"{loss.item():.4f}"})

        logger.info(f"***** Đánh giá chất lượng tập Dev (Epoch {epoch+1}) *****")
        # Gọi hàm đánh giá
        eval_loss, eval_acc, preds, out_label_ids = evaluate(model, dev_dataloader, device)
        
        logger.info(f"Epoch {epoch+1} - Eval Loss: {eval_loss:.4f} - Eval Acc: {eval_acc*100:.2f}%")

        if eval_acc > best_acc:
            best_acc = eval_acc
            save_path = os.path.join(args.output_dir, "best_model_ocnli.pt")
            torch.save(model.state_dict(), save_path)
            logger.info(f"✨ Đã lưu mô hình tối ưu mới nhất với Accuracy: {best_acc*100:.2f}% tại {save_path}")

            # =====================================================================
            # XỬ LÝ VÀ XUẤT MẪU SAI (Đã được bảo vệ an toàn bằng biến dev_examples toàn vẹn)
            # =====================================================================
            bad_cases = []
            for i, (pred, true_label) in enumerate(zip(preds, out_label_ids)):
                if pred != true_label:
                    # Đảm bảo ép kiểu sang kiểu int chuẩn của Python để tránh lỗi tương thích kiểu dữ liệu
                    idx = int(i)
                    wrong_sample = {
                        "id": idx,
                        "sentence1": dev_examples[idx]["sentence1"],
                        "sentence2": dev_examples[idx]["sentence2"],
                        "true_label": INV_LABEL_MAP.get(int(true_label), "unknown"),
                        "predicted_label": INV_LABEL_MAP.get(int(pred), "unknown")
                    }
                    bad_cases.append(wrong_sample)
            
            bad_cases_file = os.path.join(args.output_dir, "bad_cases_best_epoch.json")
            with open(bad_cases_file, "w", encoding="utf-8") as f:
                json.dump(bad_cases, f, ensure_ascii=False, indent=4)
                
            logger.info(f"🚨 Đã cập nhật và xuất thành công {len(bad_cases)} mẫu sai ra file: {bad_cases_file}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True, help="Thư mục chứa dữ liệu OCNLI (train.json, dev.json)")
    parser.add_argument("--init_checkpoint", type=str, required=True, help="Thư mục weights gốc pre-train (chứa config.json)")
    parser.add_argument("--output_dir", type=str, default="./ocnli_outputs", help="Thư mục lưu mô hình sau huấn luyện")
    
    parser.add_argument("--max_seq_length", type=int, default=128, help="Độ dài chuỗi tối đa tích hợp cặp câu")
    parser.add_argument("--train_batch_size", type=int, default=32, help="Kích thước batch size huấn luyện")
    parser.add_argument("--eval_batch_size", type=int, default=32, help="Kích thước batch size đánh giá")
    parser.add_argument("--learning_rate", type=float, default=2e-5, help="Tốc độ học tối ưu ban đầu")
    parser.add_argument("--epochs", type=int, default=5, help="Số lượng lượt huấn luyện toàn tập")
    parser.add_argument("--seed", type=int, default=42, help="Giá trị mã seed ngẫu nhiên cố định")
    
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()