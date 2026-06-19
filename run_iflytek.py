import argparse
import json
import logging
import os
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from torch.optim import AdamW
from torch.amp import autocast, GradScaler
from transformers import get_linear_schedule_with_warmup, BertModel
from safetensors.torch import load_file

from configs.pinyin_bert_config import PinyinBertConfig
from vocabs.pinyin_tokenizer import PinyinTokenizer

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

NUM_LABELS = 119


class PinyinBertForSequenceClassification(nn.Module):
    """
    Kiến trúc mô hình PinyinBert cho bài toán Phân loại văn bản (Sequence Classification).
    Kế thừa cơ chế Shared Embedding cấp độ âm vị (Onset, Rhyme, Tone).
    """
    def __init__(self, config, num_labels=119):
        super().__init__()
        self.num_labels = num_labels
        self.hidden_size = config.hidden_size
        
        # SHARED EMBEDDING CHO CẢ 3 THÀNH PHẦN NGỮ ÂM
        self.shared_embeddings = nn.Embedding(
            config.vocab_size, 
            self.hidden_size,
            padding_idx=config.pad_token_id
        )
        self.fc_emb = nn.Linear(self.hidden_size * 3, self.hidden_size)
        
        # BERT ENCODER
        self.bert = BertModel(config, add_pooling_layer=False)
        self.bert.embeddings.word_embeddings = None  # Loại bỏ word embedding mặc định của chữ Hán

        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, num_labels)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None):
        # input_ids shape: (batch_size, seq_len, 3)
        batch_size = input_ids.size(0)
        seq_len = input_ids.size(1)

        # Tách chuỗi tensor thành 3 thành phần Onset, Rhyme, Tone
        onset_ids = input_ids[:, :, 0]
        rhyme_ids = input_ids[:, :, 1]
        tone_ids = input_ids[:, :, 2]

        # Trích xuất Embedding tương ứng
        onset_emb = self.shared_embeddings(onset_ids)
        rhyme_emb = self.shared_embeddings(rhyme_ids)
        tone_emb = self.shared_embeddings(tone_ids)

        # Nối và chiếu về không gian hidden_size chuẩn của BERT
        pinyin_emb = torch.cat([onset_emb, rhyme_emb, tone_emb], dim=-1)
        inputs_embeds = self.fc_emb(pinyin_emb)

        # Đưa qua các lớp Transformer Layers
        outputs = self.bert(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )

        sequence_output = outputs.last_hidden_state
        cls_output = sequence_output[:, 0, :]  # Lấy đặc trưng của token đại diện [CLS]
        
        cls_output = self.dropout(cls_output)
        logits = self.classifier(cls_output)  # Shape: (batch_size, num_labels)

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits, labels)

        return loss, logits


def load_iflytek_data(data_path, is_training=True):
    """
    Đọc dữ liệu định dạng JSON lines từ tập IFLYTEK.
    """
    examples = []
    if not os.path.exists(data_path):
        logger.warning(f"Tệp dữ liệu không tồn tại: {data_path}")
        return examples

    with open(data_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            
            # Tập test sẽ không chứa trường 'label'
            label = int(data["label"]) if "label" in data else -1
            sentence = data.get("sentence", "")
            
            if sentence:
                examples.append({
                    "sentence": sentence,
                    "label": label
                })
    return examples


def convert_examples_to_features(examples, tokenizer, max_seq_length):
    """
    Chuyển đổi văn bản thô sang cấu trúc Pinyin IDs Tokenizer đồng bộ với Pre-train.
    """
    cls_token = (tokenizer.config.cls_token_id,) * 3
    pad_token = (tokenizer.config.pad_token_id,) * 3
    sep_token = (tokenizer.config.empty_token_id,) * 3  # Dấu ngăn cách theo convention cấu trúc của bạn

    features = []
    for example in tqdm(examples, desc="Converting features"):
        # tokenizer.encode tự động chèn CLS ở đầu, tiến hành loại bỏ để gộp thủ công đúng chuẩn
        tokens_sentence = tokenizer.encode(example["sentence"]).tolist()[1:]

        # Kiểm soát độ dài nghiêm ngặt (Trừ đi 2 chỗ trống cho [CLS] và [SEP])
        if len(tokens_sentence) > max_seq_length - 2:
            tokens_sentence = tokens_sentence[:max_seq_length - 2]

        # Ghép chuỗi hoàn chỉnh cấu trúc: [CLS] Sentence [SEP]
        input_ids = [cls_token] + tokens_sentence + [sep_token]
        
        # Thiết lập Segment IDs (Tất cả bằng 0 đối với bài toán phân loại đơn câu)
        token_type_ids = [0] * len(input_ids)
        attention_mask = [1] * len(input_ids)

        # Thực hiện Padding an toàn cuối chuỗi
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

    eval_loss = eval_loss / nb_eval_steps
    eval_accuracy = eval_accuracy / nb_eval_examples
    return eval_loss, eval_accuracy


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)
    
    logger.info(f"Loading Config từ: {args.init_checkpoint}")
    config_path = os.path.join(args.init_checkpoint, "config.json")
    weights_path = os.path.join(args.init_checkpoint, "model.safetensors")
    if not os.path.exists(weights_path):
        weights_path = os.path.join(args.init_checkpoint, "pytorch_model.bin")
    
    with open(config_path, "r", encoding="utf-8") as f:
        config_dict = json.load(f)
    config = PinyinBertConfig(**config_dict)
    
    # Khởi tạo mô hình phân loại đầu ra 119 lớp
    model = PinyinBertForSequenceClassification(config, num_labels=NUM_LABELS).to(device)
    
    logger.info(f"Loading Weights từ: {weights_path}")
    if weights_path.endswith('.safetensors'):
        state_dict = load_file(weights_path)
    else:
        state_dict = torch.load(weights_path, map_location=device)
        
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    logger.info(f"Missing keys (Khởi tạo mới ngẫu nhiên cho Classifier Head): {missing_keys}")
    logger.info(f"Unexpected keys (Bỏ qua từ tầng Pre-train MLM): {unexpected_keys}")

    tokenizer = PinyinTokenizer(config) 

    # ĐỌC VÀ CHUYỂN ĐỔI DỮ LIỆU
    logger.info("Đang đọc dữ liệu IFLYTEK...")
    train_examples = load_iflytek_data(os.path.join(args.data_dir, "train.json"), is_training=True)
    dev_examples = load_iflytek_data(os.path.join(args.data_dir, "dev.json"), is_training=True)

    logger.info(f"Số lượng mẫu huấn luyện: {len(train_examples)} | Số lượng mẫu kiểm thử: {len(dev_examples)}")

    train_features = convert_examples_to_features(train_examples, tokenizer, args.max_seq_length)
    dev_features = convert_examples_to_features(dev_examples, tokenizer, args.max_seq_length)

    train_dataloader = create_dataloader(train_features, args.train_batch_size, is_training=True)
    dev_dataloader = create_dataloader(dev_features, args.eval_batch_size, is_training=False)

    # CẤU HÌNH OPTIMIZER & SCHEDULER
    t_total = len(train_dataloader) * args.epochs
    optimizer = AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.01)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(0.1 * t_total), num_training_steps=t_total)
    scaler = GradScaler('cuda') 

    logger.info("***** Bắt đầu tiến trình Huấn luyện *****")
    best_acc = 0.0

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        with tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{args.epochs}") as pbar:
            for batch in pbar:
                batch = tuple(t.to(device) for t in batch)
                input_ids, attention_mask, token_type_ids, labels = batch

                optimizer.zero_grad()

                # Tối ưu hóa AMP hỗn hợp tăng tốc độ tính toán phần cứng
                with autocast('cuda'):
                    loss, logits = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        token_type_ids=token_type_ids,
                        labels=labels
                    )

                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                
                # Gradient Clipping chống bùng nổ đạo hàm
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()

                total_loss += loss.item()
                pbar.set_postfix({"Loss": f"{loss.item():.4f}"})

        # ĐÁNH GIÁ CHẤT LƯỢNG SAU MỖI EPOCH
        logger.info(f"***** Chạy Đánh giá trên tập Dev (Epoch {epoch+1}) *****")
        eval_loss, eval_acc = evaluate(model, dev_dataloader, device)
        logger.info(f"Epoch {epoch+1} - Train Loss: {total_loss/len(train_dataloader):.4f} - Eval Loss: {eval_loss:.4f} - Eval Acc: {eval_acc*100:.2f}%")

        # Lưu lại checkpoint tốt nhất dựa trên độ chính xác
        if eval_acc > best_acc:
            best_acc = eval_acc
            save_path = os.path.join(args.output_dir, "best_model_iflytek.pt")
            torch.save(model.state_dict(), save_path)
            logger.info(f"✨ Đã lưu mô hình tối ưu mới nhất với Accuracy: {best_acc*100:.2f}% tại {save_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True, help="Thư mục chứa dữ liệu IFLYTEK (train.json, dev.json)")
    parser.add_argument("--init_checkpoint", type=str, required=True, help="Thư mục weights gốc từ quá trình Pre-train (chứa config.json)")
    parser.add_argument("--output_dir", type=str, default="./iflytek_outputs", help="Nơi lưu checkpoint mô hình")
    
    parser.add_argument("--max_seq_length", type=int, default=256, help="Độ dài chuỗi tối đa cấu hình cắt/đệm")
    parser.add_argument("--train_batch_size", type=int, default=16, help="Kích thước batch size huấn luyện")
    parser.add_argument("--eval_batch_size", type=int, default=32, help="Kích thước batch size đánh giá")
    parser.add_argument("--learning_rate", type=float, default=3e-5, help="Tốc độ học tối ưu ban đầu")
    parser.add_argument("--epochs", type=int, default=5, help="Số lượng lượt huấn luyện toàn tập dữ liệu")
    
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()