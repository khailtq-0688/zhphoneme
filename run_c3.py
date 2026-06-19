import argparse
import json
import logging
import os
import random  # Thêm thư viện random của Python
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

NUM_CHOICES = 4


def set_seed(seed: int):
    """
    Hàm cố định seed để đảm bảo tính tái lặp của kết quả thử nghiệm đa lựa chọn.
    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # Nếu sử dụng hệ thống nhiều GPU
    
    # Thiết lập giúp đảm bảo tính đồng nhất thuật toán trên nền tảng CUDA
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    logger.info(f"👉 Đã thiết lập mã Seed cố định thành công: {seed}")


class PinyinBertForMultipleChoice(nn.Module):
    def __init__(self, config, num_choices=4):
        super().__init__()
        self.num_choices = num_choices
        self.hidden_size = config.hidden_size
        
        self.shared_embeddings = nn.Embedding(
            config.vocab_size, 
            self.hidden_size,
            padding_idx=config.pad_token_id
        )
        self.fc_emb = nn.Linear(self.hidden_size * 3, self.hidden_size)
        
        self.bert = BertModel(config, add_pooling_layer=False)
        self.bert.embeddings.word_embeddings = None
        
        self.pooler = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.Tanh()
        )

        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, 1)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, choice_mask=None, labels=None):
        batch_size = input_ids.size(0)
        seq_len = input_ids.size(2)

        input_ids = input_ids.view(-1, seq_len, 3)
        attention_mask = attention_mask.view(-1, seq_len) if attention_mask is not None else None
        token_type_ids = token_type_ids.view(-1, seq_len) if token_type_ids is not None else None

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
        cls_output = self.pooler(cls_output)
        
        cls_output = self.dropout(cls_output)
        logits = self.classifier(cls_output)
        
        reshaped_logits = logits.view(-1, self.num_choices)
        
        if choice_mask is not None:
            reshaped_logits = reshaped_logits + (1.0 - choice_mask) * -10000.0

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(reshaped_logits, labels)

        return loss, reshaped_logits


class C3Example(nn.Module):
    def __init__(self, guid, context, question, choices, label=None):
        super().__init__()
        self.guid = guid
        self.context = context
        self.question = question
        self.choices = choices
        self.label = label


def load_c3_data(data_path, is_training=True):
    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    examples = []
    for i, item in enumerate(data):
        context = " ".join(item[0])
        for j, q_dict in enumerate(item[1]):
            question = q_dict["question"]
            choices = q_dict["choice"]
            answer = q_dict.get("answer", "")

            while len(choices) < NUM_CHOICES:
                choices.append("")
            
            label = 0
            if is_training and answer:
                try:
                    label = choices.index(answer)
                except ValueError:
                    label = 0
                    
            guid = f"{i}-{j}"
            examples.append(C3Example(guid, context, question, choices, label))
            
    return examples


def truncate_seq_tuple(tokens_a, tokens_b, tokens_c, max_length):
    while True:
        total_length = len(tokens_a) + len(tokens_b) + len(tokens_c)
        if total_length <= max_length:
            break
        if len(tokens_a) >= len(tokens_b) and len(tokens_a) >= len(tokens_c):
            tokens_a.pop(0)
        elif len(tokens_b) >= len(tokens_a) and len(tokens_b) >= len(tokens_c):
            tokens_b.pop()
        else:
            tokens_c.pop()


def convert_examples_to_features(examples, tokenizer, max_seq_length):
    cls_token = (tokenizer.config.cls_token_id,) * 3
    pad_token = (tokenizer.config.pad_token_id,) * 3
    sep_token = (tokenizer.config.empty_token_id,) * 3 

    features = []
    for example in tqdm(examples, desc="Converting features"):
        choices_input_ids = []
        choices_attention_mask = []
        choices_token_type_ids = []
        choice_masks = []

        tokens_ctx = tokenizer.encode(example.context).tolist()[1:]
        tokens_q = tokenizer.encode(example.question).tolist()[1:]

        choices = example.choices[:4] 
        while len(choices) < 4:
            choices.append("")

        for choice in choices:
            if not choice:
                choice_masks.append(0)
                tokens_c = []
            else:
                choice_masks.append(1)
                tokens_c = tokenizer.encode(choice).tolist()[1:]
            
            tokens_qc = tokens_q + [sep_token] + tokens_c
            
            t_ctx = tokens_ctx[:]
            t_qc = tokens_qc[:]
            truncate_seq_tuple(t_ctx, t_qc, [], max_seq_length - 3)
                
            input_ids = [cls_token] + t_ctx + [sep_token] + t_qc + [sep_token]
            
            part1_len = 1 + len(t_ctx) + 1  
            part2_len = len(t_qc) + 1  
            token_type_ids = [0] * part1_len + [1] * part2_len
            attention_mask = [1] * len(input_ids)

            padding_length = max_seq_length - len(input_ids)
            if padding_length > 0:
                input_ids += [pad_token] * padding_length
                attention_mask += [0] * padding_length
                token_type_ids += [0] * padding_length 

            choices_input_ids.append(input_ids)
            choices_attention_mask.append(attention_mask)
            choices_token_type_ids.append(token_type_ids)

        features.append({
            "input_ids": choices_input_ids,
            "attention_mask": choices_attention_mask,
            "token_type_ids": choices_token_type_ids,
            "choice_mask": choice_masks,
            "label": example.label
        })
    return features


def create_dataloader(features, batch_size, is_training=True):
    all_input_ids = torch.tensor([f["input_ids"] for f in features], dtype=torch.long)
    all_attention_mask = torch.tensor([f["attention_mask"] for f in features], dtype=torch.long)
    all_token_type_ids = torch.tensor([f["token_type_ids"] for f in features], dtype=torch.long)
    all_choice_masks = torch.tensor([f["choice_mask"] for f in features], dtype=torch.float)
    all_labels = torch.tensor([f["label"] for f in features], dtype=torch.long)

    dataset = TensorDataset(all_input_ids, all_attention_mask, all_token_type_ids, all_choice_masks, all_labels)
    sampler = RandomSampler(dataset) if is_training else SequentialSampler(dataset)
    return DataLoader(dataset, sampler=sampler, batch_size=batch_size)


def evaluate(model, dataloader, device):
    model.eval()
    eval_loss, eval_accuracy = 0, 0
    nb_eval_steps, nb_eval_examples = 0, 0

    all_preds = []
    all_labels = []

    for batch in dataloader:
        batch = tuple(t.to(device) for t in batch)
        input_ids, attention_mask, token_type_ids, choice_masks, labels = batch

        with torch.no_grad():
            loss, logits = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                choice_mask=choice_masks,
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

        all_preds.extend(preds.tolist())
        all_labels.extend(label_ids.tolist())

    eval_loss = eval_loss / nb_eval_steps
    eval_accuracy = eval_accuracy / nb_eval_examples
    
    return eval_loss, eval_accuracy, all_preds, all_labels


def save_bad_cases(output_dir, dev_examples, all_preds, all_labels):
    """Lưu các mẫu dự đoán sai ra file text từ danh sách dữ liệu gốc."""
    error_file_path = os.path.join(output_dir, "bad_cases_best_epoch.txt")
    
    with open(error_file_path, "w", encoding="utf-8") as f:
        f.write("=== PHÂN TÍCH CÁC MẪU DỰ ĐOÁN SAI (BEST EPOCH) ===\n\n")
        
        error_count = 0
        for i in range(len(all_preds)):
            if all_preds[i] != all_labels[i]:
                error_count += 1
                pred_label = all_preds[i]
                true_label = all_labels[i]
                
                example = dev_examples[i]
                
                f.write(f"Mẫu sai thứ {error_count} (Index trong Dev set: {i}):\n")
                f.write(f"Ngữ cảnh (Context): {example.context}\n")
                f.write(f"Câu hỏi (Question): {example.question}\n")
                
                # In ra danh sách các lựa chọn để dễ đối chiếu lỗi
                f.write("Các lựa chọn (Choices):\n")
                for idx, choice in enumerate(example.choices):
                    if choice:
                        f.write(f"  [{idx}]: {choice}\n")
                
                f.write(f"[✓] ĐÁP ÁN ĐÚNG (Label gốc): {true_label} -> \"{example.choices[true_label]}\"\n")
                f.write(f"[✗] MODEL ĐOÁN SAI (Label sai): {pred_label} -> \"{example.choices[pred_label]}\"\n")
                f.write("-" * 80 + "\n")
    
    logger.info(f" 💾 Đã xuất {error_count} mẫu dự đoán sai ra file: {error_file_path}")


def train(args):
    # Kích hoạt seed ngẫu nhiên ngay đầu hàm huấn luyện
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)
    
    logger.info(f"Loading Config from: {args.init_checkpoint}")
    config_path = os.path.join(args.init_checkpoint, "config.json")
    weights_path = os.path.join(args.init_checkpoint, "model.safetensors")
    
    with open(config_path, "r", encoding="utf-8") as f:
        config_dict = json.load(f)
        
    config_dict["type_vocab_size"] = 2  
    config = PinyinBertConfig(**config_dict)
    
    model = PinyinBertForMultipleChoice(config).to(device)
    
    logger.info(f"Loading Weights from: {weights_path}")
    state_dict = load_file(weights_path)
    
    target_key = "bert.embeddings.token_type_embeddings.weight"
    if target_key in state_dict:
        pretrained_weight = state_dict[target_key]
        if pretrained_weight.shape[0] == 1 and config.type_vocab_size == 2:
            logger.info("👉 Tự động đệm tensor token_type_embeddings từ size 1 lên 2...")
            new_weight = torch.zeros(2, config.hidden_size, dtype=pretrained_weight.dtype, device=pretrained_weight.device)
            new_weight[0] = pretrained_weight[0]
            new_weight[1] = pretrained_weight[0].clone() 
            state_dict[target_key] = new_weight

    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    
    logger.info(f"Missing keys (Khởi tạo mới cho tác vụ C3): {missing_keys}")
    logger.info(f"Unexpected keys (Bỏ qua từ Pre-train): {unexpected_keys}")

    tokenizer = PinyinTokenizer(config) 

    logger.info("Loading Data...")
    cache_train_path = os.path.join(args.data_dir, f"cached_c3_train_{args.max_seq_length}.pt")
    cache_dev_path = os.path.join(args.data_dir, f"cached_c3_dev_{args.max_seq_length}.pt")

    # 1. Xử lý tập huấn luyện (Train)
    if os.path.exists(cache_train_path):
        logger.info(f"👉 Tìm thấy cache dữ liệu Train, đang tải nhanh từ: {cache_train_path}")
        train_features = torch.load(cache_train_path)
    else:
        logger.info("❌ Không tìm thấy cache Train. Bắt đầu convert từ file gốc...")
        train_examples = load_c3_data(os.path.join(args.data_dir, "d-train.json")) + \
                         load_c3_data(os.path.join(args.data_dir, "m-train.json"))
        train_features = convert_examples_to_features(train_examples, tokenizer, args.max_seq_length)
        logger.info(f"💾 Đang lưu dữ liệu Train đã convert vào cache: {cache_train_path}")
        torch.save(train_features, cache_train_path)

    # 2. Xử lý tập kiểm thử (Dev) - SỬA ĐỔI QUAN TRỌNG TẠI ĐÂY
    # Biến dev_examples luôn được khởi tạo bất kể có file cache hay không
    logger.info("📋 Đang đọc dữ liệu văn bản gốc tập Dev để sẵn sàng đối chiếu bad cases...")
    dev_examples = load_c3_data(os.path.join(args.data_dir, "d-dev.json"), is_training=True) + \
                   load_c3_data(os.path.join(args.data_dir, "m-dev.json"), is_training=True)

    if os.path.exists(cache_dev_path):
        logger.info(f"👉 Tìm thấy cache đặc trưng Dev, đang tải từ: {cache_dev_path}")
        dev_features = torch.load(cache_dev_path)
    else:
        logger.info("❌ Không tìm thấy cache đặc trưng Dev. Bắt đầu convert...")
        dev_features = convert_examples_to_features(dev_examples, tokenizer, args.max_seq_length)
        logger.info(f"💾 Đang lưu đặc trưng Dev đã convert vào cache: {cache_dev_path}")
        torch.save(dev_features, cache_dev_path)

    train_dataloader = create_dataloader(train_features, args.train_batch_size, is_training=True)
    dev_dataloader = create_dataloader(dev_features, args.eval_batch_size, is_training=False)

    t_total = len(train_dataloader) * args.epochs
    optimizer = AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.01)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(0.1 * t_total), num_training_steps=t_total)
    scaler = GradScaler() 

    logger.info("***** Running training *****")
    best_acc = 0.0

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        with tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{args.epochs}") as pbar:
            for batch in pbar:
                batch = tuple(t.to(device) for t in batch)
                input_ids, attention_mask, token_type_ids, choice_masks, labels = batch

                optimizer.zero_grad()

                with autocast("cuda"):
                    loss, logits = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        token_type_ids=token_type_ids,
                        choice_mask=choice_masks,
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

        logger.info("***** Running Evaluation *****")
        eval_loss, eval_acc, all_preds, all_labels = evaluate(model, dev_dataloader, device)
        
        logger.info(f"Epoch {epoch+1} - Train Loss: {total_loss/len(train_dataloader):.4f} - Eval Loss: {eval_loss:.4f} - Eval Acc: {eval_acc:.4f}")

        # Chỉ thực hiện lưu file bad cases khi epoch này là tốt nhất
        if eval_acc > best_acc:
            best_acc = eval_acc
            save_path = os.path.join(args.output_dir, "best_model_c3_2e-5.pt")
            torch.save(model.state_dict(), save_path)
            logger.info(f"Lưu mô hình tốt nhất với Accuracy: {best_acc:.4f} tại {save_path}")
            
            # Gọi hàm xuất file lỗi, dev_examples hiện tại đã được đảm bảo tồn tại an toàn
            save_bad_cases(args.output_dir, dev_examples, all_preds, all_labels)
            
    logger.info("=" * 50)
    logger.info(f"🏆 QUÁ TRÌNH HUẤN LUYỆN HOÀN THÀNH!")
    logger.info(f"🎯 Best Eval Accuracy đạt được: {best_acc:.4f}")
    logger.info("=" * 50)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True, help="Thư mục chứa dữ liệu C3")
    parser.add_argument("--init_checkpoint", type=str, required=True, help="Thư mục chứa config.json và model.safetensors từ quá trình Pre-train")
    parser.add_argument("--output_dir", type=str, default="./c3_outputs", help="Nơi lưu mô hình sau khi finetune")
    
    parser.add_argument("--max_seq_length", type=int, default=256)
    parser.add_argument("--train_batch_size", type=int, default=16)
    parser.add_argument("--eval_batch_size", type=int, default=32)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42, help="Giá trị mã seed ngẫu nhiên cố định")
    
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()