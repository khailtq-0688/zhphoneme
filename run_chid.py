import argparse
import json
import logging
import os
import re
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler, Dataset
from torch.optim import AdamW
from torch.amp import autocast, GradScaler
from transformers import get_linear_schedule_with_warmup, BertModel
from safetensors.torch import load_file

from configs.pinyin_bert_config import PinyinBertConfig
from vocabs.pinyin_tokenizer import PinyinTokenizer

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

NUM_CHOICES = 10  # Bài toán ChID luôn có 10 ứng cử viên thành ngữ


class PinyinBertForMultipleChoice(nn.Module):
    def __init__(self, config, num_choices=10):
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

        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, 1)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None):
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
        
        cls_output = self.dropout(cls_output)
        logits = self.classifier(cls_output)
        
        reshaped_logits = logits.view(-1, self.num_choices)

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(reshaped_logits, labels)

        return loss, reshaped_logits

class ChidDataset(Dataset):
    def __init__(self, examples, tokenizer, max_seq_len):
        super().__init__()

        self.examples = examples
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_len

    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx: int):
        sample = self.examples[idx]

        cls_token = (self.tokenizer.config.cls_token_id,) * 3
        pad_token = (self.tokenizer.config.pad_token_id,) * 3

        input_ids = []
        attention_mask = []
        for candidate_context in sample["context"]:
            ids = self.tokenizer.encode(candidate_context).tolist()[1:]
            ids = [cls_token] + ids
            mask = [1] * len(ids)
            padding_length = self.max_seq_length - len(ids)
            if padding_length > 0:
                ids += [pad_token] * padding_length
                mask += [0] * padding_length
            else:
                ids = ids[:self.max_seq_length]
                mask = mask[:self.max_seq_length]

            input_ids.append(ids)
            attention_mask.append(mask)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "label": sample["label"]
        }

def load_chid_data(data_path, answer_path=None, is_training=True):
    """
    Hàm đọc dữ liệu ChID.
    Sử dụng Regex để bóc tách các khoảng trống #idiomXXXXXX# thành các sample độc lập.
    """
    answers = {}
    if answer_path and os.path.exists(answer_path):
        with open(answer_path, 'r', encoding='utf-8') as f:
            answers = json.load(f)

    examples = []
    data = open(data_path).readlines()
    for item in data:
        item = json.loads(item)
        candidates = item["candidates"]
        
        # Đảm bảo luôn có 10 choices
        while len(candidates) < NUM_CHOICES:
            candidates.append("无效成语")

        content_list = item["content"]
        for content in content_list:
            # Tìm tất cả các mã khoảng trống trong đoạn văn (vd: #idiom577157#)
            placeholders = re.findall(r'#idiom\d+#', content)
            for placeholder in placeholders:
                # for all placeholder that is not the consider one
                for other_placeholder in placeholders:
                    if other_placeholder != placeholder:
                        answer = answers[other_placeholder]
                        content = re.sub(other_placeholder, candidates[answer], content)
                
                # create a context for each candidate
                candidate_contexts = []
                for candidate in candidates:
                    candidate_contexts.append(re.sub(placeholder, candidate, content))
                
                # Nếu đang ở chế độ train/dev, tra cứu nhãn đúng. Tập test sẽ gán nhãn 0 mặc định.
                label = answers[placeholder]
                
                examples.append({
                    "context": candidate_contexts,
                    "label": label
                })
    return examples

def collate_fn(samples):
    input_ids = [torch.tensor(sample["input_ids"]) for sample in samples]
    attention_mask = [torch.tensor(sample["attention_mask"]) for sample in samples]
    labels = [sample["label"] for sample in samples]

    input_ids = torch.stack(input_ids, dim=0)
    attention_mask = torch.stack(attention_mask, dim=0)
    labels = torch.tensor(labels)

    return input_ids, attention_mask, labels

def convert_examples_to_features(examples, tokenizer, max_seq_length):
    cls_token = (tokenizer.config.cls_token_id,) * 3
    pad_token = (tokenizer.config.pad_token_id,) * 3
    sep_token = (tokenizer.config.empty_token_id,) * 3  # Fallback SEP

    features = []
    for example in tqdm(examples, desc="Converting features"):
        choices_input_ids = []
        choices_attention_mask = []
        choices_token_type_ids = []

        tokens_ctx = tokenizer.encode(example.context).tolist()[1:]

        for choice in example.choices:
            tokens_c = tokenizer.encode(choice).tolist()[1:]
            
            t_ctx = tokens_ctx[:]
            t_c = tokens_c[:]

            # Vòng lặp Truncate: Ưu tiên cắt phần ngữ cảnh (context), giữ nguyên 4 chữ thành ngữ (choice)
            while len(t_ctx) + len(t_c) > max_seq_length - 3:
                t_ctx.pop()

            # GHÉP CHUỖI VÀ XỬ LÝ SEGMENT IDs (Đã sửa lỗi logic)
            # Cấu trúc: [CLS] Context [SEP] Choice [SEP]
            part1_len = 1 + len(t_ctx) + 1  # [CLS] + Context + [SEP]
            part2_len = len(t_c) + 1        # Choice + [SEP]
            
            input_ids = [cls_token] + t_ctx + [sep_token] + t_c + [sep_token]
            
            # Context mang Segment 0, Choice mang Segment 1
            token_type_ids = [0] * part1_len + [1] * part2_len
            attention_mask = [1] * len(input_ids)

            # Padding
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
            "label": example.label
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
        input_ids, attention_mask, labels = batch

        with torch.no_grad():
            loss, logits = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
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
    
    logger.info(f"Loading Config from: {args.init_checkpoint}")
    config_path = os.path.join(args.init_checkpoint, "config.json")
    weights_path = os.path.join(args.init_checkpoint, "model.safetensors")
    if not os.path.exists(weights_path):
        weights_path = os.path.join(args.init_checkpoint, "pytorch_model.bin")
    
    with open(config_path, "r", encoding="utf-8") as f:
        config_dict = json.load(f)
    config = PinyinBertConfig(**config_dict)
    
    model = PinyinBertForMultipleChoice(config, num_choices=NUM_CHOICES).to(device)
    
    logger.info(f"Loading Weights from: {weights_path}")
    if weights_path.endswith('.safetensors'):
        state_dict = load_file(weights_path)
    else:
        state_dict = torch.load(weights_path, map_location=device)
        
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    
    tokenizer = PinyinTokenizer(config) 

    # ĐỌC DỮ LIỆU ChID
    cached_train_file = os.path.join(args.data_dir, f"cached_train_features_{args.max_seq_length}.pt")
    cached_dev_file = os.path.join(args.data_dir, f"cached_dev_features_{args.max_seq_length}.pt")

    # # 1. XỬ LÝ TẬP TRAIN
    # if os.path.exists(cached_train_file):
    #     logger.info(f"⚡ Đã tìm thấy file cache! Đang tải nhanh Train features từ: {cached_train_file}")
    #     train_features = torch.load(cached_train_file)
    # else:
    #     logger.info("Đang đọc dữ liệu thô Train ChID (Chỉ cần làm 1 lần)...")
    #     train_examples = load_chid_data(
    #         data_path=os.path.join(args.data_dir, "train.json"), 
    #         answer_path=os.path.join(args.data_dir, "train_answer.json"), 
    #         is_training=True
    #     )
    #     logger.info(f"Train samples (Blanks): {len(train_examples)}")
    #     train_features = convert_examples_to_features(train_examples, tokenizer, args.max_seq_length)
        
    #     logger.info(f"💾 Đang lưu Train features vào cache để dùng cho lần sau: {cached_train_file}")
    #     torch.save(train_features, cached_train_file)

    train_examples = load_chid_data(
        data_path=os.path.join(args.data_dir, "train.json"), 
        answer_path=os.path.join(args.data_dir, "train_answer.json"), 
        is_training=True
    )
    train_dataset = ChidDataset(train_examples, tokenizer, args.max_seq_length)

    # # 2. XỬ LÝ TẬP DEV
    # if os.path.exists(cached_dev_file):
    #     logger.info(f"⚡ Đã tìm thấy file cache! Đang tải nhanh Dev features từ: {cached_dev_file}")
    #     dev_features = torch.load(cached_dev_file)
    # else:
    #     logger.info("Đang đọc dữ liệu thô Dev ChID...")
    #     dev_examples = load_chid_data(
    #         data_path=os.path.join(args.data_dir, "dev.json"), 
    #         answer_path=os.path.join(args.data_dir, "dev_answer.json"), 
    #         is_training=True
    #     )
    #     logger.info(f"Dev samples: {len(dev_examples)}")
    #     dev_features = convert_examples_to_features(dev_examples, tokenizer, args.max_seq_length)
        
    #     logger.info(f"💾 Đang lưu Dev features vào cache: {cached_dev_file}")
    #     torch.save(dev_features, cached_dev_file)

    dev_examples = load_chid_data(
        data_path=os.path.join(args.data_dir, "dev.json"), 
        answer_path=os.path.join(args.data_dir, "dev_answer.json"), 
        is_training=True
    )
    dev_dataset = ChidDataset(dev_examples, tokenizer, args.max_seq_length)

    # train_dataloader = create_dataloader(train_features, args.train_batch_size, is_training=True)
    # dev_dataloader = create_dataloader(dev_features, args.eval_batch_size, is_training=False)

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        num_workers=24,
        collate_fn=collate_fn
    )

    dev_dataloader = DataLoader(
        dev_dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        num_workers=24,
        collate_fn=collate_fn
    )

    # OPTIMIZER & SCHEDULER
    t_total = len(train_dataloader) * args.epochs
    optimizer = AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.01)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(0.1 * t_total), num_training_steps=t_total)
    scaler = GradScaler('cuda') 

    logger.info("***** Running training *****")
    best_acc = 0.0

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        with tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{args.epochs}") as pbar:
            for batch in pbar:
                batch = tuple(t.to(device) for t in batch)
                input_ids, attention_mask, labels = batch

                optimizer.zero_grad()

                with autocast("cuda"):
                    loss, logits = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
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
        eval_loss, eval_acc = evaluate(model, dev_dataloader, device)
        logger.info(f"Epoch {epoch+1} - Train Loss: {total_loss/len(train_dataloader):.4f} - Eval Loss: {eval_loss:.4f} - Eval Acc: {eval_acc*100:.2f}%")

        # Lưu Checkpoint tốt nhất
        if eval_acc > best_acc:
            best_acc = eval_acc
            save_path = os.path.join(args.output_dir, "best_model_chid.pt")
            torch.save(model.state_dict(), save_path)
            logger.info(f"✨ Lưu mô hình tốt nhất với Accuracy: {best_acc*100:.2f}% tại {save_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True, help="Thư mục chứa dữ liệu ChID (train, dev, answers)")
    parser.add_argument("--init_checkpoint", type=str, required=True, help="Thư mục chứa weights Pre-train")
    parser.add_argument("--output_dir", type=str, default="./chid_outputs", help="Thư mục xuất output")
    
    parser.add_argument("--max_seq_length", type=int, default=128, help="Chiều dài chuỗi context + candidate")
    parser.add_argument("--train_batch_size", type=int, default=8, help="Batch size (ChID tốn VRAM gấp 10 lần do 10 choices)")
    parser.add_argument("--eval_batch_size", type=int, default=16)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--epochs", type=int, default=5)
    
    args = parser.parse_args()
    train(args)

if __name__ == "__main__":
    main()