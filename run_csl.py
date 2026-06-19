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

class PinyinBertForSequenceClassification(nn.Module):
    def __init__(self, config, num_labels=2):
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

def load_csl_data(data_path):
    """
    Load CSL benchmark data from TSV files
    """
    examples = []
    label_set = set()
    
    if not os.path.exists(data_path):
        logger.warning(f"File not found: {data_path}")
        return examples, []
    
    with open(data_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split('\t')
            if len(parts) < 3:
                continue
            
            # Format: task_marker \t text \t label
            # task_marker is like "to category", "to discipline", "to keywords", "to title"
            marker = parts[0]
            text = parts[1]
            label = parts[2]
            
            label_set.add(label)
            examples.append({
                "text": text,
                "label": label,
                "marker": marker
            })
    
    label_list = sorted(list(label_set))
    label2id = {label: i for i, label in enumerate(label_list)}
    
    return examples, label_list

def convert_examples_to_features(examples, tokenizer, max_seq_length, label_list):
    label2id = {label: i for i, label in enumerate(label_list)}
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
            "label": label2id[example["label"]]
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
    
    tokenizer = PinyinTokenizer(config)
    
    logger.info("Đang đọc dữ liệu từ tất cả 4 subtask CSL (cls_ctg, cls_dcp, kg, ts)...")
    
    # Load data from all 4 subtasks
    subtasks = ["cls_ctg", "cls_dcp", "kg", "ts"]
    all_train_examples = []
    all_dev_examples = {}
    all_test_examples = {}
    all_labels = set()
    
    for subtask in subtasks:
        subtask_dir = os.path.join(args.data_dir, subtask)
        
        # Load train data
        train_path = os.path.join(subtask_dir, "train.tsv")
        train_exs, train_labels = load_csl_data(train_path)
        all_train_examples.extend(train_exs)
        all_labels.update(train_labels)
        
        # Load dev data
        dev_path = os.path.join(subtask_dir, "dev.tsv")
        dev_exs, dev_labels = load_csl_data(dev_path)
        all_dev_examples[subtask] = dev_exs
        all_labels.update(dev_labels)  # Add dev labels to complete label set
        
        # Load test data
        test_path = os.path.join(subtask_dir, "test.tsv")
        test_exs, test_labels = load_csl_data(test_path)
        all_test_examples[subtask] = test_exs
        all_labels.update(test_labels)  # Add test labels to complete label set
        
        logger.info(f"Subtask {subtask}: Train={len(train_exs)}, Dev={len(dev_exs)}, Test={len(test_exs)}")
    
    label_list = sorted(list(all_labels))
    num_labels = len(label_list)
    
    logger.info(f"\n=== Thống kê tổng hợp ===")
    logger.info(f"Tổng train samples: {len(all_train_examples)}")
    logger.info(f"Số lớp (từ tất cả splits): {num_labels}")
    logger.info(f"Danh sách lớp (top 15): {label_list[:15]}{'...' if len(label_list) > 15 else ''}")
    
    model = PinyinBertForSequenceClassification(config, num_labels=num_labels).to(device)
    
    logger.info(f"Loading Weights từ: {weights_path}")
    state_dict = load_file(weights_path)
    model.load_state_dict(state_dict, strict=False)
    
    # Convert examples to features - COMBINED TRAINING DATA
    logger.info("Đang convert tất cả training data thành features...")
    train_features = convert_examples_to_features(all_train_examples, tokenizer, args.max_seq_length, label_list)
    
    train_dataloader = create_dataloader(train_features, args.train_batch_size, is_training=True)
    
    t_total = len(train_dataloader) * args.epochs
    optimizer = AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.01)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(0.1 * t_total), num_training_steps=t_total)
    scaler = GradScaler()
    
    logger.info("***** Bắt đầu tiến trình Huấn luyện (Combined Dataset) *****\n")
    best_avg_acc = 0.0
    
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        nb_steps = 0
        
        for step, batch in enumerate(train_dataloader):
            batch = tuple(t.to(device) for t in batch)
            input_ids, attention_mask, token_type_ids, labels = batch
            
            optimizer.zero_grad()
            with autocast(device_type=device.type):
                loss, logits = model(input_ids, token_type_ids, attention_mask, labels)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            
            total_loss += loss.item()
            nb_steps += 1
        
        avg_train_loss = total_loss / nb_steps
        
        # Evaluate on each subtask separately
        logger.info(f"\n--- Epoch {epoch+1} ---")
        logger.info(f"Train Loss: {avg_train_loss:.4f}")
        
        results_by_subtask = {}
        overall_acc = []
        
        for subtask in subtasks:
            dev_exs = all_dev_examples[subtask]
            if len(dev_exs) > 0:
                dev_features = convert_examples_to_features(dev_exs, tokenizer, args.max_seq_length, label_list)
                dev_dataloader = create_dataloader(dev_features, args.eval_batch_size, is_training=False)
                eval_loss, eval_acc = evaluate(model, dev_dataloader, device)
                results_by_subtask[subtask] = eval_acc
                overall_acc.append(eval_acc)
                logger.info(f"  {subtask:10s} - Loss: {eval_loss:.4f}, Acc: {eval_acc:.4f}")
        
        avg_acc = sum(overall_acc) / len(overall_acc) if overall_acc else 0.0
        
        if avg_acc > best_avg_acc:
            best_avg_acc = avg_acc
            torch.save(model.state_dict(), os.path.join(args.output_dir, "best_model.pt"))
            logger.info(f"  ✓ Mô hình tốt nhất được lưu (Avg Acc: {avg_acc:.4f})")
    
    # Final evaluation on test set
    logger.info("\n\n=== FINAL EVALUATION ON TEST SETS ===")
    model.load_state_dict(torch.load(os.path.join(args.output_dir, "best_model.pt")))
    
    test_results = {}
    test_accs = []
    
    for subtask in subtasks:
        test_exs = all_test_examples[subtask]
        if len(test_exs) > 0:
            test_features = convert_examples_to_features(test_exs, tokenizer, args.max_seq_length, label_list)
            test_dataloader = create_dataloader(test_features, args.eval_batch_size, is_training=False)
            test_loss, test_acc = evaluate(model, test_dataloader, device)
            test_results[subtask] = test_acc
            test_accs.append(test_acc)
            logger.info(f"{subtask:10s} - Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}")
    
    avg_test_acc = sum(test_accs) / len(test_accs) if test_accs else 0.0
    logger.info(f"\nAverage Test Accuracy: {avg_test_acc:.4f}")
    logger.info(f"Best Model saved at: {os.path.join(args.output_dir, 'best_model.pt')}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True, help="Thư mục chứa tất cả subtask CSL (cls_ctg, cls_dcp, kg, ts)")
    parser.add_argument("--init_checkpoint", type=str, required=True, help="Thư mục weights gốc từ quá trình Pre-train")
    parser.add_argument("--output_dir", type=str, default="./csl_outputs", help="Nơi lưu checkpoint mô hình")
    parser.add_argument("--max_seq_length", type=int, default=256, help="Độ dài chuỗi tối đa")
    parser.add_argument("--train_batch_size", type=int, default=32, help="Batch size huấn luyện")
    parser.add_argument("--eval_batch_size", type=int, default=64, help="Batch size đánh giá")
    parser.add_argument("--learning_rate", type=float, default=3e-5, help="Learning rate")
    parser.add_argument("--epochs", type=int, default=10, help="Số epoch")
    
    args = parser.parse_args()
    train(args)

if __name__ == "__main__":
    main()
