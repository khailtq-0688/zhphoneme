import torch
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LambdaLR

from configs.pinyin_bert_config import PinyinBertConfig
from vocabs.pinyin_tokenizer import PinyinTokenizer
from data_utils.pinyin_dataset import PinyinDataset
from models.pinyin_bert import PinyinBert
from data_utils.pinyin_dataset import collate_fn

from tqdm import tqdm
import os

EPOCHS = 150
BS = 512
CHECKPOINT = "pinyin_bert_weights"
MODEL_NAME = "pinyin_bert_large"

device = "cuda" if torch.cuda.is_available() else "cpu"

config = PinyinBertConfig(
    hidden_size=1152,
    num_hidden_layers=24,
    num_attention_heads=18,
    intermediate_size=4096,
    hidden_act="gelu",
    hidden_dropout_prob=0.1,
    attention_probs_dropout_prob=0.1,
    max_position_embeddings=512,
    max_length=512, # config for baidubaike pretrained corpus
    type_vocab_size=1,
    is_decoder=False,
    add_cross_attention = False
)
tokenizer = PinyinTokenizer(config)
dataset = PinyinDataset(
    tokenizer=tokenizer, 
    corpus_dir="../Chinese-pretrained-corpus/baidubaike_corpus", 
    max_length=config.max_length
)
dataloader = DataLoader(
    dataset=dataset,
    batch_size=BS,
    shuffle=True,
    num_workers=4,
    collate_fn=collate_fn
)
model = PinyinBert(config).to(device)
model.train()
optimizer = torch.optim.AdamW(model.parameters(), lr=6e-4, weight_decay=0.01, betas=(0.9, 0.98), eps=10e-6)

total_steps = len(dataloader) * EPOCHS
warmup_steps = int(total_steps * 0.05)

print(f"Total steps: {total_steps}")

def lr_lambda(current_step):
    if current_step < warmup_steps:
        return float(current_step) / float(max(1, warmup_steps)) 
    return max(0.0, float(total_steps - current_step) / float(max(1, total_steps - warmup_steps)))
    
lr_scheduler = LambdaLR(optimizer, lr_lambda)

if not os.path.isdir(CHECKPOINT):
    os.mkdir(CHECKPOINT)

for epoch in range(1, EPOCHS + 1):
    total_loss = 0
    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch}/{EPOCHS}")
    
    for batch in progress_bar:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        
        loss = outputs['loss']
        loss.backward()
        optimizer.step()
        lr_scheduler.step()
        
        total_loss += loss.item()
        progress_bar.set_postfix({'loss': f"{loss.item():.4f}"})

    torch.save({
        "epoch": epoch,
        "scheduler": lr_scheduler.state_dict(),
        "optimizer": optimizer.state_dict()
    }, os.path.join(CHECKPOINT, f"{MODEL_NAME}_training.pth"))

    model.save_pretrained(os.path.join(CHECKPOINT, f"{MODEL_NAME}"))
        
    avg_loss = total_loss / len(dataloader)
    print(f"Epoch {epoch} - Average Loss: {avg_loss:.4f}")
