import torch
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm
import os
from configs.pinyin_bert_config import PinyinBertConfig
from vocabs.pinyin_tokenizer import PinyinTokenizer
from data_utils.pinyin_dataset import PinyinDataset
from models.pinyin_bert import PinyinBert
from data_utils.pinyin_dataset import collate_fn

# 1. Định nghĩa đường dẫn và thông số cơ bản trước
BS = 64
MODEL_DIR = "pinyin_bert_weights/pinyin_bert_base" 
TRAINING_STATE_PATH = "pinyin_bert_weights/pinyin_bert_base_training.pth"
device = "cuda" if torch.cuda.is_available() else "cpu"

# 2. Load Config từ folder weights để giữ nguyên cấu trúc (Small/Base/Large)
# và đặt cứng pad_token_id tại đây
config = PinyinBertConfig.from_pretrained(MODEL_DIR)
config.pad_token_id = 0

# KHÔNG khởi tạo lại config cứng ở đây nữa để tránh làm sai lệch cấu trúc model

tokenizer = PinyinTokenizer(config)

print(f"Loading weights from: {MODEL_DIR}")
model = PinyinBert.from_pretrained(MODEL_DIR, config=config).to(device)

# 4. Setup Dataloader
dataset = PinyinDataset(tokenizer=tokenizer, corpus_dir="data/baidubaike_chinese", max_length=config.max_length)
dataloader = DataLoader(dataset=dataset, batch_size=BS, shuffle=True, num_workers=24, collate_fn=collate_fn)

# 5. Khởi tạo Optimizer & Scheduler
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=0.01, betas=(0.9, 0.999), eps=1e-6)

total_steps = 3_000_000
warmup_steps = int(total_steps * 0.1)

def lr_lambda(current_step):
    if current_step < warmup_steps:
        return float(current_step) / float(max(1, warmup_steps))
    return max(0.0, float(total_steps - current_step) / float(max(1, total_steps - warmup_steps)))

lr_scheduler = LambdaLR(optimizer, lr_lambda)

scaler = torch.amp.GradScaler('cuda')

start_epoch = 0

# 6. Resume trạng thái huấn luyện
if os.path.exists(TRAINING_STATE_PATH):
    print(f"Resuming training state from: {TRAINING_STATE_PATH}")
    checkpoint = torch.load(TRAINING_STATE_PATH, map_location=device)
    optimizer.load_state_dict(checkpoint["optimizer"])
    start_epoch = checkpoint["epoch"]
    
    if "scheduler" in checkpoint:
        lr_scheduler.load_state_dict(checkpoint["scheduler"])
        print("Loaded scheduler state from checkpoint.")
    else:
        print("No scheduler state found in checkpoint. Starting scheduler from scratch.")

    if "scaler" in checkpoint:
        scaler.load_state_dict(checkpoint["scaler"])

    print(f"Resume successful. Next Epoch: {start_epoch + 1}")
else:
    print("No .pth file found! Starting from Epoch 1 with pre-trained weights.")

EPOCHS = 100 
print(f"Total Epochs: {EPOCHS}")

model.train()

for epoch in range(start_epoch + 1, EPOCHS + 1):
    total_loss = 0
    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch}/{EPOCHS}")

    for batch in progress_bar:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        optimizer.zero_grad()

        with torch.amp.autocast('cuda'):
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs['loss']      
        
        # loss = outputs['loss']
        # loss.backward()
        # optimizer.step()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()


        lr_scheduler.step()
        
        total_loss += loss.item()
        progress_bar.set_postfix({
            'loss': f"{loss.item():.4f}",
            'lr': f"{lr_scheduler.get_last_lr()[0]:.2e}"
        })

    # Lưu checkpoint sau mỗi epoch
    torch.save({
        "epoch": epoch,
        "scheduler": lr_scheduler.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict()
    }, TRAINING_STATE_PATH)

    model.save_pretrained(MODEL_DIR)

    avg_loss = total_loss / len(dataloader)
    print(f"Epoch {epoch} - Average Loss: {avg_loss:.4f}")