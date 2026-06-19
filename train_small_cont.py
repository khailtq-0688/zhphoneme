import torch
import numpy as np
import random
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm
import os
from configs.pinyin_bert_config import PinyinBertConfig
from vocabs.pinyin_tokenizer import PinyinTokenizer
from data_utils.pinyin_dataset import PinyinDataset
from models.pinyin_bert import PinyinBert
from data_utils.pinyin_dataset import collate_fn

# --- ĐỒNG BỘ SEED ---
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)

set_seed(42)
# -------------------------

# 1. Cấu hình đường dẫn ĐỌC VÀO và GHI RA riêng biệt
BS = 300  
device = "cuda" if torch.cuda.is_available() else "cpu"

SRC_MODEL_DIR = "pinyin_bert_weights/pinyin_bert_small" 
SRC_TRAINING_STATE_PATH = "pinyin_bert_weights/pinyin_bert_small_training.pth"

# Đường dẫn GHI OUT (Thư mục mới lưu kết quả train tiếp)
DST_MODEL_DIR = "pinyin_bert_weights_small_cont"
DST_TRAINING_STATE_PATH = os.path.join(DST_MODEL_DIR, "pinyin_bert_small_training.pth")

# Tự động tạo thư mục đích nếu chưa tồn tại
os.makedirs(DST_MODEL_DIR, exist_ok=True)

# 2. Load Config và Mô hình từ thư mục cũ
config = PinyinBertConfig.from_pretrained(SRC_MODEL_DIR)
config.pad_token_id = 0
tokenizer = PinyinTokenizer(config)

print(f"Loading weights from: {SRC_MODEL_DIR}")
model = PinyinBert.from_pretrained(SRC_MODEL_DIR, config=config).to(device)

# 3. Setup Dataloader
dataset = PinyinDataset(tokenizer=tokenizer, corpus_dir="data/baidubaike_chinese", max_length=config.max_length)
g = torch.Generator()
g.manual_seed(42)

dataloader = DataLoader(
    dataset=dataset,
    batch_size=BS,
    shuffle=True,
    num_workers=24,
    collate_fn=collate_fn,
    worker_init_fn=lambda worker_id: np.random.seed(42 + worker_id), 
    generator=g
)

# 4. Khởi tạo Optimizer & Scheduler (Tăng tổng step để train tiếp)
optimizer = torch.optim.AdamW(model.parameters(), lr=9e-4, weight_decay=0.01, betas=(0.9, 0.999), eps=1e-6)

total_steps = 5000000  
warmup_steps = int(total_steps * 0.10) 

def lr_lambda(current_step):
    if current_step < warmup_steps:
        return float(current_step) / float(max(1, warmup_steps))
    return max(0.0, float(total_steps - current_step) / float(max(1, total_steps - warmup_steps)))

lr_scheduler = LambdaLR(optimizer, lr_lambda)
scaler = torch.amp.GradScaler('cuda')

start_epoch = 0

# 5. Khôi phục trạng thái huấn luyện cũ
# Ưu tiên tìm file .pth ở thư mục MỚI trước (để nếu bạn có bấm dừng rồi chạy lại bản cont này thì nó load tiếp bản cont)
# Nếu không thấy ở thư mục MỚI, nó sẽ lấy từ thư mục CŨ để làm bàn đạp bắt đầu.
if os.path.exists(DST_TRAINING_STATE_PATH):
    load_path = DST_TRAINING_STATE_PATH
    print(f"Resuming training state from NEW checkpoint: {load_path}")
elif os.path.exists(SRC_TRAINING_STATE_PATH):
    load_path = SRC_TRAINING_STATE_PATH
    print(f"Starting continuous training using BASE checkpoint: {load_path}")
else:
    load_path = None
    print("Warning: No .pth training state found anywhere! Starting from Epoch 1 with static weights.")

if load_path:
    checkpoint = torch.load(load_path, map_location=device)
    optimizer.load_state_dict(checkpoint["optimizer"])
    start_epoch = checkpoint["epoch"]
    
    if "scheduler" in checkpoint:
        lr_scheduler.load_state_dict(checkpoint["scheduler"])
        print("Loaded scheduler state successfully.")
    if "scaler" in checkpoint:
        scaler.load_state_dict(checkpoint["scaler"])
    print(f"Resume successful. Next Epoch will be: {start_epoch + 1}")

# Tính toán tổng số Epoch dựa trên số step mới đặt ra (1.5M steps / len(dataloader) ~ 125 Epochs)
EPOCHS = total_steps // len(dataloader) 
print(f"Total Epochs scheduled for entire pipeline: {EPOCHS} (Training will continue from Epoch {start_epoch + 1} to {EPOCHS})")

model.train()

# 6. Vòng lặp huấn luyện
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
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        lr_scheduler.step()
        
        total_loss += loss.item()
        progress_bar.set_postfix({
            'loss': f"{loss.item():.4f}",
            'lr': f"{lr_scheduler.get_last_lr()[0]:.2e}"
        })

    # LƯU CHECKPOINT VÀO THƯ MỤC ĐÍCH MỚI
    torch.save({
        "epoch": epoch,
        "scheduler": lr_scheduler.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict()
    }, DST_TRAINING_STATE_PATH)

    model.save_pretrained(DST_MODEL_DIR)
    config.save_pretrained(DST_MODEL_DIR) # Lưu kèm config sang thư mục mới

    avg_loss = total_loss / len(dataloader)
    print(f"Epoch {epoch} - Average Loss: {avg_loss:.4f}")