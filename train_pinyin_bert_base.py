import torch
import numpy as np
import random
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LambdaLR

from configs.pinyin_bert_config import PinyinBertConfig
from vocabs.pinyin_tokenizer import PinyinTokenizer
from data_utils.pinyin_dataset import PinyinDataset
from models.pinyin_bert import PinyinBert
from data_utils.pinyin_dataset import collate_fn

from tqdm import tqdm
import os

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)

# Kích hoạt cố định seed tuyệt đối đầu chương trình
set_seed(42)

BS = 64
CHECKPOINT = "pinyin_bert_weights"
MODEL_NAME = "pinyin_bert_base"

# Đổi thành True nếu muốn khôi phục trạng thái và chạy tiếp từ checkpoint bị ngắt quãng
RESUME_FROM_CHECKPOINT = True

device = "cuda" if torch.cuda.is_available() else "cpu"

config = PinyinBertConfig(
    hidden_size=768,
    num_hidden_layers=12,
    num_attention_heads=12,
    intermediate_size=3072,
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
    corpus_dir="data/baidubaike_chinese", 
    max_length=config.max_length
)

# Khởi tạo Generator riêng cho DataLoader để đảm bảo tính nhất quán dữ liệu qua các lần xáo trộn
g = torch.Generator()
g.manual_seed(42)

dataloader = DataLoader(
    dataset=dataset,
    batch_size=BS,
    shuffle=True,
    num_workers=24,
    collate_fn=collate_fn,
    worker_init_fn=lambda worker_id: np.random.seed(42 + worker_id), # Thiết lập seed độc lập cho từng worker
    generator=g,
    pin_memory=True
)

model = PinyinBert(config).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=0.01, betas=(0.9, 0.999), eps=1e-6)

total_steps = 1_000_000
warmup_steps = int(total_steps * 0.10)

def lr_lambda(current_step):
    if current_step < warmup_steps:
        return float(current_step) / float(max(1, warmup_steps)) 
    return max(0.0, float(total_steps - current_step) / float(max(1, total_steps - warmup_steps)))
    
lr_scheduler = LambdaLR(optimizer, lr_lambda)
scaler = torch.amp.GradScaler('cuda')

start_epoch = 1
global_step = 0
checkpoint_path = os.path.join(CHECKPOINT, f"{MODEL_NAME}_training.pth")

# Cơ chế khôi phục trạng thái huấn luyện (Resume) từ checkpoint
if RESUME_FROM_CHECKPOINT and os.path.isfile(checkpoint_path):
    print(f"=> Tìm thấy checkpoint. Đang tiến hành khôi phục từ: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    start_epoch = checkpoint["epoch"]
    global_step = checkpoint["global_step"]
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    lr_scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    scaler.load_state_dict(checkpoint["scaler_state_dict"])
    
    # Cập nhật mốc bước chạy cuối cùng cho scheduler tránh bị nhảy vọt LR
    lr_scheduler.last_epoch = global_step
    print(f"=> Khôi phục thành công! Tiếp tục huấn luyện tại Epoch {start_epoch}, Step tổng: {global_step}")
else:
    print(f"=> Không kích hoạt resume hoặc không tìm thấy file checkpoint. Khởi chạy tiền huấn luyện mới từ đầu.")

print(f"Total steps mục tiêu: {total_steps} | Khởi động thực tế tại Step: {global_step}")

if not os.path.isdir(CHECKPOINT):
    os.makedirs(CHECKPOINT, exist_ok=True)

model.train()
done = False

# Vòng lặp Step-based vô hạn sử dụng while True đồng bộ theo cấu trúc ViPhonBERT
while True:
    total_loss = 0
    progress_bar = tqdm(dataloader, desc=f"Epoch {start_epoch}")

    for batch in progress_bar:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        optimizer.zero_grad()
        
        # Huấn luyện độ chính xác hỗn hợp AMP tự động
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
            'step': global_step,
            'lr': f"{lr_scheduler.get_last_lr()[0]:.2e}"
        })
        
        global_step += 1
        
        # Kiểm tra điều kiện dừng ngắt vòng lặp khi chạm ngưỡng steps mục tiêu
        if global_step >= total_steps:
            done = True
            break
            
        # Cơ chế lưu checkpoint định kỳ sau mỗi 1000 steps huấn luyện
        if global_step % 1000 == 0:
            torch.save({
                "epoch": start_epoch + 1,  
                "global_step": global_step,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": lr_scheduler.state_dict(),
                "scaler_state_dict": scaler.state_dict()
            }, checkpoint_path)

    # Lưu trọng số mô hình dạng chuẩn .save_pretrained định kỳ cuối mỗi epoch
    model.save_pretrained(os.path.join(CHECKPOINT, f"{MODEL_NAME}"))

    avg_loss = total_loss / len(dataloader)
    print(f"Epoch {start_epoch} - Average Loss: {avg_loss:.4f}")
    
    if done:
        break
        
    start_epoch += 1