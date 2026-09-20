import torch
import numpy as np
import random
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LambdaLR

from configs.segment_free_bert_config import SegmentFreeBertConfig
from vocabs.segment_free_tokenizer import SegmentFreeTokenizer
from data_utils.segment_free_dataset import SegmentFreeDataset, collate_fn
from models.segment_free_bert import SegmentFreeBert

from tqdm import tqdm
import os
import wandb

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)

set_seed(42)

BS = 64
CHECKPOINT = "segment_free_bert_weights"
MODEL_NAME = "segment_free_bert_base"
CORPUS_DIR = "data/Vietnamese-curated-corpus"

# Đổi thành True nếu muốn khôi phục và chạy tiếp từ checkpoint cũ sau khi bị ngắt quãng
RESUME_FROM_CHECKPOINT = True

device = "cuda" if torch.cuda.is_available() else "cpu"

config = SegmentFreeBertConfig(max_length=512)
tokenizer = SegmentFreeTokenizer(config)
dataset = SegmentFreeDataset(
    tokenizer=tokenizer,
    corpus_dir=CORPUS_DIR,
    max_length=config.max_length
)

g = torch.Generator()
g.manual_seed(42)

dataloader = DataLoader(
    dataset=dataset,
    batch_size=BS,
    shuffle=True,
    num_workers=24,
    collate_fn=collate_fn,
    worker_init_fn=lambda worker_id: np.random.seed(42 + worker_id), # Seed cho các worker
    generator=g,
    pin_memory=True
)

model = SegmentFreeBert(config).load_xlm_roberta_backbone().to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=0.01, betas=(0.9, 0.999), eps=1e-6)

total_steps = 1_000_000
warmup_steps = int(total_steps * 0.01)

def lr_lambda(current_step):
    if current_step < warmup_steps:
        return float(current_step) / float(max(1, warmup_steps))
    return max(0.0, float(total_steps - current_step) / float(max(1, total_steps - warmup_steps)))

lr_scheduler = LambdaLR(optimizer, lr_lambda)
scaler = torch.amp.GradScaler('cuda')

start_epoch = 1
global_step = 0
checkpoint_path = os.path.join(CHECKPOINT, f"{MODEL_NAME}_training.pth")

if RESUME_FROM_CHECKPOINT and os.path.isfile(checkpoint_path):
    print(f"=> Tìm thấy checkpoint. Đang khôi phục từ: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)

    start_epoch = checkpoint["epoch"]
    global_step = checkpoint["global_step"]
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    lr_scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    scaler.load_state_dict(checkpoint["scaler_state_dict"])

    lr_scheduler.last_epoch = global_step
    print(f"=> Khôi phục thành công! Tiếp tục train từ Epoch {start_epoch}, Step tổng {global_step}")
else:
    print("=> Không kích hoạt resume hoặc không tìm thấy file. Bắt đầu pre-train mới từ đầu.")

if not os.path.isdir(CHECKPOINT):
    os.makedirs(CHECKPOINT, exist_ok=True)

wandb.init(
    project="SegmentFreeBERT",
    name=MODEL_NAME,
    config={
        "batch_size": BS,
        "lr": 5e-5,
        "hidden_size": config.hidden_size,
        "num_hidden_layers": config.num_hidden_layers,
        "num_attention_heads": config.num_attention_heads,
        "max_length": config.max_length,
        "vocab_size": config.vocab_size,
        "warmup_steps": warmup_steps,
        "total_steps": total_steps,
    }
)

print(f"Total steps: {total_steps} | Khởi động tại Step: {global_step}")

model.train()
done = False
while True:
    total_loss = 0
    progress_bar = tqdm(dataloader, desc=f"Epoch {start_epoch}")

    for batch in progress_bar:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        optimizer.zero_grad()

        with torch.amp.autocast('cuda'):
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs['loss']

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        lr_scheduler.step()

        loss_value = loss.item()
        total_loss += loss_value
        progress_bar.set_postfix({
            'loss': f"{loss_value:.4f}",
            'step': global_step,
            'lr': f"{lr_scheduler.get_last_lr()[0]:.2e}"
        })

        wandb.log(
            {
                "train/loss": loss_value,
                "train/lr": lr_scheduler.get_last_lr()[0],
                "train/epoch": start_epoch,
            },
            step=global_step
        )

        global_step += 1
        if global_step > total_steps:
            done = True
            break

        if global_step % 1000 == 0:
            torch.save({
                "epoch": start_epoch,
                "global_step": global_step,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": lr_scheduler.state_dict(),
                "scaler_state_dict": scaler.state_dict()
            }, checkpoint_path)

    model.save_pretrained(os.path.join(CHECKPOINT, MODEL_NAME))

    avg_loss = total_loss / len(dataloader)
    print(f"Epoch {start_epoch} - Average Loss: {avg_loss:.4f}")
    wandb.log({"train/epoch_avg_loss": avg_loss}, step=global_step)

    if done:
        break

    start_epoch += 1

wandb.finish()
