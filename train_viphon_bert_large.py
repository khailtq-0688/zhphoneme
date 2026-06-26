import torch
import numpy as np
import random
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LambdaLR

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import contextlib

from configs.viphon_bert_config import ViPhonBertConfig
from vocabs.viphon_tokenizer import ViPhonTokenizer
from data_utils.viphon_dataset import ViPhonDataset
from models.viphon_bert import ViPhonBert
from data_utils.viphon_dataset import collate_fn

from tqdm import tqdm
import os
import argparse
import wandb

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint_path", required=True, type=str)
parser.add_argument("--model_name", default="viphon_bert_large", type=str)
parser.add_argument("--corpus_dir", required=True, type=str)
args = parser.parse_args()

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)

set_seed(42)

BS = 256
CHECKPOINT = args.checkpoint_path
MODEL_NAME = args.model_name
ACCUMULATION_STEPS = 8

# Đổi thành True nếu muốn khôi phục và chạy tiếp từ checkpoint cũ sau khi bị ngắt quãng
RESUME_FROM_CHECKPOINT = False 

device = "cuda" if torch.cuda.is_available() else "cpu"

dist.init_process_group(backend="nccl")
local_rank = int(os.environ["LOCAL_RANK"])
torch.cuda.set_device(local_rank)
device = torch.device("cuda", local_rank)
rank = dist.get_rank()

config = ViPhonBertConfig(
    hidden_size=768,
    num_hidden_layers=12,
    num_attention_heads=12,
    intermediate_size=3072,
    hidden_act="gelu",
    hidden_dropout_prob=0.1,
    attention_probs_dropout_prob=0.1,
    max_position_embeddings=2048,
    max_length=2048, # config for baidubaike pretrained corpus
    type_vocab_size=1,
    is_decoder=False,
    add_cross_attention = False
)
tokenizer = ViPhonTokenizer(config)
dataset = ViPhonDataset(
    tokenizer=tokenizer, 
    corpus_dir=args.corpus_dir, 
    max_length=config.max_length
)

g = torch.Generator()
g.manual_seed(42)

sampler = DistributedSampler(
    dataset,
    shuffle=False,
    seed=42,
)

dataloader = DataLoader(
    dataset,
    batch_size=BS,
    sampler=sampler,
    num_workers=24,
    pin_memory=True,
    collate_fn=collate_fn,
    persistent_workers=True
)

model = ViPhonBert(config).to(device)
model = DDP(
    model,
    device_ids=[local_rank],
    output_device=local_rank,
    find_unused_parameters=False
)
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=0.01, betas=(0.9, 0.98), eps=1e-6)

total_steps = 3_000_000
warmup_steps = int(total_steps * 0.01)

if rank == 0:
    wandb.init(
        project="ViPhonBERT",
        name=MODEL_NAME,
        config={
            "batch_size": BS,
            "lr": 5e-5,
            "hidden_size": config.hidden_size,
            "layers": config.num_hidden_layers,
            "heads": config.num_attention_heads,
            "max_length": config.max_length,
            "warmup_steps": warmup_steps,
            "total_steps": total_steps,
        }
    )
    wandb.config.update(vars(config))

def lr_lambda(current_step):
    if current_step < warmup_steps:
        return float(current_step) / float(max(1, warmup_steps))
    return max(0.0, float(total_steps - current_step) / float(max(1, total_steps - warmup_steps)))
    
lr_scheduler = LambdaLR(optimizer, lr_lambda)
scaler = torch.amp.GradScaler('cuda')

start_epoch = 1
global_step = 0
global_batch = 0
checkpoint_path = os.path.join(CHECKPOINT, f"{MODEL_NAME}_training.pth")

if RESUME_FROM_CHECKPOINT and os.path.isfile(checkpoint_path):
    print(f"=> Tìm thấy checkpoint. Đang khôi phục từ: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    start_epoch = checkpoint["epoch"]
    global_step = checkpoint["global_step"]
    global_batch = checkpoint["global_batch"]
    model.module.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    lr_scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    scaler.load_state_dict(checkpoint["scaler_state_dict"])
    
    lr_scheduler.last_epoch = global_step
    print(f"=> Khôi phục thành công! Tiếp tục train từ Epoch {start_epoch}, Step tổng {global_step}")
else:
    print(f"=> Không kích hoạt resume hoặc không tìm thấy file. Bắt đầu pre-train mới từ đầu.")

print(f"Total steps: {total_steps} | Khởi động tại Step: {global_step}")

if not os.path.isdir(CHECKPOINT):
    os.makedirs(CHECKPOINT, exist_ok=True)

model.train()
done = False
while True:
    total_loss = 0
    sampler.set_epoch(start_epoch)

    if rank == 0:
        progress_bar = tqdm(dataloader, desc=f"Epoch {start_epoch}")
    else:
        progress_bar = dataloader

    for batch_idx, batch in enumerate(progress_bar):
        if batch_idx < global_batch:
            continue

        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        optimizer.zero_grad(set_to_none=True)

        should_step = (batch_idx + 1) % ACCUMULATION_STEPS == 0

        context = (
            model.no_sync()
            if not should_step
            else contextlib.nullcontext()
        )
        with context:
            with torch.amp.autocast("cuda"):
                outputs = model(
                    input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                loss = outputs["loss"]
                loss = loss / ACCUMULATION_STEPS
        
        scaler.scale(loss).backward()
        
        if should_step:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            lr_scheduler.step()
            global_step += 1

        loss_value = loss.detach()
        loss_value *= ACCUMULATION_STEPS
        dist.all_reduce(loss_value, op=dist.ReduceOp.SUM)
        loss_value /= dist.get_world_size()
        
        total_loss += loss_value
        if rank == 0:
            progress_bar.set_postfix({
                'loss': f"{loss_value:.4f}",
                'step': global_step,
                'lr': f"{lr_scheduler.get_last_lr()[0]:.2e}"
            })

        if rank == 0:
            wandb.log(
                {
                    "train/loss": loss.item(),
                    "train/lr": lr_scheduler.get_last_lr()[0],
                },
                step=global_step
            )
        
        if global_step > total_steps:
            done = True
            break

        if global_step % 1000 == 0 and rank == 0:
            torch.save({
                "epoch": start_epoch,  
                "global_step": global_step,
                "model_state_dict": model.module.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": lr_scheduler.state_dict(),
                "scaler_state_dict": scaler.state_dict()
            }, checkpoint_path)
            model.module.save_pretrained(os.path.join(CHECKPOINT, f"{MODEL_NAME}"))

    if rank == 0:
        torch.save({
                "epoch": start_epoch,  
                "global_step": global_step,
                "model_state_dict": model.module.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": lr_scheduler.state_dict(),
                "scaler_state_dict": scaler.state_dict()
            }, os.path.join(CHECKPOINT, f"{MODEL_NAME}_training_ep_{start_epoch}.pth"))
        model.module.save_pretrained(os.path.join(CHECKPOINT, f"{MODEL_NAME}_ep_{start_epoch}"))

    avg_loss = total_loss / len(dataloader)
    if rank == 0:
        print(f"Epoch {start_epoch} - Average Loss: {avg_loss:.4f}")
    
    if done:
        break
    
    start_epoch += 1

dist.barrier()
dist.destroy_process_group()
