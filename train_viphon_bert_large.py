import torch
import numpy as np
import random
from itertools import islice
from torch.utils.data import DataLoader, Sampler
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
import shutil
import wandb

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint_path", required=True, type=str)
parser.add_argument("--model_name", default="viphon_bert_large", type=str)
parser.add_argument("--corpus_dir", required=True, type=str)
parser.add_argument("--batch_size", default=128, type=int)
parser.add_argument("--accumulation_steps", default=16, type=int)
parser.add_argument("--total_steps", default=3_000_000, type=int)
parser.add_argument("--save_every", default=1000, type=int)
parser.add_argument("--persistent_checkpoint_path", default=None, type=str)
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

class SkipBatchSampler(Sampler):
    def __init__(self, sampler, batch_size):
        self.sampler = sampler
        self.batch_size = batch_size
        self.skip_batches = 0

    def __iter__(self):
        return islice(iter(self.sampler), self.skip_batches * self.batch_size, None)

    def __len__(self):
        return max(0, len(self.sampler) - self.skip_batches * self.batch_size)

    def set_epoch(self, epoch):
        self.sampler.set_epoch(epoch)

def _fit_state_dict_shapes(state_dict, target_state_dict):
    for key, value in list(state_dict.items()):
        target = target_state_dict.get(key)
        if target is None or not torch.is_tensor(value) or value.shape == target.shape:
            continue
        if value.ndim == target.ndim and all(old >= new for old, new in zip(value.shape, target.shape)):
            state_dict[key] = value[tuple(slice(0, size) for size in target.shape)].clone()

def _fit_optimizer_state_shapes(optimizer):
    for param, state in optimizer.state.items():
        for key, value in list(state.items()):
            if not torch.is_tensor(value) or value.ndim == 0:
                continue
            if value.shape == param.shape:
                continue
            if value.ndim == param.ndim and all(old >= new for old, new in zip(value.shape, param.shape)):
                state[key] = value[tuple(slice(0, size) for size in param.shape)].clone()
            else:
                state.clear()
                break

def _save_checkpoint(state, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp_path = f"{path}.tmp"
    torch.save(state, tmp_path)
    os.replace(tmp_path, path)

def _copy_checkpoint(src_path, dst_dir):
    if not dst_dir:
        return
    os.makedirs(dst_dir, exist_ok=True)
    dst_path = os.path.join(dst_dir, os.path.basename(src_path))
    if os.path.abspath(src_path) == os.path.abspath(dst_path):
        return
    tmp_path = f"{dst_path}.tmp"
    shutil.copy2(src_path, tmp_path)
    os.replace(tmp_path, dst_path)
    print(f"=> Synced checkpoint to persistent path: {dst_path}", flush=True)

BS = args.batch_size
CHECKPOINT = args.checkpoint_path
MODEL_NAME = args.model_name
ACCUMULATION_STEPS = args.accumulation_steps
TRAIN_MAX_LENGTH = 512

# Đổi thành True nếu muốn khôi phục và chạy tiếp từ checkpoint cũ sau khi bị ngắt quãng
RESUME_FROM_CHECKPOINT = True 

device = "cuda" if torch.cuda.is_available() else "cpu"

dist.init_process_group(backend="nccl")
local_rank = int(os.environ["LOCAL_RANK"])
torch.cuda.set_device(local_rank)
device = torch.device("cuda", local_rank)
rank = dist.get_rank()
world_size = dist.get_world_size()
available_cpus = int(os.environ.get("SLURM_CPUS_PER_TASK") or len(os.sched_getaffinity(0)))
num_workers = max(1, available_cpus // world_size)

config = ViPhonBertConfig(
    hidden_size=768,
    num_hidden_layers=12,
    num_attention_heads=12,
    intermediate_size=3072,
    hidden_act="gelu",
    hidden_dropout_prob=0.1,
    attention_probs_dropout_prob=0.1,
    max_position_embeddings=TRAIN_MAX_LENGTH,
    max_length=TRAIN_MAX_LENGTH,
    type_vocab_size=1,
    is_decoder=False,
    add_cross_attention = False
)
tokenizer = ViPhonTokenizer(config)
dataset = ViPhonDataset(
    tokenizer=tokenizer, 
    corpus_dir=args.corpus_dir, 
    max_length=TRAIN_MAX_LENGTH
)

g = torch.Generator()
g.manual_seed(42)

sampler = SkipBatchSampler(DistributedSampler(
    dataset,
    shuffle=False,
    seed=42,
), BS)

dataloader = DataLoader(
    dataset,
    batch_size=BS,
    sampler=sampler,
    num_workers=num_workers,
    pin_memory=True,
    collate_fn=collate_fn,
    persistent_workers=True
)

model = ViPhonBert(config).to(device)
model.bert.gradient_checkpointing_enable({"use_reentrant": False})
model = DDP(
    model,
    device_ids=[local_rank],
    output_device=local_rank,
    find_unused_parameters=False
)
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=0.01, betas=(0.9, 0.999), eps=1e-6)

total_steps = args.total_steps
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
            "max_length": TRAIN_MAX_LENGTH,
            "warmup_steps": warmup_steps,
            "total_steps": total_steps,
        }
    )

def lr_lambda(current_step):
    if current_step < warmup_steps:
        return float(current_step) / float(max(1, warmup_steps))
    return max(0.0, float(total_steps - current_step) / float(max(1, total_steps - warmup_steps)))
    
lr_scheduler = LambdaLR(optimizer, lr_lambda)
scaler = torch.amp.GradScaler('cuda')

start_epoch = 1
global_step = 0
global_batch = 0
resume_batch_offset = 0
checkpoint_path = os.path.join(CHECKPOINT, f"{MODEL_NAME}_training.pth")

if RESUME_FROM_CHECKPOINT and os.path.isfile(checkpoint_path):
    print(f"=> Tìm thấy checkpoint. Đang khôi phục từ: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    start_epoch = checkpoint["epoch"]
    global_step = checkpoint["global_step"]
    global_batch = checkpoint.get("global_batch", min(global_step * ACCUMULATION_STEPS, len(dataloader)))
    resume_batch_offset = global_batch
    sampler.skip_batches = resume_batch_offset
    model_state_dict = checkpoint["model_state_dict"]
    _fit_state_dict_shapes(model_state_dict, model.module.state_dict())
    model.module.load_state_dict(model_state_dict)
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    _fit_optimizer_state_shapes(optimizer)
    lr_scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    scaler.load_state_dict(checkpoint["scaler_state_dict"])
    
    lr_scheduler.last_epoch = global_step
    print(f"=> Khôi phục thành công! Tiếp tục train từ Epoch {start_epoch}, Step tổng {global_step}, Batch {global_batch}")
else:
    print(f"=> Không kích hoạt resume hoặc không tìm thấy file. Bắt đầu pre-train mới từ đầu.")

print(f"Total steps: {total_steps} | Khởi động tại Step: {global_step}")

if not os.path.isdir(CHECKPOINT):
    os.makedirs(CHECKPOINT, exist_ok=True)

model.train()
optimizer.zero_grad(set_to_none=True)
done = False
while True:
    total_loss = 0
    sampler.set_epoch(start_epoch)

    if rank == 0:
        progress_bar = tqdm(dataloader, desc=f"Epoch {start_epoch}")
    else:
        progress_bar = dataloader

    full_batch_idx = resume_batch_offset - 1
    for batch_idx, batch in enumerate(progress_bar):
        full_batch_idx = resume_batch_offset + batch_idx

        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        should_step = (full_batch_idx + 1) % ACCUMULATION_STEPS == 0

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
        
        optimizer_stepped = False
        if should_step:
            old_scale = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            optimizer_stepped = scaler.get_scale() >= old_scale
            optimizer.zero_grad(set_to_none=True)
            if optimizer_stepped:
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
        
        if optimizer_stepped and global_step % args.save_every == 0 and rank == 0:
            _save_checkpoint({
                "epoch": start_epoch,  
                "global_step": global_step,
                "global_batch": full_batch_idx + 1,
                "model_state_dict": model.module.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": lr_scheduler.state_dict(),
                "scaler_state_dict": scaler.state_dict()
            }, checkpoint_path)
            _copy_checkpoint(checkpoint_path, args.persistent_checkpoint_path)
            model.module.save_pretrained(os.path.join(CHECKPOINT, f"{MODEL_NAME}"))

        if global_step >= total_steps:
            done = True
            break

    if rank == 0:
        torch.save({
                "epoch": start_epoch,  
                "global_step": global_step,
                "global_batch": full_batch_idx + 1,
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
    
    global_batch = 0
    resume_batch_offset = 0
    sampler.skip_batches = 0
    start_epoch += 1

dist.barrier()
dist.destroy_process_group()
