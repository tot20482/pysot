import os
import json
import random
import math
import time
import logging
from tqdm import tqdm

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from pysot.utils.lr_scheduler import build_lr_scheduler
from pysot.utils.distributed import get_rank, get_world_size, reduce_gradients, average_reduce, dist_init, DistModule
from pysot.utils.model_load import load_pretrain, restore_from
from pysot.utils.average_meter import AverageMeter
from pysot.models.model_builder import ModelBuilder
from pysot.core.config import cfg
from tensorboardX import SummaryWriter


# -------------------- Dataset --------------------
class ProcessedDataset(Dataset):
    def __init__(self, samples_root, ann_path):
        self.samples_root = samples_root
        with open(ann_path, 'r') as f:
            self.annotations = json.load(f)

        self.samples = []
        for vid, frames in self.annotations.items():
            for frame_id, bbox in frames.items():
                sample_file = os.path.join(samples_root, f"{vid}_f{frame_id}.npz")
                if os.path.exists(sample_file):
                    self.samples.append({
                        "sample_file": sample_file,
                        "bbox": bbox
                    })
        print(f"[Dataset] Loaded {len(self.samples)} samples")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        entry = self.samples[idx]
        data = np.load(entry["sample_file"])
        template = torch.tensor(data["template"], dtype=torch.float32)
        search = torch.tensor(data["search"], dtype=torch.float32)
        bbox = torch.tensor(entry["bbox"], dtype=torch.float32)
        return {"template": template, "search": search, "bbox": bbox}


# -------------------- Seed --------------------
def seed_torch(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# -------------------- Build dataloader --------------------
def build_data_loader(samples_root, ann_path, batch_size, num_workers):
    dataset = ProcessedDataset(samples_root, ann_path)
    sampler = None
    if get_world_size() > 1:
        from torch.utils.data.distributed import DistributedSampler
        sampler = DistributedSampler(dataset)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        sampler=sampler,
        drop_last=True,
    )
    return loader


# -------------------- Optimizer --------------------
def build_opt_lr(model, current_epoch=0):
    for param in model.backbone.parameters():
        param.requires_grad = False
    for m in model.backbone.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.eval()

    trainable_params = [
        {"params": filter(lambda x: x.requires_grad, model.backbone.parameters()),
         "lr": cfg.BACKBONE.LAYERS_LR * cfg.TRAIN.BASE_LR},
        {"params": model.rpn_head.parameters(), "lr": cfg.TRAIN.BASE_LR},
    ]

    optimizer = torch.optim.SGD(
        trainable_params,
        momentum=cfg.TRAIN.MOMENTUM,
        weight_decay=cfg.TRAIN.WEIGHT_DECAY
    )
    lr_scheduler = build_lr_scheduler(optimizer, epochs=cfg.TRAIN.EPOCH)
    lr_scheduler.step(cfg.TRAIN.START_EPOCH)
    return optimizer, lr_scheduler


# -------------------- Training loop --------------------
def train(train_loader, model, optimizer, lr_scheduler, tb_writer):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.train()

    rank = get_rank()
    world_size = get_world_size()
    num_per_epoch = len(train_loader.dataset) // (cfg.TRAIN.BATCH_SIZE * world_size)
    start_epoch = cfg.TRAIN.START_EPOCH
    epoch = start_epoch
    average_meter = AverageMeter()

    os.makedirs(cfg.TRAIN.SNAPSHOT_DIR, exist_ok=True)

    for idx, data in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{cfg.TRAIN.EPOCH}")):
        data = {k: v.to(device, non_blocking=True) for k, v in data.items()}
        outputs = model(data)
        loss = outputs["total_loss"]

        optimizer.zero_grad()
        loss.backward()
        reduce_gradients(model)
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.TRAIN.GRAD_CLIP)
        optimizer.step()

        # Logging
        batch_info = {k: average_reduce(v.item()) for k, v in outputs.items()}
        average_meter.update(**batch_info)

        if rank == 0:
            for k, v in batch_info.items():
                tb_writer.add_scalar(k, v, idx)

        # Save checkpoints at each epoch
        if (idx + 1) % num_per_epoch == 0:
            epoch += 1
            if get_rank() == 0:
                ckpt = {
                    "epoch": epoch,
                    "state_dict": model.module.state_dict(),
                    "optimizer": optimizer.state_dict(),
                }
                torch.save(ckpt, os.path.join(cfg.TRAIN.SNAPSHOT_DIR, f"checkpoint_e{epoch}.pth"))
            if epoch >= cfg.TRAIN.EPOCH:
                break


# -------------------- Main --------------------
# -------------------- Main --------------------
def main():
    # 🔹 Kiểm tra GPU
    has_gpu = torch.cuda.is_available() and torch.cuda.device_count() > 0
    if has_gpu:
        rank, world_size = dist_init()
        device = torch.device("cuda")
        print(f"🔥 Using {torch.cuda.device_count()} GPU(s)")
    else:
        rank, world_size = 0, 1
        device = torch.device("cpu")
        print("⚙️  No GPU detected — training on CPU")

    seed_torch(42)
    cfg.merge_from_file("config.yaml")

    model = ModelBuilder().to(device).train()

    # Load pretrained backbone nếu có
    if cfg.BACKBONE.PRETRAINED:
        backbone_path = "/kaggle/input/mobilenetv2/model.pth"
        if os.path.exists(backbone_path):
            load_pretrain(model.backbone, backbone_path)
        else:
            print("⚠️  Pretrained backbone not found")

    # Build dataloader
    train_loader = build_data_loader(
        samples_root="/kaggle/input/training-data/processed_dataset/samples",
        ann_path="/kaggle/input/training-data/processed_dataset/annotations/annotations.json",
        batch_size=cfg.TRAIN.BATCH_SIZE,
        num_workers=cfg.TRAIN.NUM_WORKERS
    )

    optimizer, lr_scheduler = build_opt_lr(model, cfg.TRAIN.START_EPOCH)
    tb_writer = SummaryWriter(cfg.TRAIN.LOG_DIR)

    # 🔹 Nếu có GPU → dùng DistModule, nếu không → train trực tiếp
    if has_gpu and world_size > 1:
        model = DistModule(model)
        print("✅ Using distributed training")
    else:
        print("✅ Using single-device training")

    train(train_loader, model, optimizer, lr_scheduler, tb_writer)


if __name__ == "__main__":
    main()

