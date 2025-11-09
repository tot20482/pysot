# train.py
# Copyright (c) SenseTime. All Rights Reserved.

from __future__ import absolute_import, division, print_function, unicode_literals

import os
import random
import argparse
from tqdm import tqdm

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tensorboardX import SummaryWriter

from pysot.utils.lr_scheduler import build_lr_scheduler
from pysot.utils.distributed import get_rank, get_world_size, reduce_gradients, DistModule
from pysot.utils.model_load import load_pretrain
from pysot.models.model_builder import ModelBuilder
from pysot.core.config import cfg

# -------------------- Safe Dataset --------------------
class FilteredNPZDataset(Dataset):
    def __init__(self, samples_root):
        self.samples_root = samples_root
        self.samples = []

        # Tính max_label theo OUTPUT_SIZE và số anchor
        self.max_label = cfg.TRAIN.OUTPUT_SIZE**2 * cfg.ANCHOR.ANCHOR_NUM - 1
        print(f"[Dataset] max_label computed: {self.max_label}")

        # Scan tất cả npz files và filter invalid
        for f in os.listdir(samples_root):
            if not f.endswith(".npz"):
                continue
            path = os.path.join(samples_root, f)
            data = np.load(path)

            label_cls = data["label_cls"]
            # Check NaN/Inf
            nan_inf = any([np.isnan(data[k]).any() or np.isinf(data[k]).any()
                           for k in ["templates","search","label_loc","label_loc_weight","bbox"]])
            # Check label range
            if label_cls.min() < 0 or label_cls.max() > self.max_label or nan_inf:
                print(f"⚠️ Skipping invalid file: {f} | min={label_cls.min()}, max={label_cls.max()}, NaN/Inf={nan_inf}")
                continue
            self.samples.append(path)

        print(f"[Dataset] Loaded {len(self.samples)} valid samples from {samples_root}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path = self.samples[idx]
        data = np.load(path)

        templates = torch.tensor(data["templates"], dtype=torch.float32)
        search = torch.tensor(data["search"], dtype=torch.float32)
        label_loc = torch.tensor(data["label_loc"], dtype=torch.float32)
        label_loc_weight = torch.tensor(data["label_loc_weight"], dtype=torch.float32)
        bbox = torch.tensor(data["bbox"], dtype=torch.float32)
        label_cls = torch.tensor(data["label_cls"], dtype=torch.long)
        label_cls = torch.clamp(label_cls, 0, self.max_label)

        return {
            "templates": templates,
            "search": search,
            "label_cls": label_cls,
            "label_loc": label_loc,
            "label_loc_weight": label_loc_weight,
            "bbox": bbox
        }

# -------------------- DataLoader --------------------
def build_filtered_loader(samples_root, batch_size, num_workers):
    dataset = FilteredNPZDataset(samples_root)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )
    print(f"✅ DataLoader built with {len(dataset)} samples, batch_size={batch_size}")
    return loader

# -------------------- Optimizer & LR --------------------
def build_opt_lr(model):
    # Freeze backbone
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
        weight_decay=cfg.TRAIN.WEIGHT_DECAY,
    )
    lr_scheduler = build_lr_scheduler(optimizer, epochs=cfg.TRAIN.EPOCH)
    return optimizer, lr_scheduler

# -------------------- Seed --------------------
def seed_torch(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# -------------------- Safe Training Loop --------------------
def train_filtered(train_loader, model, optimizer, lr_scheduler, tb_writer, device, world_size):
    model = model.to(device)
    model.train()
    rank = get_rank() if world_size > 1 else 0
    os.makedirs(cfg.TRAIN.SNAPSHOT_DIR, exist_ok=True)

    max_label = cfg.TRAIN.OUTPUT_SIZE**2 * cfg.ANCHOR.ANCHOR_NUM - 1

    for idx, data in enumerate(tqdm(train_loader, desc="Training")):
        # Move to device
        data = {k: v.to(device, non_blocking=True) for k, v in data.items()}
        data["label_cls"] = torch.clamp(data["label_cls"], 0, max_label)

        # Skip batch toàn background
        if data["label_cls"].max() < 1:
            continue

        try:
            outputs = model(data)
            loss = outputs["total_loss"]

            optimizer.zero_grad()
            loss.backward()
            if world_size > 1:
                reduce_gradients(model)

            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.TRAIN.GRAD_CLIP)
            optimizer.step()
            lr_scheduler.step()

            if rank == 0:
                for k, v in outputs.items():
                    tb_writer.add_scalar(k, v.item(), idx)

        except RuntimeError as e:
            print(f"⚠️ RuntimeError at batch {idx}: {e}")
            continue

    if rank == 0:
        final_ckpt_path = os.path.join(cfg.TRAIN.SNAPSHOT_DIR, "checkpoint_final.pth")
        ckpt = {
            "state_dict": model.module.state_dict() if hasattr(model, "module") else model.state_dict(),
            "optimizer": optimizer.state_dict(),
        }
        torch.save(ckpt, final_ckpt_path)
        print(f"✅ Training completed. Final checkpoint saved: {final_ckpt_path}")

# -------------------- Main --------------------
def main():
    parser = argparse.ArgumentParser(description="Train SiamRPN safely")
    parser.add_argument("--cfg", type=str, required=True, help="Path to config.yaml")
    parser.add_argument("--check_only", action="store_true", help="Only check dataset")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    has_gpu = device.type == "cuda"

    rank, world_size = 0, 1
    if has_gpu:
        print(f"🔥 Using {torch.cuda.device_count()} GPU(s)")
    else:
        print("⚙️ No GPU detected — training on CPU")

    seed_torch(42)
    cfg.merge_from_file(args.cfg)

    # Compute OUTPUT_SIZE tự động nếu chưa đúng
    computed_output_size = (cfg.TRAIN.SEARCH_SIZE - cfg.TRAIN.EXEMPLAR_SIZE) // cfg.ANCHOR.STRIDE + 1
    if cfg.TRAIN.OUTPUT_SIZE != computed_output_size:
        print(f"⚠️ Adjusting TRAIN.OUTPUT_SIZE from {cfg.TRAIN.OUTPUT_SIZE} to {computed_output_size}")
        cfg.TRAIN.OUTPUT_SIZE = computed_output_size

    samples_root = "/kaggle/input/dataset/processed_dataset/samples"
    if not os.path.exists(samples_root):
        print(f"❌ Dataset path not found: {samples_root}")
        return

    train_loader = build_filtered_loader(samples_root, batch_size=cfg.TRAIN.BATCH_SIZE,
                                         num_workers=cfg.TRAIN.NUM_WORKERS)
    if args.check_only:
        return

    model = ModelBuilder().to(device).train()
    if cfg.BACKBONE.PRETRAINED:
        backbone_path = "/kaggle/input/alexnet/model.pth"
        if os.path.exists(backbone_path):
            load_pretrain(model.backbone, backbone_path)

    optimizer, lr_scheduler = build_opt_lr(model)
    tb_writer = SummaryWriter(cfg.TRAIN.LOG_DIR)

    if has_gpu and world_size > 1:
        model = DistModule(model)
        print("✅ Using distributed training")
    else:
        print("✅ Using single-device training")

    train_filtered(train_loader, model, optimizer, lr_scheduler, tb_writer, device, world_size)

if __name__ == "__main__":
    main()
