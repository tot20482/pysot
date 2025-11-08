import os
import random
from tqdm import tqdm
import argparse

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from pysot.utils.lr_scheduler import build_lr_scheduler
from pysot.utils.distributed import (
    dist_init,
    DistModule,
    reduce_gradients,
    average_reduce,
    get_rank,
    get_world_size,
)
from pysot.utils.model_load import load_pretrain
from pysot.utils.average_meter import AverageMeter
from pysot.models.model_builder import ModelBuilder
from pysot.core.config import cfg
from tensorboardX import SummaryWriter

# -------------------- Dataset --------------------
# -------------------- Dataset --------------------
class ProcessedNPZDataset(Dataset):
    def __init__(self, samples_root):
        self.samples_root = samples_root
        self.samples = [
            os.path.join(samples_root, f) for f in os.listdir(samples_root) if f.endswith(".npz")
        ]
        print(f"[Dataset] Loaded {len(self.samples)} samples")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        data = np.load(self.samples[idx])
        
        # Fix label_cls
        label_cls = np.clip(data["label_cls"], 0, 1).astype(np.int64)

        templates = torch.tensor(data["templates"], dtype=torch.float32)
        search = torch.tensor(data["search"], dtype=torch.float32)
        label_loc = torch.tensor(data["label_loc"], dtype=torch.float32)
        label_loc_weight = torch.tensor(data["label_loc_weight"], dtype=torch.float32)
        bbox = torch.tensor(data["bbox"], dtype=torch.float32)

        # Check NaN/Inf
        for t in [templates, search, label_loc, label_loc_weight, bbox]:
            if torch.isnan(t).any() or torch.isinf(t).any():
                print(f"⚠️ NaN/Inf detected in file: {self.samples[idx]}")

        return {
            "templates": templates,
            "search": search,
            "label_cls": torch.tensor(label_cls, dtype=torch.long),
            "label_loc": label_loc,
            "label_loc_weight": label_loc_weight,
            "bbox": bbox,
        }


# -------------------- Seed --------------------
def seed_torch(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# -------------------- DataLoader --------------------
def build_data_loader_npz(samples_root, batch_size, num_workers):
    dataset = ProcessedNPZDataset(samples_root)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
    print(f"✅ DataLoader built with {len(dataset)} samples, batch_size={batch_size}")
    return loader

# -------------------- Optimizer & LR --------------------
def build_opt_lr(model):
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

# -------------------- Training loop --------------------
def train_npz(train_loader, model, optimizer, lr_scheduler, tb_writer, device, world_size):
    model = model.to(device)
    model.train()
    rank = get_rank() if world_size > 1 else 0
    os.makedirs(cfg.TRAIN.SNAPSHOT_DIR, exist_ok=True)

    # Debug device-side assert
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

    for idx, data in enumerate(tqdm(train_loader, desc="Training")):
        # Move to device
        data = {k: v.to(device, non_blocking=True) for k, v in data.items()}

        # Debug: check label_cls
        unique_labels = torch.unique(data["label_cls"])
        if unique_labels.max() > 1 or unique_labels.min() < 0:
            print(f"⚠️ Invalid label in batch {idx}: {unique_labels.tolist()}")
            continue  # skip batch lỗi

        # Skip batch toàn background (label=0)
        if data["label_cls"].max() < 1:
            continue

        # Forward
        outputs = model(data)
        loss = outputs["total_loss"]

        # Backward
        optimizer.zero_grad()
        loss.backward()
        if world_size > 1:
            reduce_gradients(model)

        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.TRAIN.GRAD_CLIP)
        optimizer.step()
        lr_scheduler.step()

        # Logging
        if rank == 0:
            batch_info = {k: v.item() for k, v in outputs.items()}
            for k, v in batch_info.items():
                tb_writer.add_scalar(k, v, idx)

    # Save final checkpoint
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
    parser = argparse.ArgumentParser(description="Train SiamRPN model")
    parser.add_argument("--cfg", type=str, required=True, help="Path to config.yaml")
    args = parser.parse_args()

    # Thiết bị
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    has_gpu = device.type == "cuda"

    # ------------------- CHỈNH Ở ĐÂY -------------------
    rank, world_size = 0, 1
    if has_gpu:
        print(f"🔥 Using {torch.cuda.device_count()} GPU(s)")
    else:
        print("⚙️  No GPU detected — training on CPU")
    # -----------------------------------------------------


    seed_torch(42)

    print(f"📂 Loading config from: {args.cfg}")
    cfg.merge_from_file(args.cfg)

    # Load mô hình
    model = ModelBuilder().to(device).train()

    # ✅ Load pretrained backbone nếu có
    if cfg.BACKBONE.PRETRAINED:
        # Đường dẫn model trong Drive
        backbone_path = "/content/drive/MyDrive/ZaloAI/model.pth"
        if os.path.exists(backbone_path):
            print(f"✅ Loading pretrained backbone from: {backbone_path}")
            load_pretrain(model.backbone, backbone_path)
        else:
            print("⚠️  Pretrained backbone not found")

    # ✅ Sửa lại đường dẫn dataset
    samples_root = "/content/drive/MyDrive/ZaloAI/processed_dataset/processed_dataset/samples"

    if not os.path.exists(samples_root):
        print(f"❌ Dataset path not found: {samples_root}")
        return
    else:
        print(f"📦 Using dataset from: {samples_root}")

    # Tạo DataLoader
    train_loader = build_data_loader_npz(
        samples_root=samples_root,
        batch_size=cfg.TRAIN.BATCH_SIZE,
        num_workers=cfg.TRAIN.NUM_WORKERS,
    )

    # Optimizer + Scheduler
    optimizer, lr_scheduler = build_opt_lr(model)
    tb_writer = SummaryWriter(cfg.TRAIN.LOG_DIR)

    # Distributed / Single-device
    if has_gpu and world_size > 1:
        model = DistModule(model)
        print("✅ Using distributed training")
    else:
        print("✅ Using single-device training")

    # Bắt đầu train
    train_npz(train_loader, model, optimizer, lr_scheduler, tb_writer, device, world_size)



if __name__ == "__main__":
    main()
