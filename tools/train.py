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
class ProcessedNPZDataset(Dataset):
    def __init__(self, samples_root, max_label=1):
        self.samples_root = samples_root
        self.samples = [
            os.path.join(samples_root, f) for f in os.listdir(samples_root) if f.endswith(".npz")
        ]
        self.max_label = max_label
        print(f"[Dataset] Loaded {len(self.samples)} samples")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path = self.samples[idx]
        data = np.load(path)

        # Clip labels
        label_cls = np.clip(data["label_cls"], 0, self.max_label).astype(np.int64)

        templates = torch.tensor(data["templates"], dtype=torch.float32)
        search = torch.tensor(data["search"], dtype=torch.float32)
        label_loc = torch.tensor(data["label_loc"], dtype=torch.float32)
        label_loc_weight = torch.tensor(data["label_loc_weight"], dtype=torch.float32)
        bbox = torch.tensor(data["bbox"], dtype=torch.float32)

        # Check NaN/Inf
        invalid = False
        for t in [templates, search, label_loc, label_loc_weight, bbox]:
            if torch.isnan(t).any() or torch.isinf(t).any():
                print(f"⚠️ NaN/Inf detected in file: {path}")
                invalid = True

        # Check label_cls
        if label_cls.min() < 0 or label_cls.max() > self.max_label:
            print(f"⚠️ label_cls out of range in file: {path} | min={label_cls.min()}, max={label_cls.max()}")
            invalid = True

        if invalid:
            # Replace with zeros to avoid crash
            templates.zero_()
            search.zero_()
            label_loc.zero_()
            label_loc_weight.zero_()
            bbox.zero_()
            label_cls[:] = 0

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
def build_data_loader_npz(samples_root, batch_size, num_workers, max_label=1):
    dataset = ProcessedNPZDataset(samples_root, max_label=max_label)
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
def train_npz(train_loader, model, optimizer, lr_scheduler, tb_writer, device, world_size, max_label=1):
    model = model.to(device)
    model.train()
    rank = get_rank() if world_size > 1 else 0
    os.makedirs(cfg.TRAIN.SNAPSHOT_DIR, exist_ok=True)

    # Debug device-side assert
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    torch.autograd.set_detect_anomaly(True)

    for idx, data in enumerate(tqdm(train_loader, desc="Training")):
        # Move to device
        data = {k: v.to(device, non_blocking=True) for k, v in data.items()}

        # Skip batch label invalid
        if data["label_cls"].max() > max_label:
            print(f"⚠️ Skipping batch {idx} due to invalid labels: {torch.unique(data['label_cls'])}")
            continue

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

# -------------------- Dataset check function --------------------
def check_dataset(samples_root, max_label=1):
    print(f"📦 Checking dataset: {samples_root}")
    invalid_files = 0
    for f in os.listdir(samples_root):
        if not f.endswith(".npz"):
            continue
        path = os.path.join(samples_root, f)
        data = np.load(path)
        label_cls = np.clip(data["label_cls"], 0, max_label).astype(np.int64)

        min_label, max_label_batch = label_cls.min(), label_cls.max()
        nan_inf = any([np.isnan(data[k]).any() or np.isinf(data[k]).any() for k in ["templates","search","label_loc","label_loc_weight","bbox"]])

        if min_label < 0 or max_label_batch > max_label or nan_inf:
            invalid_files += 1
            print(f"⚠️ Invalid file: {f} | min: {min_label}, max: {max_label_batch}, NaN/Inf: {nan_inf}")

    print(f"✅ Dataset check done. Total files: {len(os.listdir(samples_root))}, Invalid: {invalid_files}")

# -------------------- Main --------------------
def main():
    parser = argparse.ArgumentParser(description="Train SiamRPN model")
    parser.add_argument("--cfg", type=str, required=True, help="Path to config.yaml")
    parser.add_argument("--check_only", action="store_true", help="Only check dataset, do not train")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    has_gpu = device.type == "cuda"

    rank, world_size = 0, 1
    if has_gpu:
        print(f"🔥 Using {torch.cuda.device_count()} GPU(s)")
    else:
        print("⚙️  No GPU detected — training on CPU")

    seed_torch(42)
    print(f"📂 Loading config from: {args.cfg}")
    cfg.merge_from_file(args.cfg)

    # Dataset path
    samples_root = "/content/drive/MyDrive/ZaloAI/processed_dataset/processed_dataset/samples"
    if not os.path.exists(samples_root):
        print(f"❌ Dataset path not found: {samples_root}")
        return
    else:
        print(f"📦 Using dataset from: {samples_root}")

    # -------------------- Check dataset --------------------
    check_dataset(samples_root, max_label=1)
    if args.check_only:
        return

    # Load model
    model = ModelBuilder().to(device).train()
    if cfg.BACKBONE.PRETRAINED:
        backbone_path = "/content/drive/MyDrive/ZaloAI/model.pth"
        if os.path.exists(backbone_path):
            print(f"✅ Loading pretrained backbone from: {backbone_path}")
            load_pretrain(model.backbone, backbone_path)
        else:
            print("⚠️  Pretrained backbone not found")

    # DataLoader
    train_loader = build_data_loader_npz(
        samples_root=samples_root,
        batch_size=cfg.TRAIN.BATCH_SIZE,
        num_workers=cfg.TRAIN.NUM_WORKERS,
        max_label=1
    )

    # Optimizer + Scheduler
    optimizer, lr_scheduler = build_opt_lr(model)
    tb_writer = SummaryWriter(cfg.TRAIN.LOG_DIR)

    if has_gpu and world_size > 1:
        model = DistModule(model)
        print("✅ Using distributed training")
    else:
        print("✅ Using single-device training")

    # Train
    train_npz(train_loader, model, optimizer, lr_scheduler, tb_writer, device, world_size, max_label=1)


if __name__ == "__main__":
    main()
