import os
import random
from tqdm import tqdm

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import argparse

from pysot.utils.lr_scheduler import build_lr_scheduler
from torch.utils.data.distributed import DistributedSampler
from pysot.utils.distributed import get_rank, get_world_size, reduce_gradients, average_reduce, dist_init, DistModule
from pysot.utils.model_load import load_pretrain
from pysot.utils.average_meter import AverageMeter
from pysot.models.model_builder import ModelBuilder
from pysot.core.config import cfg
from tensorboardX import SummaryWriter

# -------------------- Dataset cho .npz PySOT --------------------
# -------------------- Dataset cho .npz PySOT --------------------
class ProcessedNPZDataset(Dataset):
    def __init__(self, samples_root):
        self.samples_root = samples_root
        self.samples = [
            os.path.join(samples_root, f) 
            for f in os.listdir(samples_root) if f.endswith(".npz")
        ]
        print(f"[Dataset] Loaded {len(self.samples)} samples")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        data = np.load(self.samples[idx])
        sample = {
            "templates": torch.tensor(data["templates"], dtype=torch.float32),
            "search": torch.tensor(data["search"], dtype=torch.float32),
            # Chuyển label_cls sang LongTensor
            "label_cls": torch.tensor(data["label_cls"], dtype=torch.long),
            "label_loc": torch.tensor(data["label_loc"], dtype=torch.float32),
            "label_loc_weight": torch.tensor(data["label_loc_weight"], dtype=torch.float32),
            "bbox": torch.tensor(data["bbox"], dtype=torch.float32),
        }
        return sample


# -------------------- Seed --------------------
def seed_torch(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def is_dist_avail_and_initialized():
    return torch.distributed.is_available() and torch.distributed.is_initialized()

def get_train_info():
    """Return rank and world_size, compatible CPU / single GPU / multi GPU"""
    if is_dist_avail_and_initialized():
        return get_rank(), get_world_size()
    else:
        return 0, 1

# -------------------- DataLoader --------------------
def build_data_loader_npz(samples_root, batch_size, num_workers):
    dataset = ProcessedNPZDataset(samples_root)
    sampler = None
    if is_dist_avail_and_initialized() and get_world_size() > 1:
        sampler = DistributedSampler(dataset)
        print(f"🧩 Using DistributedSampler with {get_world_size()} processes")

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        sampler=sampler,
        shuffle=(sampler is None),
        drop_last=True,
    )
    print(f"✅ DataLoader built with {len(dataset)} samples, batch_size={batch_size}")
    return loader

# -------------------- Optimizer --------------------
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
        weight_decay=cfg.TRAIN.WEIGHT_DECAY
    )
    lr_scheduler = build_lr_scheduler(optimizer, epochs=cfg.TRAIN.EPOCH)
    return optimizer, lr_scheduler

# -------------------- Training loop --------------------
def train_npz(train_loader, model, optimizer, lr_scheduler, tb_writer, device):
    model = model.to(device)
    model.train()
    rank, world_size = get_train_info()
    num_per_epoch = max(len(train_loader.dataset) // (cfg.TRAIN.BATCH_SIZE * world_size), 1)
    epoch = cfg.TRAIN.START_EPOCH
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
        optimizer.step()          # step optimizer trước
        lr_scheduler.step()       # step scheduler sau optimizer

        batch_info = {k: average_reduce(v.item()) for k, v in outputs.items()}
        average_meter.update(**batch_info)

        if rank == 0:
            for k, v in batch_info.items():
                tb_writer.add_scalar(k, v, idx)

        if (idx + 1) % num_per_epoch == 0:
            epoch += 1
            if rank == 0:
                ckpt = {
                    "epoch": epoch,
                    "state_dict": model.module.state_dict() if hasattr(model, "module") else model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                }
                torch.save(ckpt, os.path.join(cfg.TRAIN.SNAPSHOT_DIR, f"checkpoint_e{epoch}.pth"))
            if epoch >= cfg.TRAIN.EPOCH:
                break

# -------------------- Main --------------------
def main():
    parser = argparse.ArgumentParser(description="Train SiamRPN model")
    parser.add_argument("--cfg", type=str, required=True, help="Path to config.yaml")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    has_gpu = device.type == "cuda"
    if has_gpu:
        rank, world_size = dist_init()
        print(f"🔥 Using {torch.cuda.device_count()} GPU(s)")
    else:
        rank, world_size = 0, 1
        print("⚙️  No GPU detected — training on CPU")

    seed_torch(42)

    print(f"📂 Loading config from: {args.cfg}")
    cfg.merge_from_file(args.cfg)

    model = ModelBuilder().to(device).train()

    if cfg.BACKBONE.PRETRAINED:
        backbone_path = "/kaggle/input/mobilenetv2/model.pth"
        if os.path.exists(backbone_path):
            load_pretrain(model.backbone, backbone_path)
        else:
            print("⚠️  Pretrained backbone not found")

    train_loader = build_data_loader_npz(
        samples_root="/kaggle/input/training-data/processed_dataset/samples",
        batch_size=cfg.TRAIN.BATCH_SIZE,
        num_workers=cfg.TRAIN.NUM_WORKERS
    )

    optimizer, lr_scheduler = build_opt_lr(model)
    tb_writer = SummaryWriter(cfg.TRAIN.LOG_DIR)

    if has_gpu and world_size > 1:
        model = DistModule(model)
        print("✅ Using distributed training")
    else:
        print("✅ Using single-device training")

    train_npz(train_loader, model, optimizer, lr_scheduler, tb_writer, device)


if __name__ == "__main__":
    main()
