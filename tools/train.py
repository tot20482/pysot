# Copyright (c) SenseTime. All Rights Reserved.

from __future__ import absolute_import, division, print_function, unicode_literals
import argparse, logging, os, time, math, json, random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.utils import clip_grad_norm_
from tensorboardX import SummaryWriter

from pysot.utils.lr_scheduler import build_lr_scheduler
from pysot.utils.log_helper import init_log, add_file_handler, describe, commit
from pysot.utils.distributed import dist_init, DistModule, reduce_gradients, get_rank, get_world_size
from pysot.utils.model_load import load_pretrain, restore_from
from pysot.utils.average_meter import AverageMeter
from pysot.models.model_builder import ModelBuilder
from pysot.datasets.dataset import TrkDataset
from pysot.core.config import cfg

logger = logging.getLogger('global')
parser = argparse.ArgumentParser(description='siamrpn tracking')
parser.add_argument('--cfg', type=str, default='config.yaml')
parser.add_argument('--seed', type=int, default=123456)
parser.add_argument('--local_rank', type=int, default=0)
args = parser.parse_args()


def seed_torch(seed=0):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def build_data_loader():
    logger.info("Building train dataset")
    train_dataset = TrkDataset(
        samples_root="/kaggle/input/zaloai2025-aeroeyes/observing/train/samples",
        ann_path="/kaggle/input/annotation/output.json"
    )
    train_sampler = None
    if get_world_size() > 1:
        train_sampler = DistributedSampler(train_dataset)
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.TRAIN.BATCH_SIZE,
        num_workers=cfg.TRAIN.NUM_WORKERS,
        pin_memory=True,
        sampler=train_sampler,
        drop_last=True
    )
    logger.info("Train dataset built")
    return train_loader


def build_optimizer_lr(model, current_epoch=0):
    # Freeze backbone layers
    for param in model.backbone.parameters():
        param.requires_grad = False
    for m in model.backbone.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.eval()
    if current_epoch >= cfg.BACKBONE.TRAIN_EPOCH:
        for layer in cfg.BACKBONE.TRAIN_LAYERS:
            for param in getattr(model.backbone, layer).parameters():
                param.requires_grad = True
            for m in getattr(model.backbone, layer).modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.train()

    trainable_params = [{'params': filter(lambda x: x.requires_grad, model.backbone.parameters()),
                         'lr': cfg.BACKBONE.LAYERS_LR * cfg.TRAIN.BASE_LR}]
    if cfg.ADJUST.ADJUST:
        trainable_params.append({'params': model.neck.parameters(), 'lr': cfg.TRAIN.BASE_LR})
    trainable_params.append({'params': model.rpn_head.parameters(), 'lr': cfg.TRAIN.BASE_LR})
    if cfg.MASK.MASK:
        trainable_params.append({'params': model.mask_head.parameters(), 'lr': cfg.TRAIN.BASE_LR})
    if cfg.REFINE.REFINE:
        trainable_params.append({'params': model.refine_head.parameters(), 'lr': cfg.TRAIN.LR.BASE_LR})

    optimizer = torch.optim.SGD(
        trainable_params,
        momentum=cfg.TRAIN.MOMENTUM,
        weight_decay=cfg.TRAIN.WEIGHT_DECAY
    )

    lr_scheduler = build_lr_scheduler(optimizer, epochs=cfg.TRAIN.EPOCH)
    lr_scheduler.step(cfg.TRAIN.START_EPOCH)
    return optimizer, lr_scheduler


def move_to_device(batch, device):
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            batch[k] = v.to(device, non_blocking=True)
    return batch


def train(train_loader, model, optimizer, lr_scheduler, tb_writer):
    device = next(model.parameters()).device  # đảm bảo model device
    rank = get_rank()
    world_size = get_world_size()
    average_meter = AverageMeter()
    end = time.time()

    for idx, data in enumerate(train_loader):
        data = move_to_device(data, device)

        # Log GPU memory
        if idx % 100 == 0 and torch.cuda.is_available():
            mem_alloc = torch.cuda.memory_allocated() / 1024**2
            mem_reserved = torch.cuda.memory_reserved() / 1024**2
            logger.info(f"[GPU] Step {idx} | Allocated: {mem_alloc:.2f} MB | Reserved: {mem_reserved:.2f} MB")
            if tb_writer:
                tb_writer.add_scalar("gpu/memory_alloc_MB", mem_alloc, idx)
                tb_writer.add_scalar("gpu/memory_reserved_MB", mem_reserved, idx)

        outputs = model(data)
        loss = outputs['total_loss']

        if not (math.isnan(loss.item()) or math.isinf(loss.item())):
            optimizer.zero_grad()
            loss.backward()
            reduce_gradients(model)
            clip_grad_norm_(model.parameters(), cfg.TRAIN.GRAD_CLIP)
            optimizer.step()

        batch_time = time.time() - end
        end = time.time()


def main():
    # Initialize distributed training
    rank, world_size = dist_init()
    torch.cuda.set_device(args.local_rank)
    device = torch.device(f"cuda:{args.local_rank}" if torch.cuda.is_available() else "cpu")
    logger.info("Distributed init done")

    # Config
    cfg.merge_from_file(args.cfg)
    if rank == 0:
        if not os.path.exists(cfg.TRAIN.LOG_DIR):
            os.makedirs(cfg.TRAIN.LOG_DIR)
        init_log('global', logging.INFO)
        add_file_handler('global', os.path.join(cfg.TRAIN.LOG_DIR, 'logs.txt'), logging.INFO)
        logger.info("Version Information: \n{}\n".format(commit()))
        logger.info("Config: \n{}".format(json.dumps(cfg, indent=4)))

    # Build model
    model = ModelBuilder().to(device).train()

    if cfg.BACKBONE.PRETRAINED:
        backbone_path = "/kaggle/input/mobilenetv2/model.pth"
        if os.path.exists(backbone_path) and os.path.getsize(backbone_path) > 0:
            load_pretrain(model.backbone, backbone_path)
        else:
            raise FileNotFoundError(f"Pretrained backbone not found: {backbone_path}")

    # Wrap distributed
    dist_model = DistModule(model)

    # Tensorboard
    tb_writer = SummaryWriter(cfg.TRAIN.LOG_DIR) if rank == 0 else None

    # DataLoader
    train_loader = build_data_loader()

    # Optimizer & LR scheduler
    optimizer, lr_scheduler = build_optimizer_lr(model, cfg.TRAIN.START_EPOCH)

    # Resume / Pretrained
    if cfg.TRAIN.RESUME:
        model, optimizer, cfg.TRAIN.START_EPOCH = restore_from(model, optimizer, cfg.TRAIN.RESUME)
    elif cfg.TRAIN.PRETRAINED:
        load_pretrain(model, cfg.TRAIN.PRETRAINED)

    # Start training
    train(train_loader, dist_model, optimizer, lr_scheduler, tb_writer)


if __name__ == '__main__':
    seed_torch(args.seed)
    main()
