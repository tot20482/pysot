# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, unicode_literals
import argparse, logging, os, time, math, json, random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from torch.nn.utils import clip_grad_norm_

from pysot.utils.lr_scheduler import build_lr_scheduler
from pysot.utils.log_helper import init_log, print_speed, add_file_handler
from pysot.utils.model_load import load_pretrain, restore_from
from pysot.utils.average_meter import AverageMeter
from pysot.utils.misc import describe, commit
from pysot.models.model_builder import ModelBuilder
from pysot.datasets.dataset import TrkDataset
from pysot.core.config import cfg

logger = logging.getLogger('global')
parser = argparse.ArgumentParser(description='siamrpn tracking')
parser.add_argument('--cfg', type=str, default='config.yaml')
parser.add_argument('--seed', type=int, default=123456)
args = parser.parse_args()

def seed_torch(seed=0):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False

def build_data_loader():
    logger.info("Building train dataset...")
    train_dataset = TrkDataset(
        samples_root="/kaggle/input/zaloai2025-aeroeyes/observing/train/samples",
        ann_path="/kaggle/input/annotation/output.json"
    )
    logger.info(f"Dataset built successfully. Found {len(train_dataset)} samples.")

    # Tăng batch size tối đa bằng số samples nếu dataset nhỏ
    batch_size = min(cfg.TRAIN.BATCH_SIZE, len(train_dataset))
    num_workers = max(cfg.TRAIN.NUM_WORKERS, 2)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        shuffle=True
    )
    logger.info(f"Using batch size: {batch_size}, num_workers: {num_workers}")
    return train_loader

def build_optimizer_lr(model):
    for param in model.backbone.parameters():
        param.requires_grad = False
    for m in model.backbone.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.eval()
    if cfg.BACKBONE.TRAIN_EPOCH <= cfg.TRAIN.START_EPOCH:
        for layer in cfg.BACKBONE.TRAIN_LAYERS:
            for param in getattr(model.backbone, layer).parameters():
                param.requires_grad = True
            for m in getattr(model.backbone, layer).modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.train()

    trainable_params = [{'params': filter(lambda x: x.requires_grad, model.backbone.parameters()),
                         'lr': cfg.BACKBONE.LAYERS_LR * cfg.TRAIN.BASE_LR}]
    if cfg.ADJUST.ADJUST:
        trainable_params += [{'params': model.neck.parameters(), 'lr': cfg.TRAIN.BASE_LR}]
    trainable_params += [{'params': model.rpn_head.parameters(), 'lr': cfg.TRAIN.BASE_LR}]
    if cfg.MASK.MASK:
        trainable_params += [{'params': model.mask_head.parameters(), 'lr': cfg.TRAIN.BASE_LR}]
    if cfg.REFINE.REFINE:
        trainable_params += [{'params': model.refine_head.parameters(), 'lr': cfg.TRAIN.LR.BASE_LR}]

    optimizer = torch.optim.SGD(trainable_params, momentum=cfg.TRAIN.MOMENTUM, weight_decay=cfg.TRAIN.WEIGHT_DECAY)
    lr_scheduler = build_lr_scheduler(optimizer, epochs=cfg.TRAIN.EPOCH)
    return optimizer, lr_scheduler

def train(train_loader, model, optimizer, lr_scheduler, tb_writer):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.train()
    logger.info(f"Using device: {device}")
    average_meter = AverageMeter()
    end = time.time()

    for idx, data in enumerate(train_loader):
        # Chuyển toàn bộ batch sang GPU
        for k in data:
            if isinstance(data[k], torch.Tensor):
                data[k] = data[k].to(device, non_blocking=True)

        # Forward
        outputs = model(data)
        loss = outputs['total_loss']

        if not (math.isnan(loss.item()) or math.isinf(loss.item())):
            optimizer.zero_grad()
            loss.backward()
            clip_grad_norm_(model.parameters(), cfg.TRAIN.GRAD_CLIP)
            optimizer.step()
            lr_scheduler.step()  # cập nhật learning rate

        batch_time = time.time() - end
        end = time.time()
        average_meter.update(batch_time=batch_time, loss=loss.item())

        if idx % cfg.TRAIN.PRINT_FREQ == 0:
            logger.info(f"Step {idx}, Loss: {loss.item():.4f}, Batch time: {batch_time:.3f}s")
            if tb_writer:
                tb_writer.add_scalar('train/loss', loss.item(), idx)
                tb_writer.add_scalar('train/batch_time', batch_time, idx)
                mem_alloc = torch.cuda.memory_allocated() / 1024**2
                mem_reserved = torch.cuda.memory_reserved() / 1024**2
                tb_writer.add_scalar('gpu/memory_alloc_MB', mem_alloc, idx)
                tb_writer.add_scalar('gpu/memory_reserved_MB', mem_reserved, idx)

def main():
    seed_torch(args.seed)
    cfg.merge_from_file(args.cfg)
    if not os.path.exists(cfg.TRAIN.LOG_DIR):
        os.makedirs(cfg.TRAIN.LOG_DIR)
    init_log('global', logging.INFO)
    add_file_handler('global', os.path.join(cfg.TRAIN.LOG_DIR, 'logs.txt'), logging.INFO)
    logger.info("Version Information: \n{}\n".format(commit()))
    logger.info("Config: \n{}".format(json.dumps(cfg, indent=4)))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ModelBuilder().to(device)

    # Load pretrained backbone
    if cfg.BACKBONE.PRETRAINED:
        backbone_path = "/kaggle/input/mobilenetv2/model.pth"
        if os.path.exists(backbone_path) and os.path.getsize(backbone_path) > 0:
            load_pretrain(model.backbone, backbone_path)
        else:
            raise FileNotFoundError(f"Pretrained backbone not found: {backbone_path}")

    tb_writer = SummaryWriter(cfg.TRAIN.LOG_DIR) if cfg.TRAIN.LOG_DIR else None

    train_loader = build_data_loader()
    optimizer, lr_scheduler = build_optimizer_lr(model)

    # Resume training if needed
    if cfg.TRAIN.RESUME:
        model, optimizer, cfg.TRAIN.START_EPOCH = restore_from(model, optimizer, cfg.TRAIN.RESUME)
    elif cfg.TRAIN.PRETRAINED:
        load_pretrain(model, cfg.TRAIN.PRETRAINED)

    train(train_loader, model, optimizer, lr_scheduler, tb_writer)

if __name__ == '__main__':
    main()
