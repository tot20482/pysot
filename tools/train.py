# train.py - Sửa lỗi crash THPVariable_subclass_dealloc
# Copyright (c) SenseTime. All Rights Reserved.

from __future__ import absolute_import, division, print_function, unicode_literals

import argparse
import logging
import os
import time
import math
import random
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from torch.nn.utils import clip_grad_norm_

from pysot.utils.lr_scheduler import build_lr_scheduler
from pysot.utils.log_helper import init_log, print_speed, add_file_handler
from pysot.utils.model_load import load_pretrain, restore_from
from pysot.utils.average_meter import AverageMeter
from pysot.utils.misc import commit
from pysot.models.model_builder import ModelBuilder
from pysot.datasets.dataset import TrkDataset
from pysot.core.config import cfg

logger = logging.getLogger('global')

parser = argparse.ArgumentParser(description='SiamRPN training')
parser.add_argument('--cfg', type=str, default='config.yaml', help='config file path')
parser.add_argument('--seed', type=int, default=123456, help='random seed')
args = parser.parse_args()


def seed_torch(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def build_data_loader():
    logger.info("Building dataset...")
    train_dataset = TrkDataset(
        samples_root="/kaggle/input/zaloai2025-aeroeyes/observing/train/samples",
        ann_path="/kaggle/input/annotation/output.json"
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.TRAIN.BATCH_SIZE,
        num_workers=0,       # fix crash THPVariable_subclass_dealloc
        pin_memory=False,
        shuffle=True
    )
    return train_loader


def build_optimizer_scheduler(model, current_epoch=0):
    # Freeze backbone first
    for param in model.backbone.parameters():
        param.requires_grad = False
    for m in model.backbone.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.eval()
    # Unfreeze layers
    if current_epoch >= cfg.BACKBONE.TRAIN_EPOCH:
        for layer in cfg.BACKBONE.TRAIN_LAYERS:
            for param in getattr(model.backbone, layer).parameters():
                param.requires_grad = True
            for m in getattr(model.backbone, layer).modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.train()

    params = [{'params': filter(lambda x: x.requires_grad, model.backbone.parameters()),
               'lr': cfg.BACKBONE.LAYERS_LR * cfg.TRAIN.BASE_LR}]
    if cfg.ADJUST.ADJUST:
        params.append({'params': model.neck.parameters(), 'lr': cfg.TRAIN.BASE_LR})
    params.append({'params': model.rpn_head.parameters(), 'lr': cfg.TRAIN.BASE_LR})
    if cfg.MASK.MASK:
        params.append({'params': model.mask_head.parameters(), 'lr': cfg.TRAIN.BASE_LR})
    if cfg.REFINE.REFINE:
        params.append({'params': model.refine_head.parameters(), 'lr': cfg.TRAIN.BASE_LR})

    optimizer = torch.optim.SGD(params, momentum=cfg.TRAIN.MOMENTUM, weight_decay=cfg.TRAIN.WEIGHT_DECAY)
    scheduler = build_lr_scheduler(optimizer, epochs=cfg.TRAIN.EPOCH)
    scheduler.step(cfg.TRAIN.START_EPOCH)
    return optimizer, scheduler


def train(train_loader, model, optimizer, scheduler, tb_writer=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    average_meter = AverageMeter()
    start_epoch = cfg.TRAIN.START_EPOCH
    max_epoch = cfg.TRAIN.EPOCH
    epoch = start_epoch

    num_per_epoch = len(train_loader.dataset) // cfg.TRAIN.BATCH_SIZE
    end = time.time()

    if not os.path.exists(cfg.TRAIN.SNAPSHOT_DIR):
        os.makedirs(cfg.TRAIN.SNAPSHOT_DIR)

    for idx, data in tqdm(enumerate(train_loader), total=len(train_loader), desc="Training", ncols=100):
        data = {k: v.to(device, non_blocking=True) for k, v in data.items()}

        # Current epoch
        new_epoch = idx // num_per_epoch + start_epoch
        if new_epoch != epoch:
            epoch = new_epoch
            # Save checkpoint
            ckpt_path = os.path.join(cfg.TRAIN.SNAPSHOT_DIR, f'checkpoint_e{epoch}.pth')
            torch.save({
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict()
            }, ckpt_path)
            logger.info(f"Saved checkpoint: {ckpt_path}")

            if epoch >= max_epoch:
                logger.info("Training complete.")
                break

            if epoch == cfg.BACKBONE.TRAIN_EPOCH:
                optimizer, scheduler = build_optimizer_scheduler(model, epoch)

            scheduler.step()
            cur_lr = scheduler.get_cur_lr()
            logger.info(f"Epoch {epoch + 1}, LR: {cur_lr}")

        # Forward
        outputs = model(data)
        loss = outputs['total_loss']

        if torch.isnan(loss) or torch.isinf(loss) or loss.item() > 1e4:
            tqdm.write(f"Skipping step {idx+1} due to invalid loss: {loss.item()}")
            continue

        optimizer.zero_grad()
        loss.backward()
        clip_grad_norm_(model.parameters(), cfg.TRAIN.GRAD_CLIP)
        optimizer.step()

        # Logging
        batch_time = time.time() - end
        end = time.time()
        batch_info = {'batch_time': batch_time}
        for k, v in outputs.items():
            batch_info[k] = v.item()
        average_meter.update(**batch_info)

        if (idx + 1) % cfg.TRAIN.PRINT_FREQ == 0:
            logger.info(f"[Epoch {epoch+1}][{(idx+1)%num_per_epoch}/{num_per_epoch}] "
                        f"loss: {loss.item():.4f}, batch_time: {batch_time:.2f}s")
            print_speed(idx+1, average_meter.batch_time.avg, max_epoch*num_per_epoch)

        if tb_writer and (idx + 1) % 10 == 0:
            tb_writer.add_scalar('loss/total', loss.item(), idx)


def main():
    cfg.merge_from_file(args.cfg)

    if not os.path.exists(cfg.TRAIN.LOG_DIR):
        os.makedirs(cfg.TRAIN.LOG_DIR)
    init_log('global', logging.INFO)
    add_file_handler('global', os.path.join(cfg.TRAIN.LOG_DIR, 'logs.txt'), logging.INFO)
    logger.info(f"Version: {commit()}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ModelBuilder().to(device).train()

    # Load pretrained backbone
    if cfg.BACKBONE.PRETRAINED:
        backbone_path = cfg.BACKBONE.PRETRAINED
        if os.path.exists(backbone_path):
            load_pretrain(model.backbone, backbone_path)
        else:
            raise FileNotFoundError(f"Pretrained backbone not found: {backbone_path}")

    tb_writer = SummaryWriter(cfg.TRAIN.LOG_DIR) if cfg.TRAIN.LOG_DIR else None
    train_loader = build_data_loader()
    optimizer, scheduler = build_optimizer_scheduler(model, cfg.TRAIN.START_EPOCH)

    # Resume training
    if cfg.TRAIN.RESUME:
        model, optimizer, cfg.TRAIN.START_EPOCH = restore_from(model, optimizer, cfg.TRAIN.RESUME)
    elif cfg.TRAIN.PRETRAINED:
        load_pretrain(model, cfg.TRAIN.PRETRAINED)

    logger.info("Model ready ✅")
    train(train_loader, model, optimizer, scheduler, tb_writer)


if __name__ == '__main__':
    seed_torch(args.seed)
    main()
