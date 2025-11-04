# Copyright (c) SenseTime. All Rights Reserved.

from __future__ import absolute_import, division, print_function, unicode_literals

import argparse, logging, os, time, math, json, random, numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_
from torch.utils.data.distributed import DistributedSampler
from tensorboardX import SummaryWriter

from pysot.utils.lr_scheduler import build_lr_scheduler
from pysot.utils.log_helper import init_log, print_speed, add_file_handler
from pysot.utils.distributed import dist_init, DistModule, reduce_gradients, get_rank, get_world_size
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
    logger.info("Build train dataset")
    train_dataset = TrkDataset(samples_root='/kaggle/input/zaloai2025-aeroeyes/observing/train/samples')
    train_sampler = DistributedSampler(train_dataset) if get_world_size() > 1 else None
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.TRAIN.BATCH_SIZE,
        num_workers=cfg.TRAIN.NUM_WORKERS,
        pin_memory=True,
        sampler=train_sampler
    )
    logger.info(f"Found {len(train_dataset)} samples")
    return train_loader

def build_opt_lr(model, current_epoch=0):
    # Freeze backbone first
    for param in model.backbone.parameters(): param.requires_grad = False
    for m in model.backbone.modules():
        if isinstance(m, nn.BatchNorm2d): m.eval()

    if current_epoch >= cfg.BACKBONE.TRAIN_EPOCH:
        for layer in cfg.BACKBONE.TRAIN_LAYERS:
            for param in getattr(model.backbone, layer).parameters(): param.requires_grad = True
            for m in getattr(model.backbone, layer).modules():
                if isinstance(m, nn.BatchNorm2d): m.train()

    trainable_params = [
        {'params': filter(lambda x: x.requires_grad, model.backbone.parameters()), 'lr': cfg.BACKBONE.LAYERS_LR * cfg.TRAIN.BASE_LR}
    ]
    if cfg.ADJUST.ADJUST: trainable_params.append({'params': model.neck.parameters(), 'lr': cfg.TRAIN.BASE_LR})
    trainable_params.append({'params': model.rpn_head.parameters(), 'lr': cfg.TRAIN.BASE_LR})
    if cfg.MASK.MASK: trainable_params.append({'params': model.mask_head.parameters(), 'lr': cfg.TRAIN.BASE_LR})
    if cfg.REFINE.REFINE: trainable_params.append({'params': model.refine_head.parameters(), 'lr': cfg.TRAIN.LR.BASE_LR})

    optimizer = torch.optim.SGD(trainable_params, momentum=cfg.TRAIN.MOMENTUM, weight_decay=cfg.TRAIN.WEIGHT_DECAY)
    lr_scheduler = build_lr_scheduler(optimizer, epochs=cfg.TRAIN.EPOCH)
    lr_scheduler.step(cfg.TRAIN.START_EPOCH)
    return optimizer, lr_scheduler

def train(train_loader, model, optimizer, lr_scheduler, tb_writer):
    cur_lr = lr_scheduler.get_cur_lr()
    rank = get_rank()
    average_meter = AverageMeter()
    world_size = get_world_size()
    num_per_epoch = len(train_loader.dataset) // cfg.TRAIN.EPOCH // (cfg.TRAIN.BATCH_SIZE * world_size)
    start_epoch = cfg.TRAIN.START_EPOCH
    epoch = start_epoch

    if not os.path.exists(cfg.TRAIN.SNAPSHOT_DIR) and rank == 0:
        os.makedirs(cfg.TRAIN.SNAPSHOT_DIR)

    end = time.time()
    for idx, data in enumerate(train_loader):
        # update epoch
        new_epoch = idx // num_per_epoch + start_epoch
        if new_epoch != epoch:
            epoch = new_epoch
            if rank == 0:
                torch.save({'epoch': epoch,
                            'state_dict': model.module.state_dict(),
                            'optimizer': optimizer.state_dict()},
                            os.path.join(cfg.TRAIN.SNAPSHOT_DIR, f'checkpoint_e{epoch}.pth'))
            if epoch >= cfg.TRAIN.EPOCH: return
            if cfg.BACKBONE.TRAIN_EPOCH == epoch:
                optimizer, lr_scheduler = build_opt_lr(model.module, epoch)
            lr_scheduler.step(epoch)
            cur_lr = lr_scheduler.get_cur_lr()

        # forward
        outputs = model(data)
        loss = outputs['total_loss']
        if not (math.isnan(loss.item()) or math.isinf(loss.item()) or loss.item() > 1e4):
            optimizer.zero_grad()
            loss.backward()
            reduce_gradients(model)
            clip_grad_norm_(model.parameters(), cfg.TRAIN.GRAD_CLIP)
            optimizer.step()
        batch_time = time.time() - end
        end = time.time()

def main():
    rank, world_size = dist_init()
    cfg.merge_from_file(args.cfg)

    if rank == 0:
        if not os.path.exists(cfg.TRAIN.LOG_DIR): os.makedirs(cfg.TRAIN.LOG_DIR)
        init_log('global', logging.INFO)
        add_file_handler('global', os.path.join(cfg.TRAIN.LOG_DIR, 'logs.txt'), logging.INFO)
        logger.info("Version Information: \n{}\n".format(commit()))
        logger.info("config \n{}".format(json.dumps(cfg, indent=4)))

    seed_torch(args.seed)
    model = ModelBuilder().cuda().train()

    # load pretrained backbone if available
    backbone_path = "/kaggle/input/mobilenetv2/model.pth"  # chỉnh theo vị trí thực tế
    if os.path.exists(backbone_path) and os.path.getsize(backbone_path) > 0:
        load_pretrain(model.backbone, backbone_path)

    tb_writer = SummaryWriter(cfg.TRAIN.LOG_DIR) if rank == 0 else None
    train_loader = build_data_loader()
    optimizer, lr_scheduler = build_opt_lr(model, cfg.TRAIN.START_EPOCH)

    dist_model = DistModule(model)
    train(train_loader, dist_model, optimizer, lr_scheduler, tb_writer)

if __name__ == '__main__':
    main()
