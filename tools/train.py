# Copyright (c) SenseTime. All Rights Reserved.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import argparse
import logging
import os
import time
import math
import json
import random
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from torch.nn.utils import clip_grad_norm_
from torch.utils.data.distributed import DistributedSampler

from pysot.utils.lr_scheduler import build_lr_scheduler
from pysot.utils.log_helper import init_log, print_speed, add_file_handler
from pysot.utils.distributed import dist_init, DistModule, reduce_gradients,\
        average_reduce, get_rank, get_world_size
from pysot.utils.model_load import load_pretrain, restore_from
from pysot.utils.average_meter import AverageMeter
from pysot.utils.misc import describe, commit
from pysot.models.model_builder import ModelBuilder
from pysot.datasets.dataset import TrkDataset
from pysot.core.config import cfg


logger = logging.getLogger('global')
parser = argparse.ArgumentParser(description='siamrpn tracking')
parser.add_argument('--cfg', type=str, default='config.yaml',
                    help='configuration of tracking')
parser.add_argument('--seed', type=int, default=123456,
                    help='random seed')
parser.add_argument('--local_rank', type=int, default=0,
                    help='compulsory for pytorch launcer')
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
    logger.info("build train dataset")
    
    train_dataset = TrkDataset(
        samples_root="/kaggle/input/zaloai2025-aeroeyes/observing/train/samples",
        ann_path="/kaggle/input/annotation/output.json"
    )
    
    logger.info("build dataset done")

    train_sampler = None
    if get_world_size() > 1:
        train_sampler = DistributedSampler(train_dataset)
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.TRAIN.BATCH_SIZE,
        num_workers=cfg.TRAIN.NUM_WORKERS,
        pin_memory=True,
        sampler=train_sampler
    )
    return train_loader



def build_opt_lr(model, current_epoch=0):
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

    trainable_params = []
    trainable_params += [{'params': filter(lambda x: x.requires_grad,
                                           model.backbone.parameters()),
                          'lr': cfg.BACKBONE.LAYERS_LR * cfg.TRAIN.BASE_LR}]

    if cfg.ADJUST.ADJUST:
        trainable_params += [{'params': model.neck.parameters(),
                              'lr': cfg.TRAIN.BASE_LR}]

    trainable_params += [{'params': model.rpn_head.parameters(),
                          'lr': cfg.TRAIN.BASE_LR}]

    if cfg.MASK.MASK:
        trainable_params += [{'params': model.mask_head.parameters(),
                              'lr': cfg.TRAIN.BASE_LR}]

    if cfg.REFINE.REFINE:
        trainable_params += [{'params': model.refine_head.parameters(),
                              'lr': cfg.TRAIN.LR.BASE_LR}]

    optimizer = torch.optim.SGD(trainable_params,
                                momentum=cfg.TRAIN.MOMENTUM,
                                weight_decay=cfg.TRAIN.WEIGHT_DECAY)

    lr_scheduler = build_lr_scheduler(optimizer, epochs=cfg.TRAIN.EPOCH)
    lr_scheduler.step(cfg.TRAIN.START_EPOCH)
    return optimizer, lr_scheduler


def log_grads(model, tb_writer, tb_index):
    def weights_grads(model):
        grad = {}
        weights = {}
        for name, param in model.named_parameters():
            if param.grad is not None:
                grad[name] = param.grad
                weights[name] = param.data
        return grad, weights

    grad, weights = weights_grads(model)
    feature_norm, rpn_norm = 0, 0
    for k, g in grad.items():
        _norm = g.data.norm(2)
        weight = weights[k]
        w_norm = weight.norm(2)
        if 'feature' in k:
            feature_norm += _norm ** 2
        else:
            rpn_norm += _norm ** 2

        tb_writer.add_scalar('grad_all/'+k.replace('.', '/'),
                             _norm, tb_index)
        tb_writer.add_scalar('weight_all/'+k.replace('.', '/'),
                             w_norm, tb_index)
        tb_writer.add_scalar('w-g/'+k.replace('.', '/'),
                             w_norm/(1e-20 + _norm), tb_index)
    tot_norm = feature_norm + rpn_norm
    tot_norm = tot_norm ** 0.5
    feature_norm = feature_norm ** 0.5
    rpn_norm = rpn_norm ** 0.5

    tb_writer.add_scalar('grad/tot', tot_norm, tb_index)
    tb_writer.add_scalar('grad/feature', feature_norm, tb_index)
    tb_writer.add_scalar('grad/rpn', rpn_norm, tb_index)


from tqdm import tqdm
import torch, os, math, time

def train(train_loader, model, optimizer, lr_scheduler, tb_writer):
    # ===== Setup =====
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    rank = get_rank()
    world_size = get_world_size()
    average_meter = AverageMeter()
    start_epoch = cfg.TRAIN.START_EPOCH
    max_epoch = cfg.TRAIN.EPOCH
    epoch = start_epoch

    def is_valid_number(x):
        return not (math.isnan(x) or math.isinf(x) or x > 1e4)

    if get_rank() == 0 and not os.path.exists(cfg.TRAIN.SNAPSHOT_DIR):
        os.makedirs(cfg.TRAIN.SNAPSHOT_DIR)

    # Correct batches per epoch
    num_per_epoch = len(train_loader.dataset) // (cfg.TRAIN.BATCH_SIZE * world_size)
    end = time.time()

    # ===== Training Loop =====
    for idx, data in tqdm(enumerate(train_loader), total=len(train_loader),
                          desc=f"Training", ncols=100, mininterval=1.0):

        # --- Move data to GPU ---
        data = {k: v.to(device, non_blocking=True) for k, v in data.items()}

        # --- Determine current epoch ---
        new_epoch = idx // num_per_epoch + start_epoch
        if new_epoch != epoch:
            epoch = new_epoch

            # Save checkpoint
            if get_rank() == 0:
                ckpt_path = os.path.join(cfg.TRAIN.SNAPSHOT_DIR, f'checkpoint_e{epoch}.pth')
                torch.save({
                    'epoch': epoch,
                    'state_dict': model.module.state_dict(),
                    'optimizer': optimizer.state_dict()
                }, ckpt_path)
                logger.info(f"Saved checkpoint: {ckpt_path}")

            # Stop if finished
            if epoch >= max_epoch:
                logger.info("Training complete.")
                break

            # Rebuild optimizer / scheduler if backbone unfreezes
            if epoch == cfg.BACKBONE.TRAIN_EPOCH:
                logger.info('Start training backbone.')
                optimizer, lr_scheduler = build_opt_lr(model.module, epoch)

            # Safe scheduler step
            if hasattr(lr_scheduler, "cur_epoch") and lr_scheduler.cur_epoch + 1 < len(lr_scheduler.lr_spaces):
                lr_scheduler.step()
            cur_lr = lr_scheduler.get_cur_lr()
            logger.info(f"Epoch {epoch + 1}, LR: {cur_lr}")

        # --- Timing: data loading ---
        data_time = average_reduce(time.time() - end)
        tb_idx = idx
        if rank == 0:
            tb_writer.add_scalar('time/data', data_time, tb_idx)

        # --- Forward pass ---
        start_fwd = time.time()
        outputs = model(data)
        loss = outputs['total_loss']
        fwd_time = time.time() - start_fwd

        # --- Backprop ---
        if is_valid_number(loss.item()):
            optimizer.zero_grad()
            loss.backward()
            reduce_gradients(model)
            clip_grad_norm_(model.parameters(), cfg.TRAIN.GRAD_CLIP)
            optimizer.step()

        batch_time = time.time() - end
        end = time.time()

        # --- Update meters ---
        batch_info = {
            'batch_time': average_reduce(batch_time),
            'data_time': average_reduce(data_time),
            'fwd_time': fwd_time
        }
        for k, v in sorted(outputs.items()):
            batch_info[k] = average_reduce(v.item())
        average_meter.update(**batch_info)

        # --- Logging ---
        if rank == 0:
            if (idx + 1) % cfg.TRAIN.PRINT_FREQ == 0:
                info = (f"Epoch: [{epoch + 1}][{(idx + 1) % num_per_epoch}/{num_per_epoch}] "
                        f"lr: {cur_lr:.6f}, loss: {loss.item():.4f}, "
                        f"batch_time: {batch_time:.2f}s")
                logger.info(info)
                print_speed(idx + 1 + start_epoch * num_per_epoch,
                            average_meter.batch_time.avg,
                            max_epoch * num_per_epoch)
            # Less frequent TensorBoard logging to reduce overhead
            if (idx + 1) % 10 == 0:
                for k, v in batch_info.items():
                    tb_writer.add_scalar(k, v, tb_idx)

        # --- Update tqdm (not every iteration, to avoid slowdown) ---
        if (idx + 1) % 20 == 0:
            tqdm.write(f"[Epoch {epoch+1}] Step {idx+1}: loss={loss.item():.4f}, lr={cur_lr:.6f}, "
                       f"time={batch_time:.2f}s")

    logger.info("Training finished successfully ✅")



def main():
    rank, world_size = dist_init()
    logger.info("init done")

    # load cfg
    cfg.merge_from_file(args.cfg)
    if rank == 0:
        if not os.path.exists(cfg.TRAIN.LOG_DIR):
            os.makedirs(cfg.TRAIN.LOG_DIR)
        init_log('global', logging.INFO)
        if cfg.TRAIN.LOG_DIR:
            add_file_handler('global',
                             os.path.join(cfg.TRAIN.LOG_DIR, 'logs.txt'),
                             logging.INFO)

        logger.info("Version Information: \n{}\n".format(commit()))
        # logger.info("config \n{}".format(json.dumps(cfg, indent=4)))

    # create model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ModelBuilder().to(device).train()

    logger.info(f"🔥 Using device: {device}")


    # load pretrained backbone weights
    if cfg.BACKBONE.PRETRAINED:
        cur_path = os.path.dirname(os.path.realpath(__file__))
        if cfg.BACKBONE.PRETRAINED:
    # Nếu bạn biết chính xác đường dẫn tuyệt đối đến model
            backbone_path = "/kaggle/input/mobilenetv2/model.pth"  # chỉnh theo vị trí thực tế
        if os.path.exists(backbone_path) and os.path.getsize(backbone_path) > 0:
            load_pretrain(model.backbone, backbone_path)
        else:
            raise FileNotFoundError(f"Pretrained model not found or empty: {backbone_path}")


    # create tensorboard writer
    if rank == 0 and cfg.TRAIN.LOG_DIR:
        tb_writer = SummaryWriter(cfg.TRAIN.LOG_DIR)
    else:
        tb_writer = None

    # build dataset loader
    train_loader = build_data_loader()

    # build optimizer and lr_scheduler
    optimizer, lr_scheduler = build_opt_lr(model,
                                           cfg.TRAIN.START_EPOCH)

    # resume training
    if cfg.TRAIN.RESUME:
        logger.info("resume from {}".format(cfg.TRAIN.RESUME))
        assert os.path.isfile(cfg.TRAIN.RESUME), \
            '{} is not a valid file.'.format(cfg.TRAIN.RESUME)
        model, optimizer, cfg.TRAIN.START_EPOCH = \
            restore_from(model, optimizer, cfg.TRAIN.RESUME)
    # load pretrain
    elif cfg.TRAIN.PRETRAINED:
        load_pretrain(model, cfg.TRAIN.PRETRAINED)
    dist_model = DistModule(model)

    logger.info(lr_scheduler)
    logger.info("model prepare done")

    # start training
    train(train_loader, dist_model, optimizer, lr_scheduler, tb_writer)


if __name__ == '__main__':
    seed_torch(args.seed)
    main()