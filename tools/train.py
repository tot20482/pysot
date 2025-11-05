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
from torch.utils.data.distributed import DistributedSampler

from pysot.utils.lr_scheduler import build_lr_scheduler
from pysot.utils.log_helper import init_log, print_speed, add_file_handler
from pysot.utils.distributed import dist_init, DistModule, reduce_gradients, average_reduce, get_rank, get_world_size
from pysot.utils.model_load import load_pretrain, restore_from
from pysot.utils.average_meter import AverageMeter
from pysot.utils.misc import commit
from pysot.models.model_builder import ModelBuilder
from pysot.datasets.dataset import TrkDataset
from pysot.core.config import cfg

logger = logging.getLogger('global')

parser = argparse.ArgumentParser(description='siamrpn tracking')
parser.add_argument('--cfg', type=str, default='config.yaml', help='configuration file')
parser.add_argument('--seed', type=int, default=123456, help='random seed')
parser.add_argument('--local_rank', type=int, default=0, help='local rank for DDP')
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
    train_sampler = DistributedSampler(train_dataset) if get_world_size() > 1 else None
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.TRAIN.BATCH_SIZE,
        num_workers=cfg.TRAIN.NUM_WORKERS,
        pin_memory=True,
        sampler=train_sampler
    )
    return train_loader


def build_optimizer_scheduler(model, current_epoch=0):
    # Freeze backbone first
    for param in model.backbone.parameters():
        param.requires_grad = False
    for m in model.backbone.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.eval()

    # Unfreeze selected layers if needed
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


def train(train_loader, model, optimizer, scheduler, tb_writer):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    rank = get_rank()
    world_size = get_world_size()
    average_meter = AverageMeter()
    start_epoch = cfg.TRAIN.START_EPOCH
    max_epoch = cfg.TRAIN.EPOCH
    epoch = start_epoch

    num_per_epoch = len(train_loader.dataset) // (cfg.TRAIN.BATCH_SIZE * world_size)
    end = time.time()

    if rank == 0 and not os.path.exists(cfg.TRAIN.SNAPSHOT_DIR):
        os.makedirs(cfg.TRAIN.SNAPSHOT_DIR)

    for idx, data in tqdm(enumerate(train_loader), total=len(train_loader), desc="Training", ncols=100):
        data = {k: v.to(device, non_blocking=True) for k, v in data.items()}

        # determine current epoch
        new_epoch = idx // num_per_epoch + start_epoch
        if new_epoch != epoch:
            epoch = new_epoch
            if rank == 0:
                ckpt_path = os.path.join(cfg.TRAIN.SNAPSHOT_DIR, f'checkpoint_e{epoch}.pth')
                model_to_save = model.module if isinstance(model, DistModule) else model
                torch.save({
                    'epoch': epoch,
                    'state_dict': model_to_save.state_dict(),
                    'optimizer': optimizer.state_dict()
                }, ckpt_path)
                logger.info(f"Saved checkpoint: {ckpt_path}")

            if epoch >= max_epoch:
                logger.info("Training complete.")
                break

            if epoch == cfg.BACKBONE.TRAIN_EPOCH:
                optimizer, scheduler = build_optimizer_scheduler(model.module if isinstance(model, DistModule) else model, epoch)

            if hasattr(scheduler, "cur_epoch") and scheduler.cur_epoch + 1 < len(scheduler.lr_spaces):
                scheduler.step()
            cur_lr = scheduler.get_cur_lr()
            logger.info(f"Epoch {epoch + 1}, LR: {cur_lr}")

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

        batch_info = {'batch_time': average_reduce(batch_time)}
        for k, v in sorted(outputs.items()):
            batch_info[k] = average_reduce(v.item())
        average_meter.update(**batch_info)

        if rank == 0:
            if (idx + 1) % cfg.TRAIN.PRINT_FREQ == 0:
                logger.info(f"Epoch [{epoch+1}][{(idx+1)%num_per_epoch}/{num_per_epoch}] "
                            f"LR: {cur_lr:.6f}, loss: {loss.item():.4f}, batch_time: {batch_time:.2f}s")
                print_speed(idx+1 + start_epoch*num_per_epoch, average_meter.batch_time.avg, max_epoch*num_per_epoch)
            if (idx + 1) % 10 == 0 and tb_writer:
                tb_writer.add_scalar('loss/total', loss.item(), idx)

        if (idx + 1) % 20 == 0:
            tqdm.write(f"[Epoch {epoch+1}] Step {idx+1}: loss={loss.item():.4f}, LR={cur_lr:.6f}, time={batch_time:.2f}s")


def main():
    rank, world_size = dist_init()
    cfg.merge_from_file(args.cfg)
    if rank == 0:
        if not os.path.exists(cfg.TRAIN.LOG_DIR):
            os.makedirs(cfg.TRAIN.LOG_DIR)
        init_log('global', logging.INFO)
        if cfg.TRAIN.LOG_DIR:
            add_file_handler('global', os.path.join(cfg.TRAIN.LOG_DIR, 'logs.txt'), logging.INFO)
        logger.info(f"Version: {commit()}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ModelBuilder().to(device).train()

    # load pretrained backbone
    if cfg.BACKBONE.PRETRAINED:
        backbone_path = "/kaggle/input/mobilenetv2/model.pth"
        if os.path.exists(backbone_path):
            load_pretrain(model.backbone, backbone_path)
        else:
            raise FileNotFoundError(f"Pretrained backbone not found: {backbone_path}")

    tb_writer = SummaryWriter(cfg.TRAIN.LOG_DIR) if rank == 0 and cfg.TRAIN.LOG_DIR else None
    train_loader = build_data_loader()
    optimizer, scheduler = build_optimizer_scheduler(model, cfg.TRAIN.START_EPOCH)

    # resume
    if cfg.TRAIN.RESUME:
        model, optimizer, cfg.TRAIN.START_EPOCH = restore_from(model, optimizer, cfg.TRAIN.RESUME)
    elif cfg.TRAIN.PRETRAINED:
        load_pretrain(model, cfg.TRAIN.PRETRAINED)

    dist_model = DistModule(model)
    logger.info("Model ready ✅")

    train(train_loader, dist_model, optimizer, scheduler, tb_writer)


if __name__ == '__main__':
    seed_torch(args.seed)
    main()
