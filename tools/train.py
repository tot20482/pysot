# Copyright (c) SenseTime. All Rights Reserved.

from __future__ import absolute_import, division, print_function, unicode_literals

import argparse
import logging
import os
import time
import math
import json
import random
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tensorboardX import SummaryWriter
from torch.nn.utils import clip_grad_norm_
from torch.utils.data.distributed import DistributedSampler

from pysot.utils.lr_scheduler import build_lr_scheduler
from pysot.utils.log_helper import init_log, print_speed, add_file_handler
from pysot.utils.distributed import dist_init, DistModule, reduce_gradients, \
    average_reduce, get_rank, get_world_size
from pysot.utils.model_load import load_pretrain, restore_from
from pysot.utils.average_meter import AverageMeter
from pysot.utils.misc import describe, commit
from pysot.models.model_builder import ModelBuilder
from pysot.core.config import cfg


logger = logging.getLogger('global')
parser = argparse.ArgumentParser(description='siamrpn tracking')
parser.add_argument('--cfg', type=str, default='config.yaml', help='configuration of tracking')
parser.add_argument('--seed', type=int, default=123456, help='random seed')
parser.add_argument('--local_rank', type=int, default=0, help='compulsory for pytorch launcer')
args = parser.parse_args()


class ProcessedDataset(Dataset):
    def __init__(self, samples_root, ann_path):
        self.samples_root = samples_root
        with open(ann_path, 'r') as f:
            ann_json = json.load(f)
        self.samples = []

        # Flatten annotations
        for video in ann_json:
            vid = video['video_id']
            for track in video['annotations']:
                for bbox_info in track['bboxes']:
                    frame_id = bbox_info['frame']
                    sample_file = os.path.join(samples_root, f"{vid}_f{frame_id}.npz")
                    if os.path.exists(sample_file):
                        self.samples.append({
                            'sample_file': sample_file,
                            'bbox': bbox_info
                        })

        logger.info(f"[ProcessedDataset] Loaded {len(self.samples)} samples")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        entry = self.samples[idx]
        data = np.load(entry['sample_file'])
        template = torch.tensor(data['template'], dtype=torch.float32)
        search = torch.tensor(data['search'], dtype=torch.float32)
        bbox = torch.tensor([entry['bbox'][k] for k in ["x1", "y1", "x2", "y2"]], dtype=torch.float32)
        return {
            'template': template,
            'search': search,
            'bbox': bbox
        }


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

    train_dataset = ProcessedDataset(
        samples_root="/kaggle/working/processed_dataset/samples",
        ann_path="/kaggle/input/processed_dataset/annotatin-new/annotations_new.json"
    )

    logger.info(f"Number of samples in dataset: {len(train_dataset)}")
    logger.info("build dataset done")

    train_sampler = None
    if get_world_size() > 1:
        train_sampler = DistributedSampler(train_dataset)
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.TRAIN.BATCH_SIZE,
        num_workers=cfg.TRAIN.NUM_WORKERS,
        pin_memory=True,
        sampler=train_sampler,
        drop_last=True,
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


def train(train_loader, model, optimizer, lr_scheduler, tb_writer):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    cur_lr = lr_scheduler.get_cur_lr()
    rank = get_rank()
    average_meter = AverageMeter()

    def is_valid_number(x):
        return not (math.isnan(x) or math.isinf(x) or x > 1e4)

    world_size = get_world_size()
    num_per_epoch = len(train_loader.dataset) // cfg.TRAIN.EPOCH // (cfg.TRAIN.BATCH_SIZE * world_size)
    start_epoch = cfg.TRAIN.START_EPOCH
    epoch = start_epoch

    if not os.path.exists(cfg.TRAIN.SNAPSHOT_DIR) and get_rank() == 0:
        os.makedirs(cfg.TRAIN.SNAPSHOT_DIR)

    logger.info("model\n{}".format(describe(model.module)))

    end = time.time()
    for idx, data in enumerate(train_loader):
        data = {k: v.to(device, non_blocking=True) for k, v in data.items()}

        if epoch != idx // num_per_epoch + start_epoch:
            epoch = idx // num_per_epoch + start_epoch

            if get_rank() == 0:
                torch.save(
                    {
                        'epoch': epoch,
                        'state_dict': model.module.state_dict(),
                        'optimizer': optimizer.state_dict()
                    },
                    os.path.join(cfg.TRAIN.SNAPSHOT_DIR, f'checkpoint_e{epoch}.pth')
                )

            if epoch == cfg.TRAIN.EPOCH:
                return

            if cfg.BACKBONE.TRAIN_EPOCH == epoch:
                logger.info('start training backbone.')
                optimizer, lr_scheduler = build_opt_lr(model.module, epoch)
                logger.info("model\n{}".format(describe(model.module)))

            if lr_scheduler.last_epoch < len(lr_scheduler.lr_spaces) - 1:
                lr_scheduler.step(epoch)

            cur_lr = lr_scheduler.get_cur_lr()
            logger.info('epoch: {}'.format(epoch + 1))

        tb_idx = idx
        data_time = average_reduce(time.time() - end)
        if rank == 0:
            tb_writer.add_scalar('time/data', data_time, tb_idx)

        outputs = model(data)
        loss = outputs['total_loss']

        if is_valid_number(loss.item()):
            optimizer.zero_grad()
            loss.backward()
            reduce_gradients(model)

            if rank == 0 and cfg.TRAIN.LOG_GRADS:
                log_grads(model.module, tb_writer, tb_idx)

            clip_grad_norm_(model.parameters(), cfg.TRAIN.GRAD_CLIP)
            optimizer.step()

        batch_time = time.time() - end
        batch_info = {'batch_time': average_reduce(batch_time), 'data_time': average_reduce(data_time)}
        for k, v in sorted(outputs.items()):
            batch_info[k] = average_reduce(v.item())

        average_meter.update(**batch_info)

        if rank == 0:
            for k, v in batch_info.items():
                tb_writer.add_scalar(k, v, tb_idx)

            if (idx + 1) % cfg.TRAIN.PRINT_FREQ == 0:
                info = f"Epoch: [{epoch + 1}][{(idx + 1) % num_per_epoch}/{num_per_epoch}] lr: {cur_lr:.6f}\n"
                for cc, (k, v) in enumerate(batch_info.items()):
                    info += f"\t{k}: {v:.4f}"
                    info += "\n" if cc % 2 == 1 else "\t"
                logger.info(info)
                print_speed(idx + 1 + start_epoch * num_per_epoch, average_meter.batch_time.avg, cfg.TRAIN.EPOCH * num_per_epoch)

        end = time.time()


def main():
    rank, world_size = dist_init()
    logger.info("init done")

    cfg.merge_from_file(args.cfg)
    if rank == 0:
        if not os.path.exists(cfg.TRAIN.LOG_DIR):
            os.makedirs(cfg.TRAIN.LOG_DIR)
        init_log('global', logging.INFO)
        if cfg.TRAIN.LOG_DIR:
            add_file_handler('global', os.path.join(cfg.TRAIN.LOG_DIR, 'logs.txt'), logging.INFO)

        logger.info("Version Information: \n{}\n".format(commit()))
        logger.info("config \n{}".format(json.dumps(cfg, indent=4)))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ModelBuilder().to(device).train()

    logger.info(f"🔥 Using device: {device}")

    # load pretrained backbone
    if cfg.BACKBONE.PRETRAINED:
        backbone_path = "/kaggle/input/mobilenetv2/model.pth"
        if os.path.exists(backbone_path) and os.path.getsize(backbone_path) > 0:
            load_pretrain(model.backbone, backbone_path)
        else:
            raise FileNotFoundError(f"Pretrained model not found or empty: {backbone_path}")

    if rank == 0 and cfg.TRAIN.LOG_DIR:
        tb_writer = SummaryWriter(cfg.TRAIN.LOG_DIR)
    else:
        tb_writer = None

    train_loader = build_data_loader()

    optimizer, lr_scheduler = build_opt_lr(model, cfg.TRAIN.START_EPOCH)

    if cfg.TRAIN.RESUME:
        logger.info("resume from {}".format(cfg.TRAIN.RESUME))
        assert os.path.isfile(cfg.TRAIN.RESUME), \
            '{} is not a valid file.'.format(cfg.TRAIN.RESUME)
        model, optimizer, cfg.TRAIN.START_EPOCH = \
            restore_from(model, optimizer, cfg.TRAIN.RESUME)
    elif cfg.TRAIN.PRETRAINED:
        load_pretrain(model, cfg.TRAIN.PRETRAINED)

    dist_model = DistModule(model)

    logger.info(lr_scheduler)
    logger.info("model prepare done")

    train(train_loader, dist_model, optimizer, lr_scheduler, tb_writer)


if __name__ == '__main__':
    seed_torch(args.seed)
    main()
