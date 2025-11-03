from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os
import argparse
import cv2
import torch
import numpy as np
from glob import glob
from tqdm import tqdm
import matplotlib.pyplot as plt

from pysot.core.config import cfg
from pysot.models.model_builder import ModelBuilder
from pysot.tracker.tracker_builder import build_tracker

torch.set_num_threads(1)

parser = argparse.ArgumentParser(description='tracking demo (no GUI)')
parser.add_argument('--config', type=str, help='config file')
parser.add_argument('--snapshot', type=str, help='model name')
parser.add_argument('--video_name', default='', type=str,
                    help='video file or image directory')
parser.add_argument('--output', default='output.mp4', type=str,
                    help='output video file name')
args = parser.parse_args()


def get_frames(video_name):
    if video_name.endswith('avi') or video_name.endswith('mp4'):
        cap = cv2.VideoCapture(video_name)
        while True:
            ret, frame = cap.read()
            if ret:
                yield frame
            else:
                break
    else:
        images = glob(os.path.join(video_name, '*.jp*'))
        images = sorted(images,
                        key=lambda x: int(os.path.basename(x).split('.')[0]))
        for img in images:
            frame = cv2.imread(img)
            yield frame


def main():
    # load config
    cfg.merge_from_file(args.config)
    cfg.CUDA = torch.cuda.is_available() and cfg.CUDA
    device = torch.device('cuda' if cfg.CUDA else 'cpu')

    # create model
    model = ModelBuilder()

    # load model
    model.load_state_dict(torch.load(args.snapshot,
                                     map_location=lambda storage, loc: storage.cpu()))
    model.eval().to(device)

    # build tracker
    tracker = build_tracker(model)

    frames = list(get_frames(args.video_name))
    if len(frames) == 0:
        print("❌ No frames found. Check your --video_name path.")
        return

    # Instead of GUI selectROI, define a fixed ROI (x, y, w, h)
    # You can also update this manually based on your video
    h, w = frames[0].shape[:2]
    init_rect = (w // 4, h // 4, w // 2, h // 2)  # (x, y, width, height)

    tracker.init(frames[0], init_rect)

    # Prepare output video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(args.output, fourcc, 30, (w, h))

    # Tracking loop
    for frame in tqdm(frames, desc="Tracking"):
        outputs = tracker.track(frame)
        if 'polygon' in outputs:
            polygon = np.array(outputs['polygon']).astype(np.int32)
            cv2.polylines(frame, [polygon.reshape((-1, 1, 2))],
                          True, (0, 255, 0), 3)
        else:
            bbox = list(map(int, outputs['bbox']))
            cv2.rectangle(frame, (bbox[0], bbox[1]),
                          (bbox[0] + bbox[2], bbox[1] + bbox[3]),
                          (0, 255, 0), 3)
        out.write(frame)

    out.release()
    print(f"✅ Video saved as {args.output}")

    # Display final video in notebook
    try:
        from IPython.display import Video
        display(Video(args.output, embed=True))
    except ImportError:
        print("Video preview not available. You can download output.mp4 manually.")


if __name__ == '__main__':
    main()
