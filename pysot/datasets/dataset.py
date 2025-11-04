# Copyright (c) SenseTime. All Rights Reserved.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import json
import logging
import sys
import os
from glob import glob

import cv2
import numpy as np
from torch.utils.data import Dataset

from pysot.utils.bbox import center2corner, Center
from pysot.datasets.anchor_target import AnchorTarget
from pysot.datasets.augmentation import Augmentation
from pysot.core.config import cfg

logger = logging.getLogger("global")

# setting opencv
pyv = sys.version[0]
if pyv[0] == '3':
    cv2.ocl.setUseOpenCL(False)


class SubDataset(object):
    def __init__(self, name, root, anno, frame_range, num_use, start_idx):
        cur_path = os.path.dirname(os.path.realpath(__file__))
        self.name = name
        self.root = os.path.join(cur_path, '../../', root)
        self.anno = os.path.join(cur_path, '../../', anno)
        self.frame_range = frame_range
        self.num_use = num_use
        self.start_idx = start_idx
        logger.info("loading " + name)
        with open(self.anno, 'r') as f:
            meta_data = json.load(f)
            meta_data = self._filter_zero(meta_data)

        for video in list(meta_data.keys()):
            for track in meta_data[video]:
                frames = meta_data[video][track]
                frames = list(map(int,
                              filter(lambda x: x.isdigit(), frames.keys())))
                frames.sort()
                meta_data[video][track]['frames'] = frames
                if len(frames) <= 0:
                    logger.warning("{}/{} has no frames".format(video, track))
                    del meta_data[video][track]

        for video in list(meta_data.keys()):
            if len(meta_data[video]) <= 0:
                logger.warning("{} has no tracks".format(video))
                del meta_data[video]

        self.labels = meta_data
        self.num = len(self.labels)
        self.num_use = self.num if self.num_use == -1 else self.num_use
        self.videos = list(meta_data.keys())
        logger.info("{} loaded".format(self.name))
        self.path_format = '{}.{}.{}.jpg'
        self.pick = self.shuffle()

    def _filter_zero(self, meta_data):
        meta_data_new = {}
        for video, tracks in meta_data.items():
            new_tracks = {}
            for trk, frames in tracks.items():
                new_frames = {}
                for frm, bbox in frames.items():
                    if not isinstance(bbox, dict):
                        if len(bbox) == 4:
                            x1, y1, x2, y2 = bbox
                            w, h = x2 - x1, y2 - y1
                        else:
                            w, h = bbox
                        if w <= 0 or h <= 0:
                            continue
                    new_frames[frm] = bbox
                if len(new_frames) > 0:
                    new_tracks[trk] = new_frames
            if len(new_tracks) > 0:
                meta_data_new[video] = new_tracks
        return meta_data_new

    def log(self):
        logger.info("{} start-index {} select [{}/{}] path_format {}".format(
            self.name, self.start_idx, self.num_use,
            self.num, self.path_format))

    def shuffle(self):
        lists = list(range(self.start_idx, self.start_idx + self.num))
        pick = []
        while len(pick) < self.num_use:
            np.random.shuffle(lists)
            pick += lists
        return pick[:self.num_use]

    def get_image_anno(self, video, track, frame):
        frame = "{:06d}".format(frame)
        image_path = os.path.join(self.root, video,
                                  self.path_format.format(frame, track, 'x'))
        image_anno = self.labels[video][track][frame]
        return image_path, image_anno

    def get_positive_pair(self, index):
        video_name = self.videos[index]
        video = self.labels[video_name]
        track = np.random.choice(list(video.keys()))
        track_info = video[track]

        frames = track_info['frames']
        template_frame = np.random.randint(0, len(frames))
        left = max(template_frame - self.frame_range, 0)
        right = min(template_frame + self.frame_range, len(frames)-1) + 1
        search_range = frames[left:right]
        template_frame = frames[template_frame]
        search_frame = np.random.choice(search_range)
        return self.get_image_anno(video_name, track, template_frame), \
            self.get_image_anno(video_name, track, search_frame)

    def get_random_target(self, index=-1):
        if index == -1:
            index = np.random.randint(0, self.num)
        video_name = self.videos[index]
        video = self.labels[video_name]
        track = np.random.choice(list(video.keys()))
        track_info = video[track]
        frames = track_info['frames']
        frame = np.random.choice(frames)
        return self.get_image_anno(video_name, track, frame)

    def __len__(self):
        return self.num


# --- Replace / insert this TrkDataset implementation into pysot/datasets/dataset.py ---

class TrkDataset(Dataset):
    """
    TrkDataset for folder-based samples:
      samples_root/
        ├─ Backpack_0/
        |    ├─ object_images/img_1.jpg, img_2.jpg, img_3.jpg
        |    └─ drone_video.mp4
        ├─ Backpack_1/
        |    ...
    Optional annotations file (annotations.json) expected at:
      <same_parent>/annotations/annotations.json
    Format expected (same as the challenge): mapping video_id -> list of bboxes per frame.
    """

    def __init__(self, samples_root=None, num_templates=3, frame_step=1):
        super(TrkDataset, self).__init__()

        cur_path = os.path.dirname(os.path.realpath(__file__))

        # default samples root (relative to dataset.py)
        if samples_root is None:
            samples_root = os.path.join(cur_path, '../../observing/train/samples')
        self.samples_root = os.path.abspath(samples_root)
        self.num_templates = num_templates
        self.frame_step = frame_step

        # Try load annotations if exists
        ann_path = os.path.join(os.path.dirname(self.samples_root), 'annotations', 'annotations.json')
        if os.path.exists(ann_path):
            try:
                with open(ann_path, 'r') as f:
                    self.annotations = json.load(f)
            except Exception:
                self.annotations = {}
        else:
            self.annotations = {}

        # collect sample folders
        self.sample_dirs = sorted([d for d in glob(os.path.join(self.samples_root, '*')) if os.path.isdir(d)])
        if len(self.sample_dirs) == 0:
            raise RuntimeError("No sample folders found in %s" % self.samples_root)

        # augmentation (reuse existing config keys)
        self.template_aug = Augmentation(
            cfg.DATASET.TEMPLATE.SHIFT,
            cfg.DATASET.TEMPLATE.SCALE,
            cfg.DATASET.TEMPLATE.BLUR,
            cfg.DATASET.TEMPLATE.FLIP,
            cfg.DATASET.TEMPLATE.COLOR
        )
        self.search_aug = Augmentation(
            cfg.DATASET.SEARCH.SHIFT,
            cfg.DATASET.SEARCH.SCALE,
            cfg.DATASET.SEARCH.BLUR,
            cfg.DATASET.SEARCH.FLIP,
            cfg.DATASET.SEARCH.COLOR
        )

        # precompute transforms for numpy->tensor
        self.to_tensor = lambda x: x.transpose((2, 0, 1)).astype(np.float32)

        # choose length (you can scale this)
        self.num = len(self.sample_dirs) * max(1, cfg.DATASET.VIDEOS_PER_EPOCH if hasattr(cfg.DATASET, 'VIDEOS_PER_EPOCH') else 1)

        logger.info("TrkDataset: found %d samples under %s", len(self.sample_dirs), self.samples_root)

    def __len__(self):
        return self.num

    def _load_templates(self, sample_dir):
        img_dir = os.path.join(sample_dir, 'object_images')
        img_paths = sorted(glob(os.path.join(img_dir, '*.*')))[:self.num_templates]
        imgs = []
        for p in img_paths:
            img = cv2.imread(p)
            if img is None:
                raise RuntimeError("Cannot read template image: %s" % p)
            imgs.append(img)
        # if less than num_templates, replicate last
        while len(imgs) < self.num_templates:
            imgs.append(imgs[-1].copy())
        return imgs  # list of numpy BGR images

    def _sample_frame_from_video(self, video_path, ann_bboxes=None):
        """
        Return (frame_idx, frame_image, bbox) where bbox is [x1,y1,x2,y2] if available else None.
        If ann_bboxes provided (dict mapping frame -> bbox), sample one annotated frame.
        Otherwise pick a random frame (and bbox=None).
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError("Cannot open video: %s" % video_path)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0

        if ann_bboxes:
            # pick a random annotated frame from ann_bboxes keys (strings)
            frame_keys = [int(k) for k in ann_bboxes.keys()]
            if len(frame_keys) == 0:
                # fallback to random
                fidx = np.random.randint(0, max(1, frame_count))
                cap.set(cv2.CAP_PROP_POS_FRAMES, fidx)
                ret, frame = cap.read()
                cap.release()
                return fidx, frame, None
            fidx = np.random.choice(frame_keys)
            cap.set(cv2.CAP_PROP_POS_FRAMES, fidx - 1)  # annotations often 1-indexed; adjust if needed
            ret, frame = cap.read()
            cap.release()
            if not ret:
                return None, None, None
            bbox = ann_bboxes.get(str(fidx)) or ann_bboxes.get("{:d}".format(fidx))
            return fidx, frame, bbox
        else:
            # random frame
            fidx = np.random.randint(0, max(1, frame_count))
            cap.set(cv2.CAP_PROP_POS_FRAMES, fidx)
            ret, frame = cap.read()
            cap.release()
            if not ret:
                return None, None, None
            return fidx, frame, None

    def __getitem__(self, index):
        """
        Returns:
            {
                'templates': tensor [N, C, H, W],
                'search': tensor [C, Hs, Ws],
                'label_cls': cls_targets,
                'label_loc': loc_targets,
                'label_loc_weight': loc_weight,
                'bbox': np.array([x1,y1,x2,y2])
            }
        """
        # pick a sample folder randomly
        sample_dir = self.sample_dirs[np.random.randint(0, len(self.sample_dirs))]
        sample_name = os.path.basename(sample_dir)

        # load templates
        templates_np = self._load_templates(sample_dir)  # list of BGR numpy images

        # load annotated bboxes for this video if present
        ann_for_video = None
        # annotations in challenge may use video ids same as folder names
        if sample_name in self.annotations:
            ann_for_video = self.annotations[sample_name][0] if isinstance(self.annotations[sample_name], list) else self.annotations[sample_name]

        # sample a search frame and bbox (if exists)
        video_path = os.path.join(sample_dir, 'drone_video.mp4')
        frame_idx, search_frame, ann_bbox = self._sample_frame_from_video(video_path, ann_for_video)

        if search_frame is None:
            # fallback: return simple zeroed tensors to avoid crash
            template_t = np.stack([self.to_tensor(cv2.cvtColor(t, cv2.COLOR_BGR2RGB)) for t in templates_np], axis=0)
            search_t = np.zeros((3, cfg.TRAIN.SEARCH_SIZE, cfg.TRAIN.SEARCH_SIZE), dtype=np.float32)
            label_cls = np.zeros((1,))
            label_loc = np.zeros((1, 4))
            label_loc_weight = np.zeros((1, 4))
            return {
                'templates': template_t,
                'search': search_t,
                'label_cls': label_cls,
                'label_loc': label_loc,
                'label_loc_weight': label_loc_weight,
                'bbox': np.array([0,0,0,0])
            }

        # compute exemplar and search crops using the same helper _get_bbox & augmentation from original code
        # For template, we will use given object_images as templates (apply template_aug on template crops)
        # We'll compute template_box based on ann_bbox if available otherwise center crop

        # Determine template box relative to template image: here assume template images show object roughly centered
        templates_proc = []
        for timg in templates_np:
            # convert BGR->RGB then augmentation (using existing augmentation expects image and bbox)
            im = timg
            # Make a default bbox covering center area (if no annotation for template image)
            h, w = im.shape[:2]
            # default bbox: center 50% area
            cx, cy = w//2, h//2
            bw, bh = int(w*0.5), int(h*0.5)
            center_box = center2corner(Center(cx, cy, bw, bh))
            tpl_crop, _ = self.template_aug(im, center_box, cfg.TRAIN.EXEMPLAR_SIZE, gray=False)
            tpl_crop = self.to_tensor(cv2.cvtColor(tpl_crop, cv2.COLOR_BGR2RGB))
            templates_proc.append(tpl_crop)

        templates_t = np.stack(templates_proc, axis=0)  # [N, C, H, W]

        # For search image, use ann_bbox to produce a bbox; otherwise center bbox
        if ann_bbox is not None:
            # ann_bbox may be [x1,y1,x2,y2] or similar; convert to center form expected by _get_bbox
            # Here we pass ann_bbox directly to helper _get_bbox by creating a temporary image
            search_image = search_frame
            # compute search box in same style as original _get_bbox: we reuse that helper
            search_box = self._compute_search_box_from_ann(search_image, ann_bbox)
        else:
            # use center bbox as no ann available
            h, w = search_frame.shape[:2]
            cx, cy = w//2, h//2
            bw, bh = int(w*0.2), int(h*0.2)
            search_box = center2corner(Center(cx, cy, bw, bh))

        # Apply augmentation to search patch
        search_crop, bbox = self.search_aug(search_frame, search_box, cfg.TRAIN.SEARCH_SIZE, gray=False)
        search_t = self.to_tensor(cv2.cvtColor(search_crop, cv2.COLOR_BGR2RGB))

        # Compute labels using anchor_target (if we have bbox)
        # anchor_target expects bbox in [cx, cy, w, h] normalized to search patch.
        # We'll pass bbox returned by augmentation (already in format used by original code).
        cls, delta, delta_weight, overlap = self.anchor_target(bbox, cfg.TRAIN.OUTPUT_SIZE, neg=False)

        return {
            'templates': templates_t,        # numpy array [N, C, H, W] float32
            'search': search_t,              # numpy array [C, Hs, Ws] float32
            'label_cls': cls,
            'label_loc': delta,
            'label_loc_weight': delta_weight,
            'bbox': np.array(bbox)
        }

    # helper: convert annotation bbox to the format expected by _get_bbox logic
    def _compute_search_box_from_ann(self, image, ann_bbox):
        """
        ann_bbox may be [x1,y1,x2,y2] or [x,y,w,h] depending on annotations.
        We convert it to center2corner(Center(cx,cy,w,h)) consistent with _get_bbox output.
        """
        try:
            b = ann_bbox
            if isinstance(b, dict):
                # if stored as dict with keys, try to parse
                b = list(b.values())
            b = list(map(float, b))
            if len(b) == 4:
                x1, y1, x2, y2 = b
                w = x2 - x1
                h = y2 - y1
                cx = x1 + w/2.0
                cy = y1 + h/2.0
            elif len(b) == 2:
                w, h = b
                h_img, w_img = image.shape[:2]
                cx, cy = w_img/2.0, h_img/2.0
            else:
                # fallback center small box
                h_img, w_img = image.shape[:2]
                cx, cy = w_img/2.0, h_img/2.0
                w, h = w_img*0.2, h_img*0.2
        except Exception:
            h_img, w_img = image.shape[:2]
            cx, cy = w_img/2.0, h_img/2.0
            w, h = w_img*0.2, h_img*0.2

        return center2corner(Center(int(cx), int(cy), int(w), int(h)))
