import os
import json
from glob import glob
import logging
import cv2
import numpy as np
from torch.utils.data import Dataset
from pysot.datasets.augmentation import Augmentation
from pysot.datasets.anchor_target import AnchorTarget
from pysot.core.config import cfg
from pysot.utils.bbox import center2corner, Center

# Setup logging
logging.basicConfig(level=logging.INFO, filename="dataset_errors.log", filemode="w")
logger = logging.getLogger("TrkDataset")
console = logging.StreamHandler()
console.setLevel(logging.INFO)
logger.addHandler(console)

class TrkDataset(Dataset):
    def __init__(self, samples_root=None, ann_path=None, num_templates=3, frame_step=1):
        super().__init__()
        self.samples_root = os.path.abspath(samples_root or "training_dataset/observing/train/samples")
        self.num_templates = num_templates
        self.frame_step = frame_step

        # Load annotations
        ann_path = ann_path or os.path.join(os.path.dirname(self.samples_root), 'annotations', 'annotations.json')
        if os.path.exists(ann_path):
            with open(ann_path, 'r') as f:
                self.annotations = json.load(f)
            logger.info(f"Loaded annotation file: {ann_path}")
        else:
            logger.warning(f"Annotation file not found: {ann_path}, using empty annotation dict")
            self.annotations = {}

        # Sample folders
        self.sample_dirs = sorted([d for d in glob(os.path.join(self.samples_root, '*')) if os.path.isdir(d)])
        if not self.sample_dirs:
            raise RuntimeError(f"No sample directories found in {self.samples_root}")
        logger.info(f"Found {len(self.sample_dirs)} sample directories.")

        # Augmentations
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
        logger.info("Augmentation setup complete.")

        self.anchor_target = AnchorTarget()
        self.to_tensor = lambda x: x.transpose((2,0,1)).astype(np.float32)
        self.num = len(self.sample_dirs) * max(1, getattr(cfg.DATASET, 'VIDEOS_PER_EPOCH', 1))
        logger.info(f"Dataset initialized with {self.num} total samples.")

    def __len__(self):
        return self.num

    def _load_templates(self, sample_dir):
        img_dir = os.path.join(sample_dir, 'object_images')
        img_paths = sorted(glob(os.path.join(img_dir, '*.*')))[:self.num_templates]
        imgs = []
        for p in img_paths:
            img = cv2.imread(p)
            if img is None:
                logger.warning(f"Cannot read template image: {p}")
                continue
            imgs.append(img)
        if not imgs:
            imgs = [np.zeros((cfg.TRAIN.EXEMPLAR_SIZE, cfg.TRAIN.EXEMPLAR_SIZE, 3), dtype=np.uint8)] * self.num_templates
        while len(imgs) < self.num_templates:
            imgs.append(imgs[-1].copy())
        return imgs

    def _sample_frame_from_video(self, video_path, ann_bboxes=None):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logger.warning(f"Cannot open video: {video_path}")
            return None, None, None
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
        if ann_bboxes:
            frame_keys = [int(k) for k in ann_bboxes.keys()]
            fidx = np.random.choice(frame_keys) if frame_keys else np.random.randint(0, frame_count)
            cap.set(cv2.CAP_PROP_POS_FRAMES, fidx-1)
            ret, frame = cap.read()
            cap.release()
            if not ret:
                logger.warning(f"Failed to read frame {fidx} from {video_path}")
                return None, None, None
            bbox = ann_bboxes.get(str(fidx))
            return fidx, frame, bbox
        # Random frame
        fidx = np.random.randint(0, frame_count)
        cap.set(cv2.CAP_PROP_POS_FRAMES, fidx)
        ret, frame = cap.read()
        cap.release()
        if not ret:
            logger.warning(f"Failed to read random frame {fidx} from {video_path}")
            return None, None, None
        return fidx, frame, None

    def _compute_search_box_from_ann(self, image, ann_bbox):
        try:
            b = ann_bbox
            if isinstance(b, dict):
                b = list(b.values())
            b = list(map(float, b))
            if len(b) == 4:
                x1, y1, x2, y2 = b
                w, h = x2-x1, y2-y1
                cx, cy = x1+w/2.0, y1+h/2.0
            else:
                h_img, w_img = image.shape[:2]
                cx, cy = w_img/2, h_img/2
                w, h = w_img*0.2, h_img*0.2
            return center2corner(Center(int(cx), int(cy), int(w), int(h)))
        except Exception as e:
            logger.error(f"Error computing bbox: {e}")
            h_img, w_img = image.shape[:2]
            cx, cy = w_img/2, h_img/2
            w, h = w_img*0.2, h_img*0.2
            return center2corner(Center(int(cx), int(cy), int(w), int(h)))

    def __getitem__(self, index):
        sample_dir = self.sample_dirs[np.random.randint(0, len(self.sample_dirs))]
        sample_name = os.path.basename(sample_dir)

        # Templates
        templates_np = self._load_templates(sample_dir)
        templates_proc = []
        for timg in templates_np:
            h, w = timg.shape[:2]
            cx, cy = w//2, h//2
            bw, bh = int(w*0.5), int(h*0.5)
            center_box = center2corner(Center(cx, cy, bw, bh))
            tpl_crop, _ = self.template_aug(timg, center_box, cfg.TRAIN.EXEMPLAR_SIZE, gray=False)
            tpl_crop = cv2.cvtColor(tpl_crop.astype(np.float32), cv2.COLOR_BGR2RGB)
            templates_proc.append(self.to_tensor(tpl_crop))
        templates_t = np.stack(templates_proc, axis=0)

        # Search
        ann_for_video = self.annotations.get(sample_name, None)
        video_path = os.path.join(sample_dir, 'drone_video.mp4')
        frame_idx, search_frame, ann_bbox = self._sample_frame_from_video(video_path, ann_for_video)

        if search_frame is None:
            search_t = np.zeros((3, cfg.TRAIN.SEARCH_SIZE, cfg.TRAIN.SEARCH_SIZE), dtype=np.float32)
            return {
                'templates': templates_t,
                'search': search_t,
                'label_cls': np.array([0], dtype=np.int64),
                'label_loc': np.zeros((1,4), dtype=np.float32),
                'label_loc_weight': np.zeros((1,4), dtype=np.float32),
                'bbox': np.zeros(4, dtype=np.float32)
            }

        search_box = self._compute_search_box_from_ann(search_frame, ann_bbox)
        search_crop, bbox = self.search_aug(search_frame, search_box, cfg.TRAIN.SEARCH_SIZE, gray=False)
        search_t = self.to_tensor(cv2.cvtColor(search_crop.astype(np.float32), cv2.COLOR_BGR2RGB))
        cls, delta, delta_weight, overlap = self.anchor_target(bbox, cfg.TRAIN.OUTPUT_SIZE, neg=False)

        # Clip label_cls
        cls = np.clip(cls, 0, 1).astype(np.int64)

        # Check NaN/Inf
        for name, arr in zip(['search', 'label_loc', 'label_loc_weight', 'bbox'], [search_t, delta, delta_weight, bbox]):
            if np.isnan(arr).any() or np.isinf(arr).any():
                logger.warning(f"NaN/Inf in {name} for sample {sample_name}, frame {frame_idx}")
                arr = np.nan_to_num(arr)

        return {
            'templates': templates_t,
            'search': search_t,
            'label_cls': cls,
            'label_loc': delta,
            'label_loc_weight': delta_weight,
            'bbox': np.array(bbox, dtype=np.float32)
        }

def save_processed_dataset(dataset, save_dir="training_dataset/processed_dataset", max_samples=1000):
    os.makedirs(save_dir, exist_ok=True)
    logger.info(f"Start saving processed dataset to {save_dir} ...")
    total_samples = min(dataset.num, max_samples)
    for i in range(total_samples):
        data = dataset[i]
        np.savez_compressed(
            os.path.join(save_dir, f"sample_{i:06d}.npz"),
            templates=data['templates'],
            search=data['search'],
            label_cls=data['label_cls'],
            label_loc=data['label_loc'],
            label_loc_weight=data['label_loc_weight'],
            bbox=data['bbox']
        )
        if i % 100 == 0 and i > 0:
            logger.info(f"Saved {i}/{total_samples} samples...")
    logger.info("✅ Done saving processed dataset!")



def convert_annotations(input_file, output_file):
    logger.info(f"Loading annotation file: {input_file}")
    with open(input_file, "r") as f:
        data = json.load(f)

    merged = {}
    if isinstance(data, list):
        for ann in data:
            video_id = ann.get("video_id")
            if not video_id:
                continue
            frames = {}
            for block in ann.get("annotations", []):
                for bbox in block.get("bboxes", []):
                    frame = str(bbox.get("frame", -1))
                    if frame == "-1":
                        continue
                    frames[frame] = [
                        bbox.get("x1", 0),
                        bbox.get("y1", 0),
                        bbox.get("x2", 0),
                        bbox.get("y2", 0)
                    ]
            if frames:
                merged[video_id] = frames
    elif isinstance(data, dict):
        merged = data
    else:
        raise ValueError(f"Unsupported annotation format: {type(data)}")

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, "w") as f:
        json.dump(merged, f, indent=2)

    logger.info(f"✅ Conversion done! Saved to: {output_file}")
    logger.info(f"📌 Total videos processed: {len(merged)}")
    return output_file


def main():
    config_path = "experiments/siamrpn_alex_dwxcorr_otb/config.yaml"
    cfg.merge_from_file(config_path)
    cfg.freeze()

    ann_input_file = "training_dataset/observing/train/annotations/annotations.json"
    ann_output_file = "training_dataset/processed_dataset/annotations/annotations.json"
    convert_annotations(ann_input_file, ann_output_file)

    dataset = TrkDataset(
        samples_root="training_dataset/observing/train/samples",
        ann_path=ann_output_file
    )

    save_processed_dataset(dataset, save_dir="training_dataset/processed_dataset/samples", max_samples=1000)


if __name__ == "__main__":
    main()
