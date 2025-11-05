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
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("TrkDataset")
logger = logging.getLogger("convert_annotations")

class TrkDataset(Dataset):
    def __init__(self, samples_root=None, ann_path=None, num_templates=3, frame_step=1):
        super(TrkDataset, self).__init__()

        if samples_root is None:
            samples_root = "/kaggle/input/zaloai2025-aeroeyes/observing/train/samples"
        self.samples_root = os.path.abspath(samples_root)
        self.num_templates = num_templates
        self.frame_step = frame_step

        # Load annotation
        if ann_path is None:
            ann_path = os.path.join(os.path.dirname(self.samples_root), 'annotations', 'annotations.json')

        if os.path.exists(ann_path):
            with open(ann_path, 'r') as f:
                self.annotations = json.load(f)
            logger.info(f"Loaded annotation file: {ann_path}")
        else:
            logger.warning(f"Annotation file not found: {ann_path}, using empty annotation dict")
            self.annotations = {}

        # Danh sách folder sample
        self.sample_dirs = sorted([d for d in glob(os.path.join(self.samples_root, '*')) if os.path.isdir(d)])
        if len(self.sample_dirs) == 0:
            raise RuntimeError(f"No sample directories found in {self.samples_root}")
        logger.info(f"Found {len(self.sample_dirs)} sample directories in {self.samples_root}")

        # Augmentation
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

        # Anchor target
        self.anchor_target = AnchorTarget()
        logger.info("AnchorTarget setup completed.")

        self.to_tensor = lambda x: x.transpose((2, 0, 1)).astype(np.float32)

        # Số lượng mẫu
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
        while len(imgs) < self.num_templates:
            imgs.append(imgs[-1].copy())
        logger.debug(f"Loaded {len(imgs)} template images from {sample_dir}")
        return imgs

    def _sample_frame_from_video(self, video_path, ann_bboxes=None):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logger.warning(f"Cannot open video: {video_path}")
            return None, None, None
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
        logger.debug(f"Video {video_path} has {frame_count} frames")

        if ann_bboxes:
            frame_keys = [int(k) for k in ann_bboxes.keys()]
            if len(frame_keys) == 0:
                fidx = np.random.randint(0, frame_count)
            else:
                fidx = np.random.choice(frame_keys)
            cap.set(cv2.CAP_PROP_POS_FRAMES, fidx - 1)
            ret, frame = cap.read()
            cap.release()
            if not ret:
                logger.warning(f"Failed to read frame {fidx} from {video_path}")
                return None, None, None
            bbox = ann_bboxes.get(str(fidx)) or ann_bboxes.get(f"{fidx}")
            return fidx, frame, bbox

        # Random frame if no annotations
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
                w = x2 - x1
                h = y2 - y1
                cx = x1 + w / 2.0
                cy = y1 + h / 2.0
                logger.debug(f"Computed bbox from annotation: {b}")
            else:
                h_img, w_img = image.shape[:2]
                cx, cy = w_img / 2.0, h_img / 2.0
                w, h = w_img * 0.2, h_img * 0.2
            return center2corner(Center(int(cx), int(cy), int(w), int(h)))
        except Exception as e:
            logger.error(f"Error computing bbox from annotation: {e}")
            h_img, w_img = image.shape[:2]
            cx, cy = w_img / 2.0, h_img / 2.0
            w, h = w_img * 0.2, h_img * 0.2
            return center2corner(Center(int(cx), int(cy), int(w), int(h)))

    def __getitem__(self, index):
        logger.debug(f"Fetching item at index {index}")
        sample_dir = self.sample_dirs[np.random.randint(0, len(self.sample_dirs))]
        sample_name = os.path.basename(sample_dir)
        logger.debug(f"Selected sample directory: {sample_name}")

        # Template
        templates_np = self._load_templates(sample_dir)
        templates_proc = []
        for timg in templates_np:
            im = timg
            h, w = im.shape[:2]
            cx, cy = w // 2, h // 2
            bw, bh = int(w * 0.5), int(h * 0.5)
            center_box = center2corner(Center(cx, cy, bw, bh))
            tpl_crop, _ = self.template_aug(im, center_box, cfg.TRAIN.EXEMPLAR_SIZE, gray=False)

            tpl_crop = tpl_crop.astype(np.float32)
            tpl_crop = cv2.cvtColor(tpl_crop, cv2.COLOR_BGR2RGB)
            tpl_crop = self.to_tensor(tpl_crop)

            templates_proc.append(tpl_crop)
        templates_t = np.stack(templates_proc, axis=0)

        # Search
        ann_for_video = self.annotations.get(sample_name, None)
        video_path = os.path.join(sample_dir, 'drone_video.mp4')
        frame_idx, search_frame, ann_bbox = self._sample_frame_from_video(video_path, ann_for_video)

        if search_frame is None:
            logger.warning(f"No valid search frame for sample: {sample_name}")
            return {
                'templates': templates_t,
                'search': np.zeros((3, cfg.TRAIN.SEARCH_SIZE, cfg.TRAIN.SEARCH_SIZE), dtype=np.float32),
                'label_cls': np.zeros((1,)),
                'label_loc': np.zeros((1, 4)),
                'label_loc_weight': np.zeros((1, 4)),
                'bbox': np.array([0, 0, 0, 0])
            }

        search_box = self._compute_search_box_from_ann(search_frame, ann_bbox)
        search_crop, bbox = self.search_aug(search_frame, search_box, cfg.TRAIN.SEARCH_SIZE, gray=False)
        search_t = self.to_tensor(cv2.cvtColor(search_crop.astype(np.float32), cv2.COLOR_BGR2RGB))

        cls, delta, delta_weight, overlap = self.anchor_target(bbox, cfg.TRAIN.OUTPUT_SIZE, neg=False)
        logger.debug(f"Generated anchor targets for sample: {sample_name}")

        return {
            'templates': templates_t,
            'search': search_t,
            'label_cls': cls,
            'label_loc': delta,
            'label_loc_weight': delta_weight,
            'bbox': np.array(bbox)
        }


def save_processed_dataset(dataset, save_dir="/kaggle/working/processed_dataset", max_samples=1000):
    """
    Preprocess and save dataset samples into numpy files.
    Args:
        dataset: Instance của TrkDataset
        save_dir: Folder để lưu file .npz
        max_samples: Số lượng sample muốn lưu (tránh tốn dung lượng)
    """
    os.makedirs(save_dir, exist_ok=True)
    logger.info(f"Start saving processed dataset to {save_dir} ...")

    for i in range(min(len(dataset), max_samples)):
        data = dataset[i]
        np.savez_compressed(
            os.path.join(save_dir, f"sample_{i:06d}.npz"),
            templates=data['templates'],
            search=data['search'],
            label_cls=data['label_cls'],
            label_loc=data.get('label_loc'),
            label_loc_weight=data.get('label_loc_weight'),
            bbox=data.get('bbox')
        )
        if i % 100 == 0 and i > 0:
            logger.info(f"Saved {i}/{max_samples} samples...")

    logger.info("✅ Done saving processed dataset!")

def convert_annotations(input_file, output_file):
    """
    Convert ZaloAI annotation format into PySOT required format.
    input_file: một file JSON lớn chứa list video annotation
    output_file: file json sau khi convert
    """
    with open(input_file, "r") as f:
        data = json.load(f)

    merged = {}
    logger.info(f"🔍 Found {len(data)} annotation records in {input_file}")

    for ann in data:
        video_id = ann.get("video_id")
        if not video_id:
            logger.warning(f"⚠️ Missing 'video_id' in entry, skipping")
            continue

        frames = {}
        ann_list = ann.get("annotations", [])
        if not ann_list:
            logger.warning(f"⚠️ No 'annotations' found for video {video_id}, skipping")
            continue

        # Duyệt qua từng khối annotations (thường chỉ có 1)
        for block in ann_list:
            for bbox in block.get("bboxes", []):
                frame = str(bbox.get("frame", -1))
                if frame == "-1":
                    logger.warning(f"⚠️ Invalid frame in video {video_id}, skipping bbox")
                    continue
                frames[frame] = [
                    bbox.get("x1", 0),
                    bbox.get("y1", 0),
                    bbox.get("x2", 0),
                    bbox.get("y2", 0)
                ]

        if len(frames) == 0:
            logger.warning(f"⚠️ No bbox frames found for video {video_id}, skipping")
            continue

        merged[video_id] = frames
        logger.info(f"✅ Processed {video_id}: {len(frames)} frames")

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, "w") as f:
        json.dump(merged, f, indent=2)

    logger.info(f"🎯 Done! Converted annotation saved to: {output_file}")
    logger.info(f"📌 Total valid videos processed: {len(merged)}")
    return output_file



def main():
    # Load config file
    config_path = "/kaggle/working/pysot/experiments/siamrpn_alex_dwxcorr_otb/config.yaml"
    cfg.merge_from_file(config_path)
    cfg.freeze()

    # === ✅ STEP 1: Convert annotation format ===
    ann_input_file = "/kaggle/input/annotation/output.json"
    ann_output_file = "/kaggle/working/processed_dataset/annotations/annotations.json"
    convert_annotations(ann_input_file, ann_output_file)

    # === ✅ STEP 2: Init dataset ===
    dataset = TrkDataset(
        samples_root="/kaggle/input/zaloai2025-aeroeyes/observing/train/samples",
        ann_path=ann_output_file
    )
    
    # === ✅ STEP 3: Save processed samples ===
    save_processed_dataset(dataset, save_dir="/kaggle/working/processed_dataset/samples", max_samples=1000)



if __name__ == "__main__":
    main()
