# detect.py
import cv2
import torch
import torch.nn.functional as F
from torchvision import models, transforms
import numpy as np
import os
from glob import glob
from PIL import Image
from tqdm import tqdm

class InitialDetector:
    """
    Detector nhẹ: trích embedding bằng ResNet18 (pretrained),
    so sánh cosine giữa embedding trung bình của ảnh tham chiếu
    và embedding của các patch trong từng frame để tìm frame + bbox đầu tiên.
    """

    def __init__(self, device=None, patch_size=128, img_size=128):
        # Device tự chọn nếu không truyền: dùng CUDA nếu có, else CPU
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
        # Backbone nhẹ (ResNet18) - bỏ lớp FC cuối
        model = models.resnet18(pretrained=True)
        self.model = torch.nn.Sequential(*list(model.children())[:-1])  # up to avgpool
        self.model = self.model.to(self.device)
        self.model.eval()

        # Transform (PIL Image -> Tensor)
        # Thêm Normalize theo ImageNet để embedding ổn định hơn
        self.tf = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std =[0.229, 0.224, 0.225])
        ])

        self.patch_size = patch_size
        self.img_size = img_size

    def extract_feat(self, img_np):
        """
        Input: img_np - numpy array BGR (OpenCV)
        Output: 1-D tensor (512,) đã normalized (L2)
        """
        # BGR -> RGB -> PIL
        img_rgb = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_rgb)

        # transform -> [1, C, H, W]
        img_tensor = self.tf(img_pil).unsqueeze(0).to(self.device)

        with torch.no_grad():
            feat_map = self.model(img_tensor)            # shape [1, 512, 1, 1]
            feat = feat_map.view(feat_map.size(0), -1)   # [1, 512]
            feat = F.normalize(feat, p=2, dim=1)         # L2 normalize per vector

        return feat.squeeze(0)  # [512]

    def detect_first_bbox(self, video_path, ref_imgs,
                          stride=64, patch_size=None,
                          sim_thresh=0.65, frame_step=1, max_frames=None,
                          verbose=False):
        """
        Quét video, trả về (first_frame_idx, bbox) hoặc (None, None) nếu không tìm thấy.
        - ref_imgs: list of numpy images (BGR) - reference images
        - stride: bước nhảy giữa các patch
        - patch_size: kích thước patch để quét (nếu None sẽ dùng self.patch_size)
        - sim_thresh: ngưỡng cosine similarity để chấp nhận
        - frame_step: kiểm tra mỗi frame_step frame (để nhanh hơn)
        - max_frames: giới hạn số frame kiểm tra (None = tất cả)
        """
        patch_size = patch_size or self.patch_size

        # Tính embedding trung bình từ các ảnh tham chiếu
        ref_feats = []
        for r in ref_imgs:
            try:
                f = self.extract_feat(r).to(self.device)
                ref_feats.append(f)
            except Exception as e:
                if verbose:
                    print("[WARN] extract_feat failed for a reference image:", e)
        if len(ref_feats) == 0:
            return None, None
        ref_feat = torch.stack(ref_feats, dim=0).mean(0)  # [512]
        ref_feat = F.normalize(ref_feat.unsqueeze(0), p=2, dim=1).squeeze(0)

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {video_path}")

        best_sim = -1.0
        best_frame = None
        best_bbox = None
        frame_idx = 0

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if cap.get(cv2.CAP_PROP_FRAME_COUNT) > 0 else None
        if max_frames is not None:
            total_to_check = min(total_frames or max_frames, max_frames)
        else:
            total_to_check = total_frames

        pbar = None
        if verbose and total_to_check:
            pbar = tqdm(total=total_to_check, desc="Detecting")

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_idx += 1

            if frame_step > 1 and (frame_idx % frame_step != 0):
                if pbar:
                    pbar.update(1)
                continue

            h, w = frame.shape[:2]
            # skip too small frames
            if h < patch_size or w < patch_size:
                if pbar:
                    pbar.update(1)
                continue

            # iterate patches: include the right/bottom edges by +1
            xs = list(range(0, w - patch_size + 1, stride))
            ys = list(range(0, h - patch_size + 1, stride))
            # ensure last column/row includes right/bottom edge
            if xs[-1] != w - patch_size:
                xs.append(w - patch_size)
            if ys[-1] != h - patch_size:
                ys.append(h - patch_size)

            # iterate patches; compute similarity; track best
            for x in xs:
                for y in ys:
                    patch = frame[y:y + patch_size, x:x + patch_size]
                    try:
                        feat = self.extract_feat(patch)  # [512]
                    except Exception as e:
                        if verbose:
                            print("[WARN] extract_feat failed for a patch:", e)
                        continue
                    # cosine similarity (tensors 1D)
                    sim = F.cosine_similarity(feat.unsqueeze(0), ref_feat.unsqueeze(0)).item()
                    if sim > best_sim:
                        best_sim = sim
                        best_frame = frame_idx
                        best_bbox = [int(x), int(y), int(x + patch_size), int(y + patch_size)]
            if pbar:
                pbar.update(1)

            # nếu vượt ngưỡng thì dừng sớm
            if best_sim >= sim_thresh:
                if verbose:
                    print(f"[INFO] Early stop at frame {best_frame} with sim {best_sim:.4f}")
                break

            # dừng nếu đạt max_frames
            if max_frames is not None and frame_idx >= max_frames:
                break

        cap.release()
        if pbar:
            pbar.close()

        if best_frame is None:
            return None, None
        return best_frame, best_bbox
