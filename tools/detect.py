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

    def detect_in_frames(self, frames, ref_imgs,
                          stride=64, patch_size=None,
                          sim_thresh=0.65, frame_step=1, max_frames=None,
                          verbose=False):
        """
        Detect object in a list of frames, returns (first_frame_idx, bbox) or (None, None) if not found.
        Args:
            frames: list of numpy images (BGR) - frames to search in
            ref_imgs: list of numpy images (BGR) - reference images
            stride: stride between patches
            patch_size: patch size for scanning (if None will use self.patch_size)
            sim_thresh: cosine similarity threshold for acceptance
            frame_step: process every frame_step frames (for speed)
            max_frames: limit number of frames to check (None = all)
            verbose: whether to show progress bar and info
        Returns:
            tuple (frame_idx, bbox) or (None, None) if not found
        """
        patch_size = patch_size or self.patch_size

        # Calculate mean embedding from reference images
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

        best_sim = -1.0
        best_frame = None
        best_bbox = None

        total_frames = len(frames)
        if max_frames is not None:
            total_frames = min(total_frames, max_frames)

        pbar = None
        if verbose:
            pbar = tqdm(total=total_frames, desc="Detecting")

        for idx, frame in enumerate(frames):
            frame_idx = idx + 1  # 1-based indexing
            
            if frame_step > 1 and (frame_idx % frame_step != 0):
                if pbar:
                    pbar.update(1)
                continue
            
            if max_frames is not None and frame_idx > max_frames:
                break

            h, w = frame.shape[:2]
            if h < patch_size or w < patch_size:
                if pbar:
                    pbar.update(1)
                continue

            # Generate patch coordinates
            xs = list(range(0, w - patch_size + 1, stride))
            ys = list(range(0, h - patch_size + 1, stride))
            if xs[-1] != w - patch_size:
                xs.append(w - patch_size)
            if ys[-1] != h - patch_size:
                ys.append(h - patch_size)

            # Check each patch
            for x in xs:
                for y in ys:
                    patch = frame[y:y + patch_size, x:x + patch_size]
                    try:
                        feat = self.extract_feat(patch)
                    except Exception as e:
                        if verbose:
                            print("[WARN] extract_feat failed for a patch:", e)
                        continue
                    
                    sim = F.cosine_similarity(feat.unsqueeze(0), ref_feat.unsqueeze(0)).item()
                    if sim > best_sim:
                        best_sim = sim
                        best_frame = frame_idx
                        best_bbox = [int(x), int(y), int(x + patch_size), int(y + patch_size)]
            
            if pbar:
                pbar.update(1)

            if best_sim >= sim_thresh:
                if verbose:
                    print(f"[INFO] Early stop at frame {best_frame} with sim {best_sim:.4f}")
                break

        if pbar:
            pbar.close()

        if best_frame is None:
            return None, None
        return best_frame, best_bbox

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
    
device = 'cuda' if torch.cuda.is_available() else 'cpu'

def process_video_with_annotations(video_path, annotation_data, output_dir):
    """
    Process video with given annotations and detect objects
    Args:
        video_path: Path to the video file
        annotation_data: Dictionary containing video_id and annotations with bounding boxes
        output_dir: Directory to save output images
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize detector
    detector = InitialDetector(device=device, patch_size=128, img_size=128)
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    # Get all frames with annotations
    bboxes = annotation_data['annotations'][0]['bboxes']
    processed_frames = []

    for bbox in bboxes:
        frame_idx = bbox['frame']
        
        # Set frame position
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        
        if ret:
            # Draw bbox on frame (output1)
            frame_with_bbox = frame.copy()
            x1, y1, x2, y2 = bbox['x1'], bbox['y1'], bbox['x2'], bbox['y2']
            cv2.rectangle(frame_with_bbox, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Save output1
            output1_path = os.path.join(output_dir, f"frame_{frame_idx}_1.jpg")
            cv2.imwrite(output1_path, frame_with_bbox)
            
            # Extract region for detector
            frame_region = frame[y1:y2, x1:x2]
            if frame_region.size > 0:  # Check if region is valid
                # Use detector on the region (output2)
                frame_idx_detected, bbox_detected = detector.detect_first_bbox(
                    video_path, [frame_region],
                    stride=32,  # Smaller stride for better detection
                    sim_thresh=0.5,
                    frame_step=1,
                    max_frames=100,  # Limit search range
                    verbose=False
                )
                
                if frame_idx_detected is not None and bbox_detected is not None:
                    # Draw detected bbox
                    frame_with_detection = frame.copy()
                    x1d, y1d, x2d, y2d = bbox_detected
                    cv2.rectangle(frame_with_detection, (x1d, y1d), (x2d, y2d), (0, 0, 255), 2)
                    
                    # Save output2
                    output2_path = os.path.join(output_dir, f"frame_{frame_idx}_2.jpg")
                    cv2.imwrite(output2_path, frame_with_detection)
            
            processed_frames.append(frame_idx)
    
    cap.release()
    return processed_frames

if __name__ == "__main__":
    # Example usage
    import json
    
    # Load annotations
    with open(r"D:\ZaloAI\output.json", "r") as f:
        annotation_data = json.load(f)
    
    # Set up paths
    video_path = r"D:\ZaloAI\pysot\drone_video.mp4"
    output_dir = "output_frames"
    
    print("Processing with original video-based method...")
    # Process video with annotations using original method
    processed_frames = process_video_with_annotations(
        video_path=video_path,
        annotation_data=annotation_data,
        output_dir=output_dir
    )
    print(f"Processed {len(processed_frames)} frames")
    print(f"Output images saved in: {output_dir}")

    print("\nProcessing with new frame-based method...")
    # Initialize detector
    detector = InitialDetector(device='cuda', patch_size=128, img_size=128)
    
    # Load frames from annotations
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    
    # Create output directory for new method
    output_dir_new = "output_frames_new"
    os.makedirs(output_dir_new, exist_ok=True)
    
    # Process each annotated frame with new method
    processed_frames_new = []
    for bbox in annotation_data['annotations'][0]['bboxes']:
        frame_idx = bbox['frame']
        
        # Get frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        
        if ret:
            # Draw original bbox (output1)
            frame_with_bbox = frame.copy()
            x1, y1, x2, y2 = bbox['x1'], bbox['y1'], bbox['x2'], bbox['y2']
            cv2.rectangle(frame_with_bbox, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Save output1
            output1_path = os.path.join(output_dir_new, f"frame_{frame_idx}_1.jpg")
            cv2.imwrite(output1_path, frame_with_bbox)
            
            # Extract region and detect with new method
            frame_region = frame[y1:y2, x1:x2]
            if frame_region.size > 0:
                # Use new detect_in_frames method
                _, bbox_detected = detector.detect_in_frames(
                    frames=[frame],  # Pass single frame
                    ref_imgs=[frame_region],
                    stride=32,
                    sim_thresh=0.5,
                    frame_step=1,
                    max_frames=1,
                    verbose=False
                )
                
                if bbox_detected is not None:
                    # Draw detected bbox
                    frame_with_detection = frame.copy()
                    x1d, y1d, x2d, y2d = bbox_detected
                    cv2.rectangle(frame_with_detection, (x1d, y1d), (x2d, y2d), (0, 0, 255), 2)
                    
                    # Save output2
                    output2_path = os.path.join(output_dir_new, f"frame_{frame_idx}_2.jpg")
                    cv2.imwrite(output2_path, frame_with_detection)
            
            processed_frames_new.append(frame_idx)
    
    cap.release()
    print(f"Processed {len(processed_frames_new)} frames with new method")
    print(f"New output images saved in: {output_dir_new}")
    print("\nYou can compare results between:")
    print(f"Original method: {output_dir}")
    print(f"New method: {output_dir_new}")
