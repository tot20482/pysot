import os
import json
import torch
import cv2
import zipfile
from glob import glob
import imageio.v3 as iio

from pysot.models.model_builder import ModelBuilder
from pysot.core.config import cfg


# ==========================
# CONFIG
# ==========================
cfg_path = "/kaggle/working/pysot/experiments/siamrpn_alex_dwxcorr_otb/config.yaml"
checkpoint_path = "model/checkpoint_final.pth"
test_root = "/kaggle/input/zaloai2025-aeroeyes/public_test/public_test/samples"

submission_json_path = "/kaggle/working/submission.json"
submission_zip_path = "/kaggle/working/submission.zip"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ==========================
# LOAD MODEL
# ==========================
if not os.path.exists(cfg_path):
    raise FileNotFoundError(f"❌ Không tìm thấy config: {cfg_path}")

cfg.merge_from_file(cfg_path)

model = ModelBuilder().to(device).eval()

# Fix FutureWarning: explicitly set weights_only
checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
checkpoint = checkpoint["state_dict"] if "state_dict" in checkpoint else checkpoint
model.load_state_dict(checkpoint, strict=False)

print("✅ Model loaded for inference")


# ==========================
# FIX VERSION — READ VIDEO WITH IMAGEIO
# ==========================
def load_video_frames(video_path):
    """Load video frames using ImageIO (FFmpeg backend) – safe for H264."""
    frames = []
    ids = []

    if not os.path.exists(video_path):
        print(f"❌ Missing video: {video_path}")
        return [], []

    try:
        for idx, frame in enumerate(iio.imiter(video_path)):
            # imageio gives frame as RGB → convert to BGR for OpenCV consistency
            frame = frame[:, :, ::-1]
            frames.append(frame)
            ids.append(idx)

        print(f"▶️ Loaded {len(frames)} frames from {os.path.basename(video_path)}")

    except Exception as e:
        print(f"❌ Error reading video via imageio: {e}")
        return [], []

    return frames, ids


# ==========================
# Dummy Tracking
# ==========================
def run_tracking(video_path):
    frames, frame_ids = load_video_frames(video_path)

    if len(frames) == 0:
        print(f"⚠️ No frames extracted from {video_path}")
        return []

    h, w = frames[0].shape[:2]
    detections = []

    # Fake bbox (bạn thay bằng tracker thật nếu muốn)
    for fid in frame_ids:
        detections.append({
            "frame": fid,
            "x1": int(0.3 * w),
            "y1": int(0.3 * h),
            "x2": int(0.6 * w),
            "y2": int(0.6 * h),
        })

    return detections


# ==========================
# MAIN LOOP: PROCESS ALL SAMPLES
# ==========================
samples = sorted(os.listdir(test_root))
results = []

for sample in samples:
    sample_dir = os.path.join(test_root, sample)

    if not os.path.isdir(sample_dir):
        continue

    print("==================================================")
    print(f"Processing sample: {sample}")

    video_path = os.path.join(sample_dir, "drone_video.mp4")

    dets = run_tracking(video_path)

    results.append({
        "video_id": sample,
        "detections": [{"bboxes": dets}] if dets else []
    })


# ==========================
# SAVE JSON
# ==========================
with open(submission_json_path, "w") as f:
    json.dump(results, f, indent=4)

print(f"✅ Saved submission.json → {submission_json_path}")


# ==========================
# ZIP SUBMISSION
# ==========================
with zipfile.ZipFile(submission_zip_path, "w") as zipf:
    zipf.write(submission_json_path, arcname="submission.json")

print(f"✅ Saved submission.zip → {submission_zip_path}")
print("🎉 Submit generation completed!")
