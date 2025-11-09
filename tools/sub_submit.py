import os
import json
import torch
import imageio.v3 as iio

from pysot.models.model_builder import ModelBuilder
from pysot.core.config import cfg
from pysot.tracker.tracker_builder import build_tracker

# ==========================
# CONFIG
# ==========================
cfg_path = "experiments/siamrpn_alex_dwxcorr_otb/config.yaml"
checkpoint_path = "model/checkpoint_final.pth"
test_root = "testing_dataset/public_test/public_test/samples"
submission_json_path = "submission.json"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==========================
# LOAD MODEL & TRACKER
# ==========================
cfg.merge_from_file(cfg_path)
model = ModelBuilder().to(device).eval()

ckpt = torch.load(checkpoint_path, map_location=device)
state_dict = ckpt.get("state_dict", ckpt)
model.load_state_dict(state_dict, strict=False)

tracker = build_tracker(model)
print("✅ Tracker ready")

# ==========================
# LOAD VIDEO FRAMES
# ==========================
def load_video_frames(video_path):
    frames = []
    if not os.path.exists(video_path):
        print(f"❌ Missing video: {video_path}")
        return []

    try:
        for frame in iio.imiter(video_path):
            frames.append(frame[:, :, ::-1])  # RGB → BGR
    except Exception as e:
        print(f"❌ Error reading video: {e}")
        return []

    return frames

# ==========================
# TRACK ONE VIDEO
# ==========================
def track_video(video_path):
    frames = load_video_frames(video_path)
    if len(frames) == 0:
        return []

    h, w = frames[0].shape[:2]
    init_bbox = [0, 0, w, h]  # dummy box
    tracker.init(frames[0], init_bbox)

    results = []
    for fid, frame in enumerate(frames):
        outputs = tracker.track(frame)
        bbox = outputs["bbox"]
        results.append({
            "frame": fid,
            "x1": int(bbox[0]),
            "y1": int(bbox[1]),
            "x2": int(bbox[0] + bbox[2]),
            "y2": int(bbox[1] + bbox[3])
        })
    return results

# ==========================
# UPDATE submission.json
# ==========================
with open(submission_json_path, "r") as f:
    submission = json.load(f)

# convert list → dict
sub_dict = {item["video_id"]: item for item in submission}

targets = ["LifeJacket_0", "LifeJacket_1"]

for vid in targets:
    video_path = os.path.join(test_root, vid, "drone_video.mp4")
    print(f"🔄 Re-tracking {vid} ...")
    dets = track_video(video_path)

    sub_dict[vid] = {
        "video_id": vid,
        "detections": [{"bboxes": dets}]
    }

# convert back to list
new_submission = list(sub_dict.values())

with open(submission_json_path, "w") as f:
    json.dump(new_submission, f, indent=4)

print("✅ Updated submission.json for LifeJacket_0 & LifeJacket_1")
