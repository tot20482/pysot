import os
import json

# Paths
samples_dir = "training_dataset/observing/train/samples/"
annotations_file = "training_dataset/observing/train/annotations/annotations.json"

print(f"📌 samples_dir = {samples_dir}")
print(f"📌 annotations_file = {annotations_file}")

# Load existing annotations
with open(annotations_file, "r") as f:
    annotations_data = json.load(f)

print(f"Loaded {len(annotations_data)} existing annotation(s):")
for ann in annotations_data:
    print(f"  - video_id: {ann['video_id']}")

# Lấy danh sách video_id đã có
existing_video_ids = {item["video_id"] for item in annotations_data}
print(f"Existing video_ids: {existing_video_ids}")

# Duyệt tất cả folder trong samples/
sample_folders = [f for f in os.listdir(samples_dir) if os.path.isdir(os.path.join(samples_dir, f))]
print(f"Found {len(sample_folders)} sample folder(s): {sample_folders}")

added_count = 0

for folder in sample_folders:
    if folder not in existing_video_ids:
        # Tạo record mới với annotations rỗng
        new_record = {
            "video_id": folder,
            "annotations": [{"bboxes": []}]
        }
        annotations_data.append(new_record)
        added_count += 1
        print(f"✅ Added new annotation for video: {folder}")
    else:
        print(f"Skipped existing video_id: {folder}")

# Lưu lại file annotation đã cập nhật
os.makedirs(os.path.dirname(annotations_file), exist_ok=True)
with open(annotations_file, "w") as f:
    json.dump(annotations_data, f, indent=4)

print(f"📌 Done! Added {added_count} new sample(s).")
