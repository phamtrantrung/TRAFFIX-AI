"""
Bước 1: Trích frame từ video để làm dữ liệu train.
Mỗi video lấy đúng FRAMES_PER_VIDEO ảnh, dàn đều trong video.
Số thứ tự ảnh tự tiếp nối qua các lần chạy (không bị ghi đè khi chạy nhiều video).
Tự động chia sẵn train/val (85%/15%) theo đúng cấu trúc thư mục YOLO cần:
  dataset/images/train/*.jpg
  dataset/images/val/*.jpg
"""
import cv2
import os
import random
import glob

VIDEO_PATH = r"D:\DAI_HOC_GTVT\Thuctap\Video test\Video 4.mp4"   # đổi cho từng video
OUTPUT_ROOT = "dataset/images"
FRAMES_PER_VIDEO = 300
VAL_RATIO = 0.15
SEED = 42

os.makedirs(f"{OUTPUT_ROOT}/train", exist_ok=True)
os.makedirs(f"{OUTPUT_ROOT}/val", exist_ok=True)

def get_next_index():
    """Tìm số thứ tự tiếp theo dựa trên ảnh đã có ở cả train và val."""
    existing = glob.glob(f"{OUTPUT_ROOT}/train/img_*.jpg") + glob.glob(f"{OUTPUT_ROOT}/val/img_*.jpg")
    if not existing:
        return 0
    indices = [int(os.path.basename(p).replace("img_", "").replace(".jpg", "")) for p in existing]
    return max(indices) + 1

cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    raise RuntimeError(f"Khong mo duoc video: {VIDEO_PATH}")

total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
if total_frames < FRAMES_PER_VIDEO:
    raise RuntimeError(f"Video chỉ có {total_frames} frame, không đủ {FRAMES_PER_VIDEO}")

# Chọn đều FRAMES_PER_VIDEO vị trí frame trải khắp video
frame_indices = sorted(set(
    int(i * total_frames / FRAMES_PER_VIDEO) for i in range(FRAMES_PER_VIDEO)
))

start_index = get_next_index()
current_idx = start_index
saved_paths = []

for target_frame in frame_indices:
    cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
    ret, frame = cap.read()
    if not ret:
        continue
    filename = f"{OUTPUT_ROOT}/train/img_{current_idx:05d}.jpg"
    cv2.imwrite(filename, frame)
    saved_paths.append(filename)
    current_idx += 1

cap.release()
print(f"Đã trích {len(saved_paths)} ảnh (đánh số từ img_{start_index:05d} đến img_{current_idx-1:05d}).")

# ---- Chia train/val ngẫu nhiên (chỉ trên số ảnh vừa trích) ----
random.seed(SEED)
random.shuffle(saved_paths)
val_count = int(len(saved_paths) * VAL_RATIO)
val_paths = saved_paths[:val_count]

for p in val_paths:
    new_path = p.replace("/train/", "/val/")
    os.rename(p, new_path)

print(f"DONE! Train: {len(saved_paths) - val_count} ảnh | Val: {val_count} ảnh")
print(f"Ảnh nằm ở: {OUTPUT_ROOT}/train/ và {OUTPUT_ROOT}/val/")