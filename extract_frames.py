"""
Bước 1: Trích frame từ video để làm dữ liệu train.
Tự động chia sẵn train/val (85%/15%) theo đúng cấu trúc thư mục YOLO cần:
  dataset/images/train/*.jpg
  dataset/images/val/*.jpg
(labels sẽ được tạo ở bước 2 - auto_label.py)
"""
import cv2
import os
import random

VIDEO_PATH = "uploads/HLD162.mp4"      # đổi lại đường dẫn video của bạn
OUTPUT_ROOT = "dataset/images"
SAMPLE_EVERY_N_FRAMES = 10           # lấy 1 frame mỗi 10 frame (như bản cũ)
VAL_RATIO = 0.15                     # 15% ảnh dành cho tập validation
SEED = 42

os.makedirs(f"{OUTPUT_ROOT}/train", exist_ok=True)
os.makedirs(f"{OUTPUT_ROOT}/val", exist_ok=True)

cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    raise RuntimeError(f"Khong mo duoc video: {VIDEO_PATH}")

frame_count = 0
saved_count = 0
saved_paths = []

while True:
    ret, frame = cap.read()
    if not ret:
        break

    if frame_count % SAMPLE_EVERY_N_FRAMES == 0:
        # Lưu tạm vào train, sẽ phân bổ lại train/val ở bước sau (để
        # random đều, không bị thiên lệch theo thứ tự video)
        filename = f"{OUTPUT_ROOT}/train/img_{saved_count:05d}.jpg"
        cv2.imwrite(filename, frame)
        saved_paths.append(filename)
        saved_count += 1

    frame_count += 1

cap.release()
print(f"Đã trích {saved_count} ảnh.")

# ---- Chia train/val ngẫu nhiên ----
random.seed(SEED)
random.shuffle(saved_paths)
val_count = int(len(saved_paths) * VAL_RATIO)
val_paths = saved_paths[:val_count]

for p in val_paths:
    new_path = p.replace("/train/", "/val/")
    os.rename(p, new_path)

print(f"DONE! Train: {saved_count - val_count} ảnh | Val: {val_count} ảnh")
print(f"Ảnh nằm ở: {OUTPUT_ROOT}/train/ và {OUTPUT_ROOT}/val/")
