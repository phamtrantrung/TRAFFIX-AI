"""
Bước 2: Tự động gán nhãn sơ bộ cho ảnh vừa trích, dùng chính model
yolov8s đang chạy trong hệ thống. Sinh ra file .txt định dạng YOLO
(class_id x_center y_center width height - tất cả normalize 0-1) cho
từng ảnh, đặt cùng tên, trong dataset/labels/train và dataset/labels/val.

QUAN TRỌNG: đây chỉ là nhãn SƠ BỘ để bạn đỡ phải vẽ box từ đầu. Bạn VẪN
CẦN dùng công cụ gán nhãn (khuyên dùng LabelImg, xem hướng dẫn bên dưới)
để:
  1. Sửa lại các nhãn bị NHẦM LOẠI XE (đúng trọng tâm vấn đề đang gặp -
     xe tải bị nhận nhầm thành xe buýt, v.v.)
  2. Thêm box cho các xe mà model BỎ SÓT hoàn toàn
  3. Xóa box sai / box của vật không phải xe

Cài LabelImg (chạy 1 lần):
    pip install labelImg --break-system-packages
Mở LabelImg để sửa nhãn:
    labelImg dataset/images/train dataset/classes.txt
(Nhớ chọn định dạng "YOLO" trong LabelImg, góc dưới bên trái cửa sổ)
"""
import os
from ultralytics import YOLO

MODEL_PATH = "models/yolov8s.pt"
IMAGES_ROOT = "dataset/images"
LABELS_ROOT = "dataset/labels"
CONF_THRESHOLD = 0.35  # bỏ qua các phát hiện có độ tin cậy quá thấp

# Map từ class COCO gốc (id model yolov8s) -> id lớp mới trong dataset
# riêng của mình (thứ tự này PHẢI khớp với "names" trong data.yaml)
COCO_TO_CUSTOM = {
    2: 0,  # car
    3: 1,  # motorbike
    5: 2,  # bus
    7: 3,  # truck
}
CUSTOM_CLASS_NAMES = ["car", "motorbike", "bus", "truck"]

# Ghi sẵn file classes.txt cho LabelImg dùng
with open("dataset/classes.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(CUSTOM_CLASS_NAMES))

model = YOLO(MODEL_PATH)

for split in ("train", "val"):
    img_dir = f"{IMAGES_ROOT}/{split}"
    label_dir = f"{LABELS_ROOT}/{split}"
    os.makedirs(label_dir, exist_ok=True)

    image_files = [f for f in os.listdir(img_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
    print(f"[{split}] Đang gán nhãn cho {len(image_files)} ảnh...")

    for idx, fname in enumerate(image_files):
        img_path = os.path.join(img_dir, fname)
        results = model.predict(img_path, conf=CONF_THRESHOLD, verbose=False)[0]

        label_path = os.path.join(label_dir, os.path.splitext(fname)[0] + ".txt")
        lines = []

        if results.boxes is not None:
            h, w = results.orig_shape
            for box in results.boxes:
                cls_id = int(box.cls[0])
                if cls_id not in COCO_TO_CUSTOM:
                    continue  # bỏ qua vật không phải xe (người, đèn...)

                custom_id = COCO_TO_CUSTOM[cls_id]
                x1, y1, x2, y2 = box.xyxy[0].tolist()

                # Đổi sang định dạng YOLO: x_center, y_center, width, height (normalize 0-1)
                xc = ((x1 + x2) / 2) / w
                yc = ((y1 + y2) / 2) / h
                bw = (x2 - x1) / w
                bh = (y2 - y1) / h

                lines.append(f"{custom_id} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}")

        with open(label_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))

        if (idx + 1) % 50 == 0:
            print(f"  ...{idx + 1}/{len(image_files)}")

print("DONE! Nhãn sơ bộ đã lưu trong dataset/labels/train và dataset/labels/val")
print("Bước tiếp theo: mở LabelImg để SOÁT LẠI và sửa các nhãn sai (đặc biệt car/truck/bus).")
