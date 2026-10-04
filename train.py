"""
train.py — Fine-tune YOLOv8s trên dataset xe Việt Nam tự xây dựng (TRAFFIX-AI)

CHUẨN BỊ TRƯỚC KHI CHẠY (đã hoàn thành ở các bước trước):
  1. extract_frames.py  -> trích khung hình từ video giám sát thật
  2. auto_label.py       -> sinh nhãn sơ bộ bằng yolov8s gốc (COCO)
  3. Rà soát + sửa nhãn thủ công trên Make Sense (car/truck/bus dễ nhầm,
     motorbike ở xa dễ bị bỏ sót), Export lại YOLO format, đè vào
     dataset/labels/train và dataset/labels/val
  4. Đặt file data.yaml (đi kèm) vào trong thư mục dataset/

CÁCH CHẠY:
    python train.py

Mặc định: 50 epoch, ảnh 640x640, tự nhận diện GPU nếu máy có CUDA
(nếu không có GPU thì tự chạy CPU, chỉ chậm hơn chứ không lỗi).
Có thể chỉnh nhanh các thông số ở phần CONFIG bên dưới mà không cần
truyền tham số dòng lệnh.
"""

import os
import shutil
from datetime import datetime

from ultralytics import YOLO
import torch


# ============== CONFIG - chỉnh ở đây nếu cần ==============
BASE_MODEL = "models/yolov8s.pt"      # model gốc COCO dùng làm điểm khởi đầu (transfer learning)
DATA_YAML = "dataset/data.yaml"       # file cấu hình dataset (đi kèm script này)

EPOCHS = 50
IMG_SIZE = 640
BATCH = 8            # giảm xuống 4 nếu bị lỗi hết RAM/VRAM khi chạy
PATIENCE = 15         # dừng sớm nếu val loss không cải thiện sau 15 epoch liên tiếp

PROJECT_DIR = "runs/train"            # thư mục ultralytics tự lưu kết quả từng lần chạy
RUN_NAME = "traffix_vn_vehicles"      # tên lần chạy này (tạo thư mục runs/train/traffix_vn_vehicles)

# Nơi copy model tốt nhất sau khi train xong, để dùng luôn trong main.py
FINAL_MODEL_DEST = "models/yolov8s_vn.pt"
# ============================================================


def main():
    if not os.path.exists(BASE_MODEL):
        raise FileNotFoundError(
            f"Không tìm thấy model gốc: {BASE_MODEL}\n"
            f"Kiểm tra lại đường dẫn hoặc tải yolov8s.pt về đặt vào thư mục models/."
        )
    if not os.path.exists(DATA_YAML):
        raise FileNotFoundError(
            f"Không tìm thấy {DATA_YAML}\n"
            f"Copy file data.yaml (đi kèm) vào trong thư mục dataset/ rồi chạy lại."
        )

    device = 0 if torch.cuda.is_available() else "cpu"
    print(f"Thiết bị huấn luyện: {'GPU (CUDA)' if device == 0 else 'CPU'}")
    if device == "cpu":
        print("  -> Chạy trên CPU sẽ khá chậm (có thể vài giờ tùy cấu hình máy),")
        print("     nhưng vẫn chạy được bình thường, không cần dừng giữa chừng.")

    print(f"\nBắt đầu fine-tune từ {BASE_MODEL}")
    print(f"Dataset: {DATA_YAML}")
    print(f"Epochs: {EPOCHS} | Ảnh: {IMG_SIZE}x{IMG_SIZE} | Batch: {BATCH}\n")

    model = YOLO(BASE_MODEL)

    results = model.train(
        data=DATA_YAML,
        epochs=EPOCHS,
        imgsz=IMG_SIZE,
        batch=BATCH,
        patience=PATIENCE,
        device=device,
        project=PROJECT_DIR,
        name=RUN_NAME,
        exist_ok=True,          # cho phép chạy lại đè lên cùng tên (khi thử nghiệm nhiều lần)
        # augmentation nhẹ, hợp lý cho ảnh giao thông (không lật ngang vì sẽ đảo chiều xe)
        fliplr=0.0,
        mosaic=1.0,
        degrees=5.0,
        translate=0.1,
        scale=0.3,
    )

    # Model tốt nhất (val mAP cao nhất) ultralytics tự lưu tại đây
    best_weights = os.path.join(PROJECT_DIR, RUN_NAME, "weights", "best.pt")

    if os.path.exists(best_weights):
        os.makedirs(os.path.dirname(FINAL_MODEL_DEST) or ".", exist_ok=True)
        shutil.copy2(best_weights, FINAL_MODEL_DEST)
        print(f"\nDONE! Model tốt nhất đã copy vào: {FINAL_MODEL_DEST}")
        print(f"(Bản gốc đầy đủ log/biểu đồ nằm ở: {PROJECT_DIR}/{RUN_NAME}/)")
        print("\nBước tiếp theo để dùng model mới trong hệ thống:")
        print(f"  1. Mở main.py, tìm dòng VEHICLE_MODEL_PATH")
        print(f"  2. Đổi giá trị thành: \"{FINAL_MODEL_DEST}\"")
        print(f"  3. Chạy lại python app.py và test thử vài phút xem car/truck/bus")
        print(f"     có còn bị nhận nhầm như trước không")
        print(f"\nMẹo: giữ nguyên models/yolov8s.pt cũ, để có thể đổi lại nếu model mới")
        print(f"     chưa tốt hơn (ví dụ dataset còn ít ảnh cho 1 loại xe nào đó).")
    else:
        print(f"\nCẢNH BÁO: không tìm thấy {best_weights}")
        print("Kiểm tra lại log phía trên xem quá trình train có báo lỗi giữa chừng không.")

    print(f"\nHoàn tất lúc: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()
