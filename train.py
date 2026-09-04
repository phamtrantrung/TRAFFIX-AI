from ultralytics import YOLO

# FIX: dùng yolov8s.pt (bản đang chạy thật trong hệ thống) làm điểm
# xuất phát để fine-tune, thay vì yolo11n.pt (khác dòng model, và "n"
# là bản nhỏ nhất - không tận dụng được lợi thế độ chính xác cao hơn
# của "s" đã chọn ở main.py). Fine-tune từ model đã quen nhận diện xe
# nói chung sẽ hội tụ nhanh hơn train từ đầu.
model = YOLO("models/yolov8s.pt")

if __name__ == "__main__":
    model.train(
        data="data.yaml",       # file cấu hình dataset
        epochs=80,               # dữ liệu ít nên cần nhiều epoch hơn 1 chút
        imgsz=640,
        batch=8,                 # giảm nếu bị hết RAM/VRAM, tăng nếu máy khỏe
        device="cpu",             # đổi thành '0' nếu có GPU NVIDIA
        patience=20,              # dừng sớm nếu 20 epoch liền không cải thiện
        project="runs/detect",
        name="vn_vehicles_v1",
        exist_ok=True,
    )

    print("\nDONE! Model tốt nhất nằm ở: runs/detect/vn_vehicles_v1/weights/best.pt")
    print("Copy file đó vào models/ (vd: models/yolov8s_vn.pt) rồi cập nhật")
    print("VEHICLE_MODEL_PATH trong main.py để dùng model mới train.")
