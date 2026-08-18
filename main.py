# MAIN VIDEO PROCESSING LOOP
import cv2
import time
from collections import defaultdict

from ultralytics import YOLO

import globals as G
from utils.geometry import point_in_roi, side_of_line
from utils.drawing import draw_lines, draw_roi, draw_stats, draw_roi_count
from utils.violation_engine import ViolationEngine
from utils.behavior_analysis import BehaviorAnalyzer
from utils.license_plate import PlateReader
from utils import database

# ---- CẤU HÌNH ----
VEHICLE_MODEL_PATH = "models/yolov8n.pt"          # model detect xe (đã có sẵn)
PLATE_MODEL_PATH = "models/plate_yolov8n.pt"       # model detect biển số (cần train riêng)
LOCATION_NAME = "CAM_NGA_TU_01"
VEHICLE_CLASSES = {2: "car", 3: "motorbike", 5: "bus", 7: "truck"}  # theo COCO id, chỉnh lại nếu model custom


def run(video_source, enable_plate_reading=True):
    database.init_db()
    database.archive_and_clear_run()  # lưu dữ liệu lần chạy trước, xóa sạch để chạy mới
    G.reset_state()

    vehicle_model = YOLO(VEHICLE_MODEL_PATH)

    plate_reader = None
    if enable_plate_reading:
        try:
            plate_reader = PlateReader(PLATE_MODEL_PATH, use_gpu=(G.device_mode == "cuda"))
        except Exception as e:
            print(f"[WARN] Khong the khoi tao PlateReader: {e}. Chay tiep khong doc bien so.")

    violation_engine = ViolationEngine(
        plate_reader=plate_reader,
        location_name=LOCATION_NAME,
        evidence_dir="evidence",
        stop_threshold_sec=5.0,
    )

    # wrong_way_expected_dir: vector hướng đi đúng (dx, dy) theo tọa độ ảnh.
    # None = không kiểm tra đi ngược chiều. Cấu hình thủ công theo góc đặt
    # camera thực tế, ví dụ (1, 0) nếu hướng đúng là từ trái sang phải.
    behavior_analyzer = BehaviorAnalyzer(
        location_name=LOCATION_NAME,
        wrong_way_expected_dir=None,
    )

    cap = cv2.VideoCapture(video_source)
    if not cap.isOpened():
        raise RuntimeError(f"Khong mo duoc nguon video: {video_source}")

    # Thống kê theo line: {line_idx: {class_name: count}}
    line_stats = defaultdict(lambda: defaultdict(int))
    counted_crossings = set()  # (track_id, line_idx) đã đếm rồi, tránh đếm trùng
    prev_side_map = {}  # (track_id, line_idx) -> giá trị side_of_line ở frame trước

    frame_idx = 0
    prev_time = time.time()

    while cap.isOpened() and not G.stopped:
        if G.paused:
            time.sleep(0.05)
            continue

        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1

        # --- TRACKING ---
        # persist=True để YOLO giữ track ID xuyên suốt các frame (dùng ByteTrack mặc định)
        results = vehicle_model.track(frame, persist=True, verbose=False)

        if results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            track_ids = results[0].boxes.id.cpu().numpy().astype(int)
            class_ids = results[0].boxes.cls.cpu().numpy().astype(int)

            for bbox, track_id, cls_id in zip(boxes, track_ids, class_ids):
                cls_name = VEHICLE_CLASSES.get(cls_id, "unknown")
                x1, y1, x2, y2 = bbox
                cx, cy = (x1 + x2) / 2, (y1 + y2) / 2

                # --- ĐẾM XE QUA LINE (thống kê lưu lượng, không dùng cho vi phạm) ---
                for line_idx, line in enumerate(G.lines):
                    a = (line["p1"]["x"], line["p1"]["y"])
                    b = (line["p2"]["x"], line["p2"]["y"])
                    side = side_of_line((cx, cy), a, b)

                    prev_key = (track_id, line_idx)
                    prev_side = prev_side_map.get(prev_key)
                    prev_side_map[prev_key] = side

                    crossing_key = (track_id, line_idx)
                    if (prev_side is not None and prev_side * side < 0
                            and crossing_key not in counted_crossings):
                        counted_crossings.add(crossing_key)
                        line_stats[line_idx][cls_name] += 1
                        database.insert_traffic_stat(
                            LOCATION_NAME, str(line_idx), cls_name,
                            line.get("direction", "in")
                        )

                # --- ĐẾM XE TRONG ROI (dedup theo set ID) ---
                for roi_idx, roi in enumerate(G.roi_zones):
                    if point_in_roi((cx, cy), roi):
                        with G.state_lock:
                            G.roi_counts[roi_idx].add(track_id)

                # --- KIỂM TRA VI PHẠM ---
                # Chỉ tính vi phạm khi xe đi vào vùng ROI (restricted zone).
                # Đã tắt kiểm tra vượt đèn đỏ theo line vì traffic_light_state
                # chưa có nguồn dữ liệu thật - để bật sẽ sinh vi phạm giả cho
                # mọi xe đi qua line. Bật lại khi có logic detect đèn thật.
                violation_engine.check_restricted_zone(
                    frame, track_id, (x1, y1, x2, y2), cls_name, G.roi_zones
                )

                # --- PHÂN TÍCH HÀNH VI BẤT THƯỜNG ---
                behavior_events = behavior_analyzer.update(track_id, (cx, cy))
                for ev in behavior_events:
                    behavior_analyzer.record_event(ev, vehicle_class=cls_name)

                # Vẽ bbox + track id lên frame
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                cv2.putText(frame, f"{cls_name}#{track_id}", (int(x1), int(y1) - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

            # Dọn dẹp lịch sử các track_id không còn xuất hiện trong frame này
            behavior_analyzer.cleanup(set(track_ids.tolist()))

        # --- VẼ OVERLAY ---
        draw_lines(frame, G.lines)
        draw_roi(frame, G.roi_zones)
        draw_stats(frame, line_stats)
        with G.state_lock:
            draw_roi_count(frame, G.roi_counts)

        # FPS
        now = time.time()
        fps = 1.0 / max(now - prev_time, 1e-6)
        prev_time = now
        cv2.putText(frame, f"FPS: {fps:.1f}", (frame.shape[1] - 150, frame.shape[0] - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        yield frame  # cho phép app.py (Flask) lấy frame để stream ra web

    cap.release()


if __name__ == "__main__":
    # Chạy thử độc lập (hiển thị bằng cv2.imshow) - dùng để test nhanh
    # ngoài môi trường web. video_path lấy từ globals hoặc chỉnh trực tiếp.
    source = G.video_path or 0
    for frame in run(source, enable_plate_reading=False):
        cv2.imshow("Traffic Analytics", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            G.stopped = True
            break
    cv2.destroyAllWindows()