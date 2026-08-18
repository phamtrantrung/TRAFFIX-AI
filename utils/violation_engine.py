# VIOLATION ENGINE
# Kết hợp geometry (point_in_roi, side_of_line) + lịch sử tracking để
# phát hiện các loại vi phạm, sau đó gọi license_plate để đọc biển số
# và ghi vào database.

import os
import time
import cv2
from datetime import datetime

from utils.geometry import point_in_roi, side_of_line
from utils import database


class ViolationEngine:
    def __init__(self, plate_reader, location_name="CAM_01",
                 evidence_dir="evidence", stop_threshold_sec=5.0):
        """
        plate_reader: instance của PlateReader (utils/license_plate.py)
        location_name: tên điểm camera, lưu kèm vào DB
        """
        self.plate_reader = plate_reader
        self.location_name = location_name
        self.evidence_dir = evidence_dir

        os.makedirs(self.evidence_dir, exist_ok=True)

        # Trạng thái theo dõi từng track_id
        self._line_prev_side = {}
        self._already_flagged = set()  # (track_id, violation_type) đã ghi nhận

    def check_restricted_zone(self, frame, track_id, bbox, cls_name, roi_zones):
        """Phát hiện xe đi vào vùng ROI cấm - ghi vi phạm ngay khi vừa
        vào, không cần chờ đứng yên."""
        cx = (bbox[0] + bbox[2]) / 2
        cy = (bbox[1] + bbox[3]) / 2
        center = (cx, cy)

        for roi_idx, roi in enumerate(roi_zones):
            if point_in_roi(center, roi):
                flag_key = (track_id, "restricted_zone", roi_idx)
                if flag_key not in self._already_flagged:
                    self._already_flagged.add(flag_key)
                    self._record_violation(
                        frame, track_id, bbox, cls_name,
                        "restricted_zone"
                    )
                break

    def check_line_crossing_violation(self, track_id, bbox, lines,
                                       traffic_light_state="green"):
        """Phát hiện vượt đèn đỏ: track_id băng qua line trong khi
        traffic_light_state == 'red'. traffic_light_state cần được cung
        cấp từ nguồn khác (cảm biến/HSV màu đèn/đầu vào thủ công).
        Hiện chưa được main.py gọi tới - giữ lại để dùng khi có logic
        detect đèn thật."""
        cx = (bbox[0] + bbox[2]) / 2
        cy = (bbox[1] + bbox[3]) / 2
        point = (cx, cy)

        for line_idx, line in enumerate(lines):
            a = (line["p1"]["x"], line["p1"]["y"])
            b = (line["p2"]["x"], line["p2"]["y"])

            side = side_of_line(point, a, b)
            key = f"{track_id}_{line_idx}"
            prev_side = self._line_prev_side.get(key)

            crossed = (
                prev_side is not None
                and prev_side * side < 0  # đổi dấu = đã băng qua
            )

            self._line_prev_side[key] = side

            if crossed and traffic_light_state == "red":
                flag_key = (track_id, "red_light", line_idx)
                if flag_key not in self._already_flagged:
                    self._already_flagged.add(flag_key)
                    return True, line_idx

        return False, None

    def _record_violation(self, frame, track_id, bbox, cls_name, violation_type):
        """Lưu ảnh bằng chứng, chạy OCR biển số, ghi vào database."""
        x1, y1, x2, y2 = [int(v) for v in bbox]

        plate_info = None
        if self.plate_reader is not None:
            try:
                plate_info = self.plate_reader.process_vehicle(frame, (x1, y1, x2, y2))
            except Exception as e:
                print(f"[ViolationEngine] Loi doc bien so: {e}")

        plate_text = plate_info["plate_text"] if plate_info else None
        plate_conf = plate_info["confidence"] if plate_info else None

        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        evidence_filename = f"{violation_type}_{track_id}_{timestamp_str}.jpg"
        evidence_path = os.path.join(self.evidence_dir, evidence_filename)

        annotated = frame.copy()
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 0, 255), 3)
        label = f"{violation_type} | {plate_text or 'UNKNOWN'}"
        cv2.putText(annotated, label, (x1, max(20, y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.imwrite(evidence_path, annotated)

        database.insert_violation(
            track_id=track_id,
            plate_number=plate_text,
            plate_confidence=plate_conf,
            violation_type=violation_type,
            vehicle_class=cls_name,
            location=self.location_name,
            evidence_image_path=evidence_path,
        )

        return {
            "track_id": track_id,
            "violation_type": violation_type,
            "plate_text": plate_text,
            "evidence_path": evidence_path,
        }

    def record_red_light_violation(self, frame, track_id, bbox, cls_name):
        """Gọi khi check_line_crossing_violation trả về True. Hiện chưa
        được main.py gọi tới - giữ lại để dùng khi bật lại tính năng
        phát hiện vượt đèn đỏ."""
        return self._record_violation(frame, track_id, bbox, cls_name, "red_light")