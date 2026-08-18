# BEHAVIOR ANALYSIS MODULE
# Phát hiện các hành vi bất thường của phương tiện dựa trên lịch sử
# quỹ đạo di chuyển (không chỉ đơn thuần trong/ngoài vùng ROI như
# violation_engine). Đây là lớp "phân tích hành vi" bổ sung, tách biệt
# khỏi việc xác định vi phạm mang tính pháp lý.

import math
import time
from collections import deque, defaultdict

from utils import database


class BehaviorAnalyzer:
    def __init__(self, location_name="CAM_01",
                 history_len=30,          # số điểm lịch sử giữ lại mỗi track_id
                 stop_speed_threshold=2.0,  # px/s - dưới ngưỡng này coi là gần như đứng yên
                 stop_duration_sec=4.0,     # đứng yên ngoài ROI cấm quá lâu -> nghi vấn sự cố/tai nạn
                 weaving_angle_threshold=35, # độ lệch góc di chuyển (độ) để coi là lạng lách
                 wrong_way_expected_dir=None):  # vector hướng đi đúng (dx, dy), None = không kiểm tra
        self.location_name = location_name
        self.history_len = history_len
        self.stop_speed_threshold = stop_speed_threshold
        self.stop_duration_sec = stop_duration_sec
        self.weaving_angle_threshold = weaving_angle_threshold
        self.wrong_way_expected_dir = wrong_way_expected_dir

        # track_id -> deque[(timestamp, x, y)]
        self._history = defaultdict(lambda: deque(maxlen=history_len))
        # track_id -> timestamp bắt đầu đứng yên (None nếu đang di chuyển)
        self._stationary_since = {}
        self._flagged_events = set()  # (track_id, event_type) đã ghi nhận, tránh lặp

    def update(self, track_id, center_point, timestamp=None):
        """Gọi mỗi frame cho mỗi track_id đang active. Trả về list các
        event bất thường phát hiện được ở frame này (có thể rỗng)."""
        ts = timestamp if timestamp is not None else time.time()
        hist = self._history[track_id]
        hist.append((ts, center_point[0], center_point[1]))

        events = []

        if len(hist) < 3:
            return events  # chưa đủ dữ liệu để phân tích

        speed = self._instant_speed(hist)

        stop_event = self._check_sudden_stop(track_id, speed, ts)
        if stop_event:
            events.append(stop_event)

        weave_event = self._check_lane_weaving(track_id, hist)
        if weave_event:
            events.append(weave_event)

        if self.wrong_way_expected_dir is not None:
            wrong_way_event = self._check_wrong_way(track_id, hist)
            if wrong_way_event:
                events.append(wrong_way_event)

        return events

    def _instant_speed(self, hist):
        """Tốc độ tức thời (px/giây) dựa trên 2 điểm gần nhất."""
        (t1, x1, y1), (t2, x2, y2) = hist[-2], hist[-1]
        dt = t2 - t1
        if dt <= 0:
            return 0.0
        dist = math.hypot(x2 - x1, y2 - y1)
        return dist / dt

    def _check_sudden_stop(self, track_id, speed, ts):
        """Phát hiện xe đột ngột đứng yên (nghi vấn va chạm/sự cố) -
        khác với đỗ xe trong ROI cấm (đã xử lý ở violation_engine)."""
        if speed < self.stop_speed_threshold:
            if track_id not in self._stationary_since:
                self._stationary_since[track_id] = ts
            elif ts - self._stationary_since[track_id] >= self.stop_duration_sec:
                flag_key = (track_id, "sudden_stop")
                if flag_key not in self._flagged_events:
                    self._flagged_events.add(flag_key)
                    return {"track_id": track_id, "event_type": "sudden_stop",
                             "detail": f"Dung yen bat thuong > {self.stop_duration_sec}s"}
        else:
            self._stationary_since.pop(track_id, None)
        return None

    def _check_lane_weaving(self, track_id, hist):
        """Phát hiện lạng lách: tính góc di chuyển giữa các đoạn liên
        tiếp trong cửa sổ lịch sử, nếu độ lệch góc trung bình vượt
        ngưỡng -> nghi vấn lạng lách/đi không ổn định làn."""
        if len(hist) < 6:
            return None

        angles = []
        pts = list(hist)[-6:]
        for i in range(1, len(pts)):
            _, x1, y1 = pts[i - 1]
            _, x2, y2 = pts[i]
            dx, dy = x2 - x1, y2 - y1
            if dx == 0 and dy == 0:
                continue
            angles.append(math.degrees(math.atan2(dy, dx)))

        if len(angles) < 3:
            return None

        # Tính độ lệch góc lớn nhất giữa các đoạn liên tiếp
        max_delta = 0
        for i in range(1, len(angles)):
            delta = abs(angles[i] - angles[i - 1])
            delta = min(delta, 360 - delta)  # chuẩn hóa về [0, 180]
            max_delta = max(max_delta, delta)

        if max_delta >= self.weaving_angle_threshold:
            flag_key = (track_id, "lane_weaving")
            if flag_key not in self._flagged_events:
                self._flagged_events.add(flag_key)
                return {"track_id": track_id, "event_type": "lane_weaving",
                         "detail": f"Doi huong dot ngot {max_delta:.0f} do"}
        return None

    def _check_wrong_way(self, track_id, hist):
        """Phát hiện đi ngược chiều: so sánh vector hướng di chuyển
        thực tế với hướng đi đúng đã cấu hình (wrong_way_expected_dir)."""
        (t1, x1, y1), (t2, x2, y2) = hist[-3], hist[-1]
        dx, dy = x2 - x1, y2 - y1
        dist = math.hypot(dx, dy)
        if dist < 5:  # di chuyển quá ít, chưa đủ tin cậy để đánh giá hướng
            return None

        ex, ey = self.wrong_way_expected_dir
        expected_len = math.hypot(ex, ey)
        if expected_len == 0:
            return None

        cos_angle = (dx * ex + dy * ey) / (dist * expected_len)

        if cos_angle < -0.5:  # lệch hướng đúng > ~120 độ -> đi ngược chiều
            flag_key = (track_id, "wrong_way")
            if flag_key not in self._flagged_events:
                self._flagged_events.add(flag_key)
                return {"track_id": track_id, "event_type": "wrong_way",
                         "detail": "Di chuyen nguoc chieu quy dinh"}
        return None

    def record_event(self, event, vehicle_class=None):
        """Ghi sự kiện bất thường vào database (bảng behavior_events)."""
        database.insert_behavior_event(
            track_id=event["track_id"],
            event_type=event["event_type"],
            detail=event["detail"],
            vehicle_class=vehicle_class,
            location=self.location_name,
        )

    def cleanup(self, active_track_ids):
        """Dọn dẹp lịch sử của các track_id không còn xuất hiện trong
        frame hiện tại, tránh rò rỉ bộ nhớ khi chạy dài hạn."""
        stale_ids = [tid for tid in self._history if tid not in active_track_ids]
        for tid in stale_ids:
            del self._history[tid]
            self._stationary_since.pop(tid, None)
