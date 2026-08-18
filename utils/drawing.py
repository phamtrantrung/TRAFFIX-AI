# DRAWING FUNCTIONS
import cv2
import numpy as np

# COLORS (BGR - OpenCV dùng BGR chứ không phải RGB)
COLORS = [
    (0, 255, 0),
    (0, 0, 255),
    (255, 0, 0),
    (0, 255, 255),
    (255, 0, 255),
]


def draw_lines(frame, lines):
    for idx, line in enumerate(lines):
        p1 = (int(line["p1"]["x"]), int(line["p1"]["y"]))
        p2 = (int(line["p2"]["x"]), int(line["p2"]["y"]))
        direction = line.get("direction", "in")
        color = COLORS[idx % len(COLORS)]

        cv2.line(frame, p1, p2, color, 3)
        cv2.putText(
            frame, direction.upper(), p1,
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2
        )


def draw_roi(frame, roi_zones):
    for idx, roi in enumerate(roi_zones):
        if len(roi) != 4:
            continue

        pts = np.array(
            [[int(p["x"]), int(p["y"])] for p in roi],
            np.int32
        )
        pts = pts.reshape((-1, 1, 2))

        overlay = frame.copy()
        cv2.fillPoly(overlay, [pts], (0, 0, 255))
        cv2.addWeighted(overlay, 0.25, frame, 0.75, 0, frame)
        cv2.polylines(frame, [pts], True, (0, 0, 255), 3)

        # FIX: nhãn đặt tại điểm trên cùng (topmost) thay vì điểm đầu tiên
        # trong danh sách - tránh nhãn hiện sai vị trí tùy thứ tự người
        # dùng click khi vẽ ROI.
        top_point = min(roi, key=lambda p: p["y"])
        label_pos = (int(top_point["x"]) + 10, int(top_point["y"]) - 10)

        cv2.putText(
            frame, f"VIOLATION ZONE {idx + 1}", label_pos,
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2
        )


def draw_stats(frame, stats, frame_height=None):
    """stats: dict dạng {line_key: {class_name: count}}.
    line_key có thể là bất kỳ kiểu nào (int, str...) - không giả định
    là số nguyên tuần tự."""
    y_offset = 30
    max_y = frame_height if frame_height else frame.shape[0]

    # FIX: dùng enumerate() bọc ngoài để có chỉ số hiển thị/màu sắc an
    # toàn, tách biệt với key thật của dict (key thật chỉ dùng để hiển thị)
    for display_idx, (key, stat) in enumerate(stats.items()):
        if y_offset > max_y - 20:
            break  # FIX: tránh vẽ tràn ra ngoài đáy frame

        cv2.putText(
            frame, f"LINE {key}", (20, y_offset),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7,
            COLORS[display_idx % len(COLORS)], 2
        )
        y_offset += 25

        for cls_name, value in stat.items():
            if y_offset > max_y - 10:
                break
            cv2.putText(
                frame, f"{cls_name}: {value}", (40, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2
            )
            y_offset += 25

        y_offset += 10


def draw_roi_count(frame, roi_counts):
    roi_y = 40
    frame_h, frame_w = frame.shape[0], frame.shape[1]

    # FIX: đảm bảo x không âm trên frame có độ phân giải thấp
    x_pos = max(10, frame_w - 250)

    for zone_idx, ids_set in roi_counts.items():
        if roi_y > frame_h - 10:
            break  # FIX: tránh vẽ tràn ra ngoài đáy frame khi nhiều ROI

        cv2.putText(
            frame, f"ROI {zone_idx}: {len(ids_set)}",
            (x_pos, roi_y),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2
        )
        roi_y += 35


def draw_violation_alert(frame, plate_text, violation_type):
    """Vẽ cảnh báo vi phạm nổi bật lên góc trên bên trái của frame."""
    label = f"VI PHAM: {violation_type} | Bien so: {plate_text}"
    (text_w, text_h), _ = cv2.getTextSize(
        label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2
    )
    cv2.rectangle(frame, (10, 10), (20 + text_w, 40 + text_h), (0, 0, 200), -1)
    cv2.putText(
        frame, label, (15, 35),
        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2
    )
