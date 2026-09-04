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


def _draw_label_with_bg(frame, text, org, color, font_scale=0.6, thickness=2,
                         text_color=(255, 255, 255), padding=4):
    """Vẽ chữ có nền (hộp bo màu color phía sau, chữ màu text_color đè
    lên trên) để luôn đọc được rõ ràng bất kể nền video là gì (cỏ, đường,
    trời...). org là điểm góc dưới-trái của chữ (giống cv2.putText)."""
    (text_w, text_h), baseline = cv2.getTextSize(
        text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness
    )
    x, y = org
    top_left = (x - padding, y - text_h - padding)
    bottom_right = (x + text_w + padding, y + baseline + padding)

    overlay = frame.copy()
    cv2.rectangle(overlay, top_left, bottom_right, color, -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)
    cv2.rectangle(frame, top_left, bottom_right, color, 1, cv2.LINE_AA)

    cv2.putText(
        frame, text, org,
        cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color, thickness, cv2.LINE_AA
    )


def _draw_arrow_on_line(frame, p1, p2, color, direction):
    """Vẽ mũi tên nhỏ ở giữa line, chỉ theo hướng di chuyển (direction:
    'in' hoặc 'out') để trực quan hơn thay vì chỉ có chữ IN/OUT."""
    mx, my = (p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2
    dx, dy = (p2[0] - p1[0]), (p2[1] - p1[1])
    length = max((dx ** 2 + dy ** 2) ** 0.5, 1e-6)
    ux, uy = dx / length, dy / length  # vector đơn vị dọc theo line

    # Vector vuông góc với line - hướng "in"/"out" thực tế phụ thuộc vào
    # cách đặt camera, ở đây chỉ minh họa trực quan hướng vuông góc line
    # để phân biệt rõ 2 chiều, không suy luận hướng thật của xe.
    perp_x, perp_y = -uy, ux
    if direction == "out":
        perp_x, perp_y = -perp_x, -perp_y

    arrow_len = 22
    start = (int(mx), int(my))
    end = (int(mx + perp_x * arrow_len), int(my + perp_y * arrow_len))
    cv2.arrowedLine(frame, start, end, color, 3, cv2.LINE_AA, tipLength=0.4)


def draw_lines(frame, lines):
    for idx, line in enumerate(lines):
        p1 = (int(line["p1"]["x"]), int(line["p1"]["y"]))
        p2 = (int(line["p2"]["x"]), int(line["p2"]["y"]))
        direction = line.get("direction", "in")
        color = COLORS[idx % len(COLORS)]

        # Line chính - dày hơn, chống răng cưa (anti-aliased)
        cv2.line(frame, p1, p2, color, 3, cv2.LINE_AA)

        # Đầu line bo tròn (chấm tròn viền trắng ở 2 đầu) cho mềm mại hơn
        for p in (p1, p2):
            cv2.circle(frame, p, 6, color, -1, cv2.LINE_AA)
            cv2.circle(frame, p, 6, (255, 255, 255), 1, cv2.LINE_AA)

        # Mũi tên nhỏ giữa line chỉ hướng, thay vì chỉ chữ IN/OUT khô khan
        _draw_arrow_on_line(frame, p1, p2, color, direction)

        # Nhãn "LINE n · IN/OUT" có nền, đặt phía trên điểm đầu line
        label = f"LINE {idx} · {direction.upper()}"
        label_org = (p1[0], p1[1] - 14)
        _draw_label_with_bg(frame, label, label_org, color, font_scale=0.6, thickness=2)


def draw_roi(frame, roi_zones):
    for idx, roi in enumerate(roi_zones):
        if len(roi) < 3:
            continue  # cần tối thiểu 3 điểm mới tạo được vùng khép kín

        pts = np.array(
            [[int(p["x"]), int(p["y"])] for p in roi],
            np.int32
        )
        pts = pts.reshape((-1, 1, 2))

        overlay = frame.copy()
        cv2.fillPoly(overlay, [pts], (0, 0, 255))
        cv2.addWeighted(overlay, 0.25, frame, 0.75, 0, frame)
        cv2.polylines(frame, [pts], True, (0, 0, 255), 3, cv2.LINE_AA)

        top_point = min(roi, key=lambda p: p["y"])
        label_pos = (int(top_point["x"]) + 10, int(top_point["y"]) - 10)

        _draw_label_with_bg(
            frame, f"VIOLATION ZONE {idx + 1}", label_pos,
            (0, 0, 255), font_scale=0.7, thickness=2
        )


def draw_stats(frame, stats, frame_height=None):
    """stats: dict dạng {line_key: {class_name: count}}.
    line_key có thể là bất kỳ kiểu nào (int, str...) - không giả định
    là số nguyên tuần tự."""
    y_offset = 30
    max_y = frame_height if frame_height else frame.shape[0]

    for display_idx, (key, stat) in enumerate(stats.items()):
        if y_offset > max_y - 20:
            break  # tránh vẽ tràn ra ngoài đáy frame

        color = COLORS[display_idx % len(COLORS)]
        _draw_label_with_bg(
            frame, f"LINE {key}", (20, y_offset), color,
            font_scale=0.7, thickness=2
        )
        y_offset += 30

        if not stat:
            # Chưa có xe nào qua line này - hiện dòng "chưa có dữ liệu"
            # thay vì để trống hoàn toàn, để người xem biết line đã sẵn
            # sàng đếm chứ không phải bị lỗi.
            cv2.putText(
                frame, "(chua co xe qua)", (40, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (150, 150, 150), 1, cv2.LINE_AA
            )
            y_offset += 22

        for cls_name, value in stat.items():
            if y_offset > max_y - 10:
                break
            cv2.putText(
                frame, f"{cls_name}: {value}", (40, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA
            )
            y_offset += 25

        y_offset += 10


def draw_roi_count(frame, roi_counts):
    roi_y = 40
    frame_h, frame_w = frame.shape[0], frame.shape[1]

    x_pos = max(10, frame_w - 250)

    for zone_idx, ids_set in roi_counts.items():
        if roi_y > frame_h - 10:
            break  # tránh vẽ tràn ra ngoài đáy frame khi nhiều ROI

        _draw_label_with_bg(
            frame, f"ROI {zone_idx}: {len(ids_set)}", (x_pos, roi_y),
            (0, 0, 255), font_scale=0.8, thickness=2
        )
        roi_y += 40


def draw_violation_alert(frame, plate_text, violation_type):
    """Vẽ cảnh báo vi phạm nổi bật lên góc trên bên trái của frame."""
    label = f"VI PHAM: {violation_type} | Bien so: {plate_text}"
    (text_w, text_h), _ = cv2.getTextSize(
        label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2
    )
    cv2.rectangle(frame, (10, 10), (20 + text_w, 40 + text_h), (0, 0, 200), -1)
    cv2.putText(
        frame, label, (15, 35),
        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA
    )
