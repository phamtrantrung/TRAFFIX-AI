# GEOMETRY FUNCTIONS
import cv2
import numpy as np


def point_in_roi(point, polygon):
    """Kiểm tra một điểm có nằm trong vùng ROI (đa giác bất kỳ, từ 3 điểm
    trở lên - hỗ trợ cả ROI vẽ tự do freehand) hay không."""
    if len(polygon) < 3:
        return False

    pts = np.array(
        [[int(p["x"]), int(p["y"])] for p in polygon],
        np.int32
    )
    pts = pts.reshape((-1, 1, 2))

    point_f = (float(point[0]), float(point[1]))

    return cv2.pointPolygonTest(pts, point_f, False) >= 0


def side_of_line(p, a, b):
    """Tích có hướng - xác định điểm p nằm bên nào của đường thẳng a-b.
    Dương: bên trái | Âm: bên phải | 0: nằm trên đường thẳng."""
    return (
        (b[0] - a[0]) * (p[1] - a[1])
        - (b[1] - a[1]) * (p[0] - a[0])
    )


def point_near_line(p, a, b, tolerance=10):
    """Kiểm tra khoảng cách vuông góc từ điểm p đến đoạn thẳng a-b có
    nằm trong ngưỡng dung sai hay không. Chính xác hơn point_on_segment
    (bounding-box) vì tính đúng khoảng cách hình học, không phụ thuộc
    góc nghiêng của đường thẳng."""
    ax, ay = a
    bx, by = b

    line_len = ((bx - ax) ** 2 + (by - ay) ** 2) ** 0.5
    if line_len == 0:
        return False

    dist = abs(side_of_line(p, a, b)) / line_len

    # Đồng thời kiểm tra điểm chiếu có nằm trong đoạn thẳng (không phải
    # phần kéo dài vô hạn của đường thẳng)
    t = ((p[0] - ax) * (bx - ax) + (p[1] - ay) * (by - ay)) / (line_len ** 2)
    if t < -0.05 or t > 1.05:
        return False

    return dist <= tolerance


def point_on_segment(p, a, b):
    """[Giữ lại để tương thích ngược] Kiểm tra điểm có nằm trong
    bounding-box mở rộng của đoạn thẳng a-b. Lưu ý: không chính xác
    100% với đường xiên góc lớn - nên ưu tiên dùng point_near_line."""
    px, py = p
    ax, ay = a
    bx, by = b

    min_x = min(ax, bx) - 10
    max_x = max(ax, bx) + 10
    min_y = min(ay, by) - 10
    max_y = max(ay, by) + 10

    return (min_x <= px <= max_x) and (min_y <= py <= max_y)


def point_projects_onto_segment(p, a, b, tolerance=0.05):
    """Kiểm tra điểm p có chiếu vuông góc rơi vào TRONG đoạn thẳng a-b hay
    không (không tính phần đường thẳng kéo dài vô hạn). tolerance là biên
    nới lỏng 2 đầu đoạn (theo tỉ lệ t, 0-1), tránh bỏ sót các trường hợp
    xe cắt sát ngay đầu mút line."""
    ax, ay = a
    bx, by = b
    line_len_sq = (bx - ax) ** 2 + (by - ay) ** 2
    if line_len_sq == 0:
        return False

    t = ((p[0] - ax) * (bx - ax) + (p[1] - ay) * (by - ay)) / line_len_sq
    return -tolerance <= t <= 1 + tolerance