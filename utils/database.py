# DATABASE MODULE
import sqlite3
import os
from datetime import datetime
from contextlib import contextmanager

DB_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "traffic.db")


SCHEMA = """
CREATE TABLE IF NOT EXISTS violations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    track_id INTEGER,
    plate_number TEXT,
    plate_confidence REAL,
    violation_type TEXT NOT NULL,
    vehicle_class TEXT,
    location TEXT,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
    evidence_image_path TEXT,
    reviewed INTEGER DEFAULT 0
);

CREATE TABLE IF NOT EXISTS traffic_stats (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
    location TEXT,
    line_id TEXT,
    vehicle_class TEXT,
    direction TEXT,
    count INTEGER DEFAULT 1
);

CREATE TABLE IF NOT EXISTS roi_occupancy (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
    location TEXT,
    roi_id TEXT,
    vehicle_count INTEGER
);

CREATE TABLE IF NOT EXISTS behavior_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    track_id INTEGER,
    event_type TEXT NOT NULL,      -- 'sudden_stop', 'lane_weaving', 'wrong_way'
    detail TEXT,
    vehicle_class TEXT,
    location TEXT,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_violations_timestamp ON violations(timestamp);
CREATE INDEX IF NOT EXISTS idx_stats_timestamp ON traffic_stats(timestamp);
CREATE INDEX IF NOT EXISTS idx_behavior_timestamp ON behavior_events(timestamp);
"""

# Schema mô tả súc tích để đưa vào prompt cho chatbot (Text-to-SQL)
SCHEMA_DESCRIPTION = """
Bảng violations (vi phạm giao thông):
  - id: mã vi phạm
  - track_id: ID theo dõi xe trong video
  - plate_number: biển số xe (có thể NULL nếu OCR không đọc được)
  - plate_confidence: độ tin cậy OCR (0-1)
  - violation_type: loại vi phạm ('red_light', 'wrong_lane', 'illegal_parking', 'restricted_zone')
  - vehicle_class: loại xe ('car', 'motorbike', 'truck', 'bus')
  - location: tên điểm camera giám sát
  - timestamp: thời điểm vi phạm
  - evidence_image_path: đường dẫn ảnh bằng chứng
  - reviewed: đã được người kiểm duyệt xác nhận hay chưa (0/1)

Bảng traffic_stats (thống kê lưu lượng xe qua line đếm):
  - timestamp: thời điểm ghi nhận
  - location: tên điểm camera
  - line_id: tên/id đường kẻ đếm
  - vehicle_class: loại xe
  - direction: hướng di chuyển ('in'/'out')
  - count: số lượng (thường là 1 mỗi bản ghi, cộng dồn khi truy vấn)

Bảng roi_occupancy (mật độ xe trong vùng ROI theo thời gian):
  - timestamp, location, roi_id, vehicle_count

Bảng behavior_events (hành vi bất thường phát hiện được - không nhất
thiết là vi phạm pháp lý, dùng cho mục đích phân tích/cảnh báo sớm):
  - track_id: ID theo dõi xe
  - event_type: loại hành vi ('sudden_stop' - dừng đột ngột bất thường,
    'lane_weaving' - lạng lách, 'wrong_way' - đi ngược chiều)
  - detail: mô tả chi tiết
  - vehicle_class: loại xe
  - location: điểm camera
  - timestamp: thời điểm phát hiện
"""


def init_db():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    with get_connection() as conn:
        conn.executescript(SCHEMA)
        conn.commit()


@contextmanager
def get_connection():
    conn = sqlite3.connect(DB_PATH, timeout=10)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        conn.close()


def insert_violation(track_id, plate_number, plate_confidence, violation_type,
                      vehicle_class, location, evidence_image_path):
    # Ép track_id về int chuẩn của Python, dù đầu vào là numpy.int32,
    # numpy.float32, bytes, hay str số - tránh lưu sai kiểu vào cột INTEGER
    if track_id is not None:
        if isinstance(track_id, bytes):
            try:
                track_id = int.from_bytes(track_id, byteorder="little")
            except Exception:
                track_id = int(track_id.decode("utf-8", errors="ignore"))
        else:
            track_id = int(track_id)

    with get_connection() as conn:
        cur = conn.execute(
            """INSERT INTO violations
               (track_id, plate_number, plate_confidence, violation_type,
                vehicle_class, location, evidence_image_path, timestamp)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (track_id, plate_number, plate_confidence, violation_type,
             vehicle_class, location, evidence_image_path, datetime.now())
        )
        conn.commit()
        return cur.lastrowid


def insert_traffic_stat(location, line_id, vehicle_class, direction):
    # FIX (lệch múi giờ): trước đây câu INSERT không truyền timestamp,
    # nên SQLite tự dùng DEFAULT CURRENT_TIMESTAMP của cột - giá trị này
    # luôn là giờ UTC, KHÔNG phải giờ local máy chạy (VN = UTC+7). Trong
    # khi đó insert_violation()/insert_behavior_event() đều truyền tường
    # minh datetime.now() (giờ local), nên timestamp của traffic_stats bị
    # lệch ~7 tiếng so với violations/behavior_events. Hậu quả: lọc báo
    # cáo/trang Phân tích theo khoảng giờ local sẽ bỏ sót gần hết dữ liệu
    # traffic_stats thật (chúng nằm ở mốc giờ UTC khác hẳn), dù xe đã
    # thực sự đi qua line đúng lúc đó. Sửa: truyền tường minh datetime.now()
    # giống 2 hàm insert kia, để cả 3 bảng dùng chung 1 chuẩn giờ local.
    with get_connection() as conn:
        conn.execute(
            """INSERT INTO traffic_stats
               (location, line_id, vehicle_class, direction, count, timestamp)
               VALUES (?, ?, ?, ?, 1, ?)""",
            (location, line_id, vehicle_class, direction, datetime.now())
        )
        conn.commit()


def insert_roi_occupancy(location, roi_id, vehicle_count):
    # FIX (lệch múi giờ): lỗi y hệt insert_traffic_stat() ở trên - trước
    # đây không truyền timestamp nên bị SQLite tự gán giờ UTC thay vì
    # giờ local, gây lệch dữ liệu khi lọc theo khoảng thời gian local.
    with get_connection() as conn:
        conn.execute(
            """INSERT INTO roi_occupancy (location, roi_id, vehicle_count, timestamp)
               VALUES (?, ?, ?, ?)""",
            (location, roi_id, vehicle_count, datetime.now())
        )
        conn.commit()


def insert_behavior_event(track_id, event_type, detail, vehicle_class, location):
    with get_connection() as conn:
        cur = conn.execute(
            """INSERT INTO behavior_events
               (track_id, event_type, detail, vehicle_class, location, timestamp)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (track_id, event_type, detail, vehicle_class, location, datetime.now())
        )
        conn.commit()
        return cur.lastrowid


def get_recent_behavior_events(limit=50):
    with get_connection() as conn:
        rows = conn.execute(
            "SELECT * FROM behavior_events ORDER BY timestamp DESC LIMIT ?",
            (limit,)
        ).fetchall()
        return [dict(r) for r in rows]


# --- Truy vấn phục vụ trang Phân tích (Analytics) trên dashboard ---

def _normalize_datetime_local(value):
    """Chuẩn hóa giá trị từ <input type="datetime-local"> (dạng
    '2026-08-18T14:44', dùng chữ 'T', không có giây) về đúng định dạng
    SQLite đang lưu trong cột timestamp (dùng dấu cách, có giây - vd
    '2026-08-18 14:44:00'). Nếu không chuẩn hóa, so sánh chuỗi giữa 2
    định dạng khác nhau ('T' vs dấu cách) sẽ cho kết quả SAI (do 'T' có
    mã ASCII lớn hơn dấu cách, làm lệch thứ tự so sánh)."""
    if not value:
        return value
    value = value.replace("T", " ")
    if len(value) == 16:  # "YYYY-MM-DD HH:MM" - thiếu giây
        value += ":00"
    return value


def _time_filter_clause(days_back, start, end):
    """Trả về (sql_clause, params) dùng chung cho các hàm thống kê theo giờ.
    Nếu có start/end (chuỗi ISO datetime, vd '2026-08-18T08:00') thì ưu
    tiên lọc theo khoảng đó; nếu không thì lọc theo N ngày gần nhất
    (days_back) như cũ."""
    if start and end:
        start = _normalize_datetime_local(start)
        end = _normalize_datetime_local(end)
        return "timestamp >= ? AND timestamp <= ?", (start, end)
    return f"timestamp >= datetime('now', '-{int(days_back)} days')", ()


def get_hourly_traffic(days_back=1, start=None, end=None):
    """Lưu lượng xe theo từng giờ - lọc theo N ngày gần nhất (mặc định)
    hoặc theo khoảng thời gian tùy chỉnh (start/end) nếu được truyền vào."""
    clause, params = _time_filter_clause(days_back, start, end)
    with get_connection() as conn:
        rows = conn.execute(
            f"""SELECT strftime('%Y-%m-%d %H:00', timestamp) as hour_bucket,
                       COUNT(*) as total
                FROM traffic_stats
                WHERE {clause}
                GROUP BY hour_bucket
                ORDER BY hour_bucket""",
            params
        ).fetchall()
        return [dict(r) for r in rows]


def get_hourly_traffic_by_class(days_back=1, start=None, end=None):
    """Lưu lượng xe theo giờ, TÁCH theo từng loại xe (car/truck/bus/motorbike)
    - dùng cho biểu đồ tổng hợp (stacked bar theo loại xe) trên trang
    Phân tích. Hỗ trợ lọc theo N ngày gần nhất hoặc khoảng thời gian
    tùy chỉnh (start/end)."""
    clause, params = _time_filter_clause(days_back, start, end)
    with get_connection() as conn:
        rows = conn.execute(
            f"""SELECT strftime('%Y-%m-%d %H:00', timestamp) as hour_bucket,
                       vehicle_class,
                       COUNT(*) as total
                FROM traffic_stats
                WHERE {clause}
                GROUP BY hour_bucket, vehicle_class
                ORDER BY hour_bucket""",
            params
        ).fetchall()
        return [dict(r) for r in rows]


def get_violations_by_hour(days_back=1, start=None, end=None):
    clause, params = _time_filter_clause(days_back, start, end)
    with get_connection() as conn:
        rows = conn.execute(
            f"""SELECT strftime('%Y-%m-%d %H:00', timestamp) as hour_bucket,
                       COUNT(*) as total
                FROM violations
                WHERE {clause}
                GROUP BY hour_bucket
                ORDER BY hour_bucket""",
            params
        ).fetchall()
        return [dict(r) for r in rows]


def get_vehicle_class_distribution():
    with get_connection() as conn:
        rows = conn.execute(
            """SELECT vehicle_class, COUNT(*) as total
               FROM traffic_stats
               GROUP BY vehicle_class
               ORDER BY total DESC"""
        ).fetchall()
        return [dict(r) for r in rows]


def get_behavior_event_summary():
    with get_connection() as conn:
        rows = conn.execute(
            """SELECT event_type, COUNT(*) as total
               FROM behavior_events
               GROUP BY event_type
               ORDER BY total DESC"""
        ).fetchall()
        return [dict(r) for r in rows]


def get_daily_trend(days_back=7):
    """Xu hướng số lượng xe + vi phạm theo từng ngày trong N ngày gần
    nhất - dùng cho biểu đồ so sánh xu hướng."""
    with get_connection() as conn:
        traffic_rows = conn.execute(
            f"""SELECT date(timestamp) as day, COUNT(*) as total
                FROM traffic_stats
                WHERE timestamp >= datetime('now', '-{int(days_back)} days')
                GROUP BY day ORDER BY day"""
        ).fetchall()
        violation_rows = conn.execute(
            f"""SELECT date(timestamp) as day, COUNT(*) as total
                FROM violations
                WHERE timestamp >= datetime('now', '-{int(days_back)} days')
                GROUP BY day ORDER BY day"""
        ).fetchall()
        return {
            "traffic": [dict(r) for r in traffic_rows],
            "violations": [dict(r) for r in violation_rows],
        }


def get_recent_violations(limit=50):
    with get_connection() as conn:
        rows = conn.execute(
            "SELECT * FROM violations ORDER BY timestamp DESC LIMIT ?",
            (limit,)
        ).fetchall()
        results = []
        for r in rows:
            d = dict(r)
            tid = d.get("track_id")
            # Lớp an toàn: xử lý cả những bản ghi cũ đã lỡ lưu track_id
            # dạng bytes trước khi có bản sửa này
            if isinstance(tid, bytes):
                try:
                    d["track_id"] = int.from_bytes(tid, byteorder="little")
                except Exception:
                    d["track_id"] = tid.decode("utf-8", errors="ignore")
            results.append(d)
        return results


def get_violation_summary():
    with get_connection() as conn:
        rows = conn.execute(
            """SELECT violation_type, COUNT(*) as total
               FROM violations GROUP BY violation_type"""
        ).fetchall()
        return [dict(r) for r in rows]


# --- Dùng cho chatbot Text-to-SQL ---

FORBIDDEN_KEYWORDS = ["DROP", "DELETE", "UPDATE", "INSERT", "ALTER",
                       "TRUNCATE", "ATTACH", "PRAGMA", "REPLACE"]


def is_safe_select_query(sql: str) -> bool:
    """Chỉ cho phép câu SELECT thuần, chặn mọi từ khóa có thể sửa/xóa
    dữ liệu hoặc truy cập file hệ thống."""
    sql_upper = sql.strip().upper()
    if not sql_upper.startswith("SELECT"):
        return False
    if any(word in sql_upper for word in FORBIDDEN_KEYWORDS):
        return False
    if ";" in sql.strip()[:-1]:  # chặn nhiều câu lệnh nối bằng ";"
        return False
    return True


def run_safe_query(sql: str, max_rows: int = 200):
    """Chạy câu SQL do chatbot sinh ra, chỉ sau khi đã qua kiểm tra an toàn.
    Dùng kết nối read-only để phòng ngừa thêm một lớp nữa."""
    if not is_safe_select_query(sql):
        raise ValueError("Câu truy vấn không an toàn hoặc không phải SELECT.")

    uri = f"file:{DB_PATH}?mode=ro"
    conn = sqlite3.connect(uri, uri=True, timeout=10)
    conn.row_factory = sqlite3.Row
    try:
        cur = conn.execute(sql)
        rows = cur.fetchmany(max_rows)
        return [dict(r) for r in rows]
    finally:
        conn.close()
# --- Lưu trữ theo từng lần chạy (archive) ---

import json

ARCHIVE_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "archives")

ARCHIVE_TABLES = ["violations", "traffic_stats", "roi_occupancy", "behavior_events"]


def archive_and_clear_run():
    """Lưu toàn bộ dữ liệu hiện có (vi phạm, thống kê xe qua line, mật độ
    ROI, hành vi bất thường) ra 1 file JSON trong thư mục archives/, đặt
    tên theo thời điểm lưu. Sau đó xóa sạch dữ liệu trong các bảng để lần
    chạy tiếp theo bắt đầu từ đầu. Gọi hàm này ở đầu mỗi lần bắt đầu chạy
    (trước khi xử lý video mới).

    Nếu tất cả các bảng đều trống thì không tạo file (tránh archive rỗng
    khi app chưa từng chạy lần nào)."""
    os.makedirs(ARCHIVE_DIR, exist_ok=True)

    snapshot = {}
    with get_connection() as conn:
        for table in ARCHIVE_TABLES:
            rows = conn.execute(f"SELECT * FROM {table}").fetchall()
            snapshot[table] = [dict(r) for r in rows]

    total_rows = sum(len(v) for v in snapshot.values())
    if total_rows == 0:
        return None  # không có gì để lưu

    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    archive_path = os.path.join(ARCHIVE_DIR, f"run_{timestamp_str}.json")

    with open(archive_path, "w", encoding="utf-8") as f:
        json.dump(snapshot, f, ensure_ascii=False, indent=2, default=str)

    with get_connection() as conn:
        for table in ARCHIVE_TABLES:
            conn.execute(f"DELETE FROM {table}")
            conn.execute("DELETE FROM sqlite_sequence WHERE name = ?", (table,))
        conn.commit()

    print(f"[Archive] Đã lưu {total_rows} bản ghi vào {archive_path}, đã xóa dữ liệu cũ.")
    return archive_path


def list_archives():
    """Liệt kê các file archive đã lưu, mới nhất trước."""
    os.makedirs(ARCHIVE_DIR, exist_ok=True)
    files = sorted(os.listdir(ARCHIVE_DIR), reverse=True)
    return [f for f in files if f.endswith(".json")]


def load_archive(filename):
    """Đọc lại nội dung 1 file archive theo tên (lấy từ list_archives())."""
    path = os.path.join(ARCHIVE_DIR, filename)
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
