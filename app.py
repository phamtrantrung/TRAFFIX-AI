# FLASK WEB APPLICATION
import os
import time
import threading
import cv2
from werkzeug.utils import secure_filename
from flask import Flask, render_template, Response, request, jsonify, send_from_directory

import globals as G
from utils import database
from utils import chatbot_engine
import main as video_main

app = Flask(__name__)
database.init_db()

UPLOAD_DIR = "uploads"
ALLOWED_VIDEO_EXT = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
os.makedirs(UPLOAD_DIR, exist_ok=True)

_video_thread = None
_latest_frame = None
_frame_lock = threading.Lock()


def _video_worker(source, enable_plate_reading):
    global _latest_frame
    for frame in video_main.run(source, enable_plate_reading=enable_plate_reading):
        with _frame_lock:
            _latest_frame = frame.copy()


@app.route("/")
def index():
    return render_template("dashboard.html")


@app.route("/chatbot")
def chatbot_page():
    return render_template("chatbot.html")


@app.route("/analytics")
def analytics_page():
    return render_template("analytics.html")


@app.route("/api/upload_video", methods=["POST"])
def upload_video():
    if "video" not in request.files:
        return jsonify({"status": "error", "message": "Không có file được gửi lên"}), 400

    f = request.files["video"]
    if f.filename == "":
        return jsonify({"status": "error", "message": "Chưa chọn file"}), 400

    ext = os.path.splitext(f.filename)[1].lower()
    if ext not in ALLOWED_VIDEO_EXT:
        return jsonify({
            "status": "error",
            "message": f"Định dạng {ext} không được hỗ trợ. Chỉ nhận: {', '.join(sorted(ALLOWED_VIDEO_EXT))}"
        }), 400

    filename = secure_filename(f.filename)
    save_path = os.path.join(UPLOAD_DIR, filename)
    f.save(save_path)

    # Lay 1 khung hinh dau tien de hien thi len canvas cho nguoi dung ve line/ROI
    cap = cv2.VideoCapture(save_path)
    ok, frame = cap.read()
    cap.release()
    if ok:
        cv2.imwrite(os.path.join(UPLOAD_DIR, "preview_frame.jpg"), frame)

    G.video_path = save_path
    return jsonify({"status": "ok", "path": save_path, "filename": filename})


@app.route("/api/preview_frame")
def preview_frame():
    preview_path = os.path.join(UPLOAD_DIR, "preview_frame.jpg")
    if not os.path.exists(preview_path):
        return jsonify({"status": "error", "message": "Chua co khung hinh xem truoc"}), 404
    return send_from_directory(UPLOAD_DIR, "preview_frame.jpg")


@app.route("/api/start", methods=["POST"])
def start_video():
    global _video_thread
    # get_json(silent=True) thay vi request.json: tranh Flask tu dong
    # nem loi 400 khi request khong co header Content-Type: application/json
    # hoac khong co body - truong hop nay se tra ve None, roi "or {}" xu ly
    # thanh dict rong, dung gia tri mac dinh o duoi.
    data = request.get_json(silent=True) or {}
    source = data.get("source", G.video_path or 0)
    enable_plate = data.get("enable_plate_reading", True)

    if _video_thread is not None and _video_thread.is_alive():
        return jsonify({"status": "error", "message": "Video dang chay roi"}), 400

    G.reset_state()
    _video_thread = threading.Thread(
        target=_video_worker, args=(source, enable_plate), daemon=True
    )
    _video_thread.start()
    return jsonify({"status": "ok"})


@app.route("/api/stop", methods=["POST"])
def stop_video():
    G.stopped = True
    return jsonify({"status": "ok"})


@app.route("/api/pause", methods=["POST"])
def pause_video():
    G.paused = not G.paused
    return jsonify({"status": "ok", "paused": G.paused})


@app.route("/api/lines", methods=["GET", "POST"])
def manage_lines():
    if request.method == "POST":
        G.lines = request.json.get("lines", [])
        return jsonify({"status": "ok"})
    return jsonify({"lines": G.lines})


@app.route("/api/roi", methods=["GET", "POST"])
def manage_roi():
    if request.method == "POST":
        G.roi_zones = request.json.get("roi_zones", [])
        return jsonify({"status": "ok"})
    return jsonify({"roi_zones": G.roi_zones})

@app.route("/api/status")
def api_status():
    is_running = _video_thread is not None and _video_thread.is_alive()
    return jsonify({
        "running": is_running,
        "paused": G.paused,
        "video_path": G.video_path,
    })


def _mjpeg_generator():
    # FIX: vong lap goc dung "while True: continue" khong sleep khi chua co
    # frame -> chiem 100% CPU/GIL cua 1 thread, neu mo nhieu ket noi /video_feed
    # cung luc (vd img load + onerror fallback + F5 lai trang) se lam nghen
    # het cac request Flask khac (timeout toan bo, du server khong crash).
    # Them time.sleep() de nhuong CPU/GIL cho cac thread khac.
    while True:
        with _frame_lock:
            frame = None if _latest_frame is None else _latest_frame.copy()

        if frame is None:
            time.sleep(0.03)  # chua co frame -> nghi ~30ms roi kiem tra lai
            continue

        ret, buffer = cv2.imencode(".jpg", frame)
        if not ret:
            time.sleep(0.03)
            continue

        frame_bytes = buffer.tobytes()
        yield (b"--frame\r\n"
               b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n")

        time.sleep(0.03)  # gioi han toc do gui frame (~30fps), tranh xa CPU lien tuc


@app.route("/video_feed")
def video_feed():
    return Response(_mjpeg_generator(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/api/violations")
def api_violations():
    limit = int(request.args.get("limit", 50))
    return jsonify(database.get_recent_violations(limit))


@app.route("/api/violations/summary")
def api_violation_summary():
    return jsonify(database.get_violation_summary())


@app.route("/api/behavior_events")
def api_behavior_events():
    limit = int(request.args.get("limit", 50))
    return jsonify(database.get_recent_behavior_events(limit))


@app.route("/api/analytics/hourly_traffic")
def api_hourly_traffic():
    days_back = int(request.args.get("days", 1))
    return jsonify(database.get_hourly_traffic(days_back))


@app.route("/api/analytics/violations_by_hour")
def api_violations_by_hour():
    days_back = int(request.args.get("days", 1))
    return jsonify(database.get_violations_by_hour(days_back))


@app.route("/api/analytics/vehicle_distribution")
def api_vehicle_distribution():
    return jsonify(database.get_vehicle_class_distribution())


@app.route("/api/analytics/behavior_summary")
def api_behavior_summary():
    return jsonify(database.get_behavior_event_summary())


@app.route("/api/analytics/daily_trend")
def api_daily_trend():
    days_back = int(request.args.get("days", 7))
    return jsonify(database.get_daily_trend(days_back))


@app.route("/api/analytics/combined")
def api_analytics_combined():
    days_back = int(request.args.get("days", 1))
    # Khoảng thời gian tùy chỉnh (tùy chọn): ?start=2026-08-18T08:00&end=2026-08-18T18:00
    # Nếu có cả start và end thì ưu tiên dùng, bỏ qua "days".
    start = request.args.get("start")
    end = request.args.get("end")

    traffic_by_class = database.get_hourly_traffic_by_class(days_back, start=start, end=end)
    violations = database.get_violations_by_hour(days_back, start=start, end=end)
    return jsonify({
        "traffic_by_class": traffic_by_class,   # [{hour_bucket, vehicle_class, total}, ...]
        "violations": violations                # [{hour_bucket, total}, ...]
    })


@app.route("/api/roi_counts")
def api_roi_counts():
    with G.state_lock:
        return jsonify({str(k): len(v) for k, v in G.roi_counts.items()})


@app.route("/evidence/<path:filename>")
def evidence_file(filename):
    return send_from_directory("evidence", filename)


@app.route("/api/chatbot", methods=["POST"])
def api_chatbot():
    question = (request.json or {}).get("question", "").strip()
    if not question:
        return jsonify({"answer": "Bạn chưa nhập câu hỏi.", "error": True}), 400

    result = chatbot_engine.ask(question)
    return jsonify(result)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True, threaded=True)
