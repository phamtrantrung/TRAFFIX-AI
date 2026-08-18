# GLOBAL VARIABLES
from collections import defaultdict
import threading

# VIDEO
video_path = None

# RTSP
rtsp_url = ""

source_type = "video"

# LINE
lines = []

# ROI
roi_zones = []

# CONTROL
paused = False
stopped = False

# ROI COUNT
roi_counts = defaultdict(set)

# VIOLATION
violated_ids = set()

# DEVICE
device_mode = "cpu"

# THREAD LOCK - bảo vệ các biến dùng chung giữa thread xử lý video
# và thread phục vụ web (Flask). Dùng khi đọc/ghi roi_counts, violated_ids.
state_lock = threading.Lock()


def reset_state():
    """Reset toàn bộ trạng thái khi bắt đầu phiên xử lý video mới."""
    global roi_counts, violated_ids, stopped, paused
    with state_lock:
        roi_counts = defaultdict(set)
        violated_ids = set()
        stopped = False
        paused = False
