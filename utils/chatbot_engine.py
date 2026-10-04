# AI CHATBOT ENGINE (Text-to-SQL) - Dùng Google Gemini
#
# Yêu cầu: pip install google-generativeai
# Cần biến môi trường GOOGLE_API_KEY đã được set.

import os
import re
from datetime import datetime, timedelta

import google.generativeai as genai

from utils import database

MODEL_NAME = "gemini-3.5-flash-lite"   # Có thể đổi thành gemini thích hợp

_model = None

# Từ khóa nhận diện người dùng đang muốn XEM BIỂU ĐỒ (không phải chỉ hỏi
# số liệu bằng chữ). Dùng để nhắc Gemini chủ động sinh SQL dạng nhóm theo
# danh mục (GROUP BY) thay vì trả về 1 con số tổng duy nhất - vì có nhóm
# thì mới có gì để vẽ.
CHART_INTENT_KEYWORDS = ["biểu đồ", "vẽ", "chart", "trực quan", "đồ thị"]

# Bảng màu cố định cho các loại phương tiện, để biểu đồ nhất quán màu sắc
# giữa các lần hỏi khác nhau (khớp với màu dùng ở trang Phân tích).
VEHICLE_COLOR_MAP = {
    "car": "#2dd4bf", "ô tô": "#2dd4bf", "o to": "#2dd4bf",
    "motorbike": "#a78bfa", "xe máy": "#a78bfa", "xe may": "#a78bfa",
    "bus": "#38bdf8", "xe buýt": "#38bdf8", "xe buyt": "#38bdf8",
    "truck": "#fbbf24", "xe tải": "#fbbf24", "xe tai": "#fbbf24",
}
FALLBACK_PALETTE = ["#4fd1c5", "#f472b6", "#fb923c", "#60a5fa", "#c084fc", "#facc15"]


def _get_model():
    global _model
    if _model is None:
        api_key = os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            raise RuntimeError(
                "Chưa thiết lập biến môi trường GOOGLE_API_KEY."
            )
        genai.configure(api_key=api_key)
        _model = genai.GenerativeModel(MODEL_NAME)
    return _model


def _extract_sql(text):
    """Trích xuất câu SQL từ phản hồi của LLM."""
    code_block = re.search(r"```(?:sql)?\s*(.*?)```", text, re.DOTALL | re.IGNORECASE)
    if code_block:
        return code_block.group(1).strip()

    select_match = re.search(r"(SELECT.*)", text, re.DOTALL | re.IGNORECASE)
    if select_match:
        return select_match.group(1).strip().rstrip(";")

    return text.strip()


def _wants_chart(question: str) -> bool:
    q = question.lower()
    return any(kw in q for kw in CHART_INTENT_KEYWORDS)


def natural_language_to_sql(question: str) -> str:
    model = _get_model()

    # FIX (chatbot trả lời "không có xe nào" dù dữ liệu có thật): hàm
    # date('now') của SQLite trả về giờ UTC, trong khi timestamp trong
    # database được lưu theo GIỜ ĐỊA PHƯƠNG (Việt Nam, UTC+7). Trong khoảng
    # 0h-7h sáng giờ VN, ngày UTC vẫn còn là ngày hôm trước nên điều kiện
    # "hôm nay" bị lệch một ngày -> truy vấn không khớp dòng nào. Cách sửa:
    # truyền thẳng thời điểm hiện tại theo giờ địa phương của máy chủ vào
    # prompt, để Gemini dùng ngày cụ thể thay vì tự gọi date('now').
    now = datetime.now()
    now_local = now.strftime("%Y-%m-%d %H:%M:%S")
    today_local = now.strftime("%Y-%m-%d")
    yesterday_local = (now - timedelta(days=1)).strftime("%Y-%m-%d")

    chart_hint = ""
    if _wants_chart(question):
        # Người dùng có ý muốn xem biểu đồ (chữ "biểu đồ"/"vẽ"/...) nhưng
        # không phải lúc nào cũng tự nêu rõ nên nhóm theo tiêu chí gì.
        # Chủ động nhắc Gemini: PHẢI trả về dạng nhóm theo danh mục (2 cột:
        # tên nhóm + số lượng, từ 2 dòng trở lên) thay vì 1 con số tổng -
        # có nhóm thì hệ thống mới vẽ được biểu đồ.
        chart_hint = """
NGƯỜI DÙNG ĐANG MUỐN XEM BIỂU ĐỒ. Vì vậy câu SQL BẮT BUỘC phải trả về
kết quả dạng NHÓM (GROUP BY) với đúng 2 cột: một cột tên nhóm (ví dụ
vehicle_class, hoặc khung giờ) và một cột số lượng (COUNT/SUM), trả về
từ 2 dòng trở lên. TUYỆT ĐỐI KHÔNG trả về một con số tổng duy nhất
(như SELECT COUNT(*) không GROUP BY) vì sẽ không có gì để vẽ.
Nếu câu hỏi không nói rõ nhóm theo gì, mặc định nhóm theo loại phương
tiện (vehicle_class) trong phạm vi thời gian gần nhất hợp lý (ví dụ
hôm nay) và giới hạn hợp lý (ví dụ theo khung giờ nếu có nhiều dữ liệu).
"""

    system_prompt = f"""Bạn là chuyên gia SQL. Nhiệm vụ: chuyển câu hỏi
tiếng Việt của người dùng thành MỘT câu lệnh SQLite SELECT duy nhất,
dựa trên schema database sau:

{database.SCHEMA_DESCRIPTION}

THÔNG TIN THỜI GIAN (giờ địa phương Việt Nam):
- Bây giờ là: {now_local}
- "Hôm nay" là ngày {today_local}
- "Hôm qua" là ngày {yesterday_local}
- Mọi timestamp trong database được lưu theo GIỜ ĐỊA PHƯƠNG, KHÔNG phải UTC.
{chart_hint}
QUY TẮC BẮT BUỘC:
- CHỈ trả về câu SQL, không giải thích, không markdown, không dấu ```
- CHỈ được dùng SELECT, tuyệt đối không UPDATE/DELETE/INSERT/DROP
- Với "hôm nay", "hôm qua": dùng đúng ngày đã cho ở trên (ví dụ
  date(timestamp) = '{today_local}'). TUYỆT ĐỐI KHÔNG dùng date('now')
  hoặc datetime('now') trần vì SQLite trả về giờ UTC, sẽ bị lệch ngày.
- Với "tuần này", "tuần trước" hoặc các mốc tương đối khác: tính từ ngày
  {today_local}, và nếu buộc phải dùng hàm ngày giờ của SQLite thì luôn
  kèm 'localtime', ví dụ date('now','localtime','-7 days').
- Khi người dùng nêu khoảng giờ cụ thể (ví dụ "từ 6h20 đến 6h25 hôm nay"):
  so sánh trực tiếp timestamp với chuỗi ngày-giờ đầy đủ, ví dụ
  timestamp >= '{today_local} 06:20:00' AND timestamp < '{today_local} 06:25:00'
  (hoặc <= nếu người dùng muốn gồm cả mốc cuối).
- Luôn thêm LIMIT 200 nếu câu hỏi có thể trả về nhiều dòng
"""

    full_prompt = system_prompt + "\n\nCâu hỏi của người dùng: " + question
    response = model.generate_content(full_prompt)
    return _extract_sql(response.text)


def explain_result(question: str, sql: str, result_rows: list) -> str:
    model = _get_model()

    system_prompt = """Bạn là trợ lý phân tích dữ liệu giao thông.
Nhiệm vụ: dựa vào câu hỏi gốc và kết quả truy vấn database, trả lời
bằng tiếng Việt, ngắn gọn, tự nhiên, dễ hiểu cho người không rành kỹ
thuật. Không nhắc đến SQL hay tên bảng/cột trong câu trả lời."""

    user_content = f"""Câu hỏi: {question}
Kết quả truy vấn (JSON): {result_rows}

Trả lời ngắn gọn bằng tiếng Việt."""

    full_prompt = system_prompt + "\n\n" + user_content
    response = model.generate_content(full_prompt)
    return response.text


_TIME_LABEL_RE = re.compile(r"^\d{1,2}([:h]\d{2})|^\d{4}-\d{2}-\d{2}|^\d{1,2}h$")


def _pick_color(label: str, idx: int) -> str:
    key = label.strip().lower()
    if key in VEHICLE_COLOR_MAP:
        return VEHICLE_COLOR_MAP[key]
    return FALLBACK_PALETTE[idx % len(FALLBACK_PALETTE)]


def _try_build_chart_data(rows: list, question: str = ""):
    """Tự động dựng dữ liệu biểu đồ nếu phù hợp - chọn loại chart (line
    cho dữ liệu theo thời gian, bar cho dữ liệu theo danh mục), gán màu
    theo từng loại phương tiện, kèm tiêu đề để hiển thị phía frontend."""
    if not rows or len(rows) < 2:
        return None

    keys = list(rows[0].keys())
    if len(keys) != 2:
        return None

    label_key, value_key = keys[0], keys[1]

    try:
        values = [float(r[value_key]) for r in rows]
    except (ValueError, TypeError):
        return None

    labels = [str(r[label_key]) for r in rows]

    # Nếu nhãn trông giống mốc thời gian (giờ/ngày) -> dùng line chart cho
    # dễ nhìn xu hướng; ngược lại (tên danh mục như loại xe) -> bar chart.
    looks_like_time = any(_TIME_LABEL_RE.match(l) for l in labels[:3])
    chart_type = "line" if looks_like_time else "bar"

    colors = [_pick_color(l, i) for i, l in enumerate(labels)]

    title = f"{value_key.replace('_', ' ').capitalize()} theo {label_key.replace('_', ' ')}"

    return {
        "type": chart_type,
        "labels": labels[:30],
        "values": values[:30],
        "colors": colors[:30],
        "label": value_key,
        "title": title,
    }


def ask(question: str) -> dict:
    """Hàm chính - gọi từ Flask route /api/chatbot."""
    try:
        sql = natural_language_to_sql(question)
    except Exception as e:
        return {"answer": f"Xin lỗi, có lỗi khi xử lý câu hỏi: {e}", "sql": None, "error": True}

    # In câu SQL ra terminal để kiểm tra/debug (có thể xóa dòng này sau khi
    # đã xác nhận chatbot chạy ổn định).
    print("[Chatbot SQL]", sql)

    if not database.is_safe_select_query(sql):
        return {
            "answer": "Xin lỗi, mình chỉ có thể trả lời các câu hỏi tra cứu dữ liệu (không thể sửa/xóa dữ liệu).",
            "sql": sql,
            "error": True,
        }

    try:
        rows = database.run_safe_query(sql)
    except Exception as e:
        return {"answer": f"Không thể truy vấn dữ liệu: {e}", "sql": sql, "error": True}

    try:
        answer = explain_result(question, sql, rows)
    except Exception as e:
        answer = f"Đã lấy được {len(rows)} kết quả nhưng không thể diễn giải: {e}"

    chart_data = _try_build_chart_data(rows, question)

    # Nếu người dùng rõ ràng muốn xem biểu đồ nhưng dữ liệu trả về không
    # đủ điều kiện để vẽ (ví dụ chỉ có 1 dòng/1 loại xe duy nhất trong
    # khung thời gian đó), nói rõ lý do thay vì im lặng không vẽ gì cả -
    # tránh gây cảm giác "chatbot không hiểu yêu cầu".
    if _wants_chart(question) and chart_data is None:
        answer += (
            " (Không đủ dữ liệu đa dạng để vẽ biểu đồ trong khoảng này - "
            "cần ít nhất 2 nhóm khác nhau, ví dụ 2 loại xe trở lên.)"
        )

    return {"answer": answer, "sql": sql, "raw_rows": rows, "chart_data": chart_data, "error": False}
