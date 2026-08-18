# AI CHATBOT ENGINE (Text-to-SQL) - Dùng Google Gemini
#
# Yêu cầu: pip install google-generativeai
# Cần biến môi trường GOOGLE_API_KEY đã được set.

import os
import re
import google.generativeai as genai

from utils import database

MODEL_NAME = "gemini-3.5-flash-lite"   # Có thể đổi thành gemini thích hợp

_model = None


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


def natural_language_to_sql(question: str) -> str:
    model = _get_model()

    system_prompt = f"""Bạn là chuyên gia SQL. Nhiệm vụ: chuyển câu hỏi
tiếng Việt của người dùng thành MỘT câu lệnh SQLite SELECT duy nhất,
dựa trên schema database sau:

{database.SCHEMA_DESCRIPTION}

QUY TẮC BẮT BUỘC:
- CHỈ trả về câu SQL, không giải thích, không markdown, không dấu ```
- CHỈ được dùng SELECT, tuyệt đối không UPDATE/DELETE/INSERT/DROP
- Nếu câu hỏi liên quan đến "hôm nay", "hôm qua", "tuần này" thì dùng
  hàm ngày giờ chuẩn SQLite: date('now'), date('now','-1 day'),
  date('now','weekday 0','-7 days') v.v.
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


def _try_build_chart_data(rows: list):
    """Tự động dựng dữ liệu biểu đồ nếu phù hợp."""
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

    return {
        "type": "bar",
        "labels": labels[:30],
        "values": values[:30],
        "label": value_key,
    }


def ask(question: str) -> dict:
    """Hàm chính - gọi từ Flask route /api/chatbot."""
    try:
        sql = natural_language_to_sql(question)
    except Exception as e:
        return {"answer": f"Xin lỗi, có lỗi khi xử lý câu hỏi: {e}", "sql": None, "error": True}

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

    chart_data = _try_build_chart_data(rows)

    return {"answer": answer, "sql": sql, "raw_rows": rows, "chart_data": chart_data, "error": False}