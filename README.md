# Hệ thống Giám sát & Phạt nguội Giao thông Thông minh

Hệ thống video analytics: đếm xe, phát hiện vi phạm (vượt đèn đỏ, đỗ sai
quy định), phát hiện hành vi bất thường (lạng lách, dừng đột ngột, đi
ngược chiều), tự động đọc biển số, dashboard web thời gian thực với
biểu đồ phân tích xu hướng, và trợ lý AI hỏi-đáp dữ liệu bằng ngôn ngữ
tự nhiên (có kèm biểu đồ khi phù hợp).

## 1. Cài đặt

```bash
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Linux/Mac

pip install -r requirements.txt
```

Nếu máy không có GPU (như log lỗi CUDA trước đó), hệ thống mặc định
`device_mode = "cpu"` trong `globals.py` — không cần chỉnh gì thêm,
nhưng tốc độ xử lý sẽ chậm hơn.

## 2. Chuẩn bị model

- `models/yolov8n.pt` — model detect xe. Có thể dùng model YOLO đã
  train sẵn (COCO) hoặc model custom bạn đã có từ trước.
- `models/plate_yolov8n.pt` — model detect biển số xe. **Cần train
  riêng** hoặc tải model có sẵn cho biển số Việt Nam (tìm trên
  Roboflow Universe với từ khóa "vietnamese license plate").
  Nếu chưa có model này, chạy với `enable_plate_reading=False` để hệ
  thống vẫn hoạt động (bỏ qua bước đọc biển số).

## 3. Cấu hình API key cho Chatbot

```bash
# Windows PowerShell
$env:ANTHROPIC_API_KEY="sk-ant-..."

# Linux/Mac
export ANTHROPIC_API_KEY="sk-ant-..."
```

## 4. Chạy thử nhanh (không qua web, chỉ xử lý video)

```bash
python main.py
```
Nhấn `q` để dừng. Chỉnh `G.video_path` trong `globals.py` hoặc sửa
biến `source` trong `main.py` để trỏ tới file video của bạn.

## 5. Chạy đầy đủ qua web dashboard

```bash
python app.py
```
Mở trình duyệt tại `http://localhost:5000`

- Trang **Dashboard**: xem video trực tiếp, thống kê ROI, danh sách vi phạm
- Trang **Phân tích** (`/analytics`): biểu đồ xu hướng lưu lượng theo giờ, vi phạm theo giờ, phân bố loại phương tiện, hành vi bất thường, xu hướng 7 ngày
- Trang **Trợ lý AI** (`/chatbot`): hỏi đáp dữ liệu bằng tiếng Việt, tự động vẽ biểu đồ khi kết quả phù hợp

## 6. Cấu trúc thư mục

```
project/
├── app.py                     # Flask web server
├── main.py                    # Vòng lặp xử lý video chính
├── globals.py                 # Biến trạng thái toàn cục
├── utils/
│   ├── geometry.py            # point_in_roi, side_of_line, ...
│   ├── drawing.py             # Vẽ overlay lên frame
│   ├── license_plate.py       # Detect + OCR biển số
│   ├── violation_engine.py    # Logic phát hiện vi phạm
│   ├── behavior_analysis.py   # Phát hiện hành vi bất thường
│   ├── database.py            # SQLite: violations, traffic_stats, behavior_events
│   └── chatbot_engine.py      # Text-to-SQL chatbot + chart data
├── templates/
│   ├── dashboard.html
│   ├── analytics.html         # Trang biểu đồ phân tích
│   └── chatbot.html
├── static/style.css
├── models/                    # Đặt file .pt model ở đây
├── data/traffic.db            # Tự tạo khi chạy lần đầu
└── evidence/                  # Ảnh bằng chứng vi phạm tự động lưu
```

## 7. Việc bạn cần tự làm thêm

1. **Train/tải model detect biển số** — đây là phần duy nhất cần dữ
   liệu/huấn luyện riêng, không thể đóng gói sẵn.
2. **Xác định trạng thái đèn giao thông** (`traffic_light_state` trong
   `main.py`) — hiện đang mặc định `"green"`. Cần thêm logic thực tế
   (đọc từ cảm biến, hoặc phân tích màu vùng đèn tín hiệu bằng CV).
3. **Điều chỉnh `VEHICLE_CLASSES`** trong `main.py` theo đúng class ID
   của model bạn đang dùng (COCO id khác với model custom).
4. **Test và đo đạc số liệu thực tế** (FPS, độ chính xác OCR biển số,
   tỷ lệ phát hiện vi phạm đúng/sai) để đưa vào báo cáo — đây là phần
   quan trọng nhất cho luận văn, không thể giả lập được.
