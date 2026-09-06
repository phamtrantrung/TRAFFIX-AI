# TRAFFIX-AI — Hệ thống Giám sát và Phát hiện Vi phạm Giao thông Thông minh

Hệ thống video analytics giám sát giao thông theo thời gian thực: đếm
lưu lượng xe theo line, phát hiện xe đi vào vùng cấm (ROI), phân tích
lưu lượng/vi phạm trực quan theo thời gian, và trợ lý AI Chatbot
(Google Gemini) cho phép tra cứu dữ liệu bằng tiếng Việt tự nhiên.

Dự án hướng tới đối tượng phục vụ là **cơ quan/đơn vị quy mô nhỏ** và
**các đồ án/dự án tốt nghiệp** cần một giải pháp giám sát giao thông
chi phí thấp, dễ triển khai (chạy tốt trên CPU phổ thông, không cần
GPU chuyên dụng), dễ tùy biến cấu hình line/ROI theo từng vị trí thực
tế.

## 1. Tính năng đã hoàn thành

- **Đếm xe theo line**: phát hiện + phân loại phương tiện (ô tô,
  xe máy, xe tải, xe buýt) bằng YOLOv8, theo dõi bằng ByteTrack, đếm
  khi xe cắt qua line cấu hình (vẽ trực tiếp trên canvas), tách theo
  hướng di chuyển.
- **Xác định xe vào vùng vi phạm (ROI)**: vẽ vùng cấm tự do bằng
  chuột (freehand, không giới hạn 4 điểm), ghi nhận vi phạm ngay khi
  xe đi vào vùng, lưu kèm ảnh bằng chứng. Line (đếm lưu lượng) và ROI
  (ghi vi phạm) hoạt động **hoàn toàn độc lập** với nhau theo thiết kế.
- **Phân tích lưu lượng**: trang Phân tích với biểu đồ kết hợp (lưu
  lượng theo loại xe + số vi phạm theo giờ), 4 thẻ thống kê nhanh,
  bảng chi tiết, bộ lọc thời gian (24h/3 ngày/7 ngày/tùy chỉnh). Có
  thể xuất báo cáo **Word (.docx) và Excel (.xlsx)** tự động, tích hợp
  sẵn phần trả lời của Trợ lý AI giới hạn theo đúng khoảng thời gian
  đang xem.
- **Trợ lý AI Chatbot**: dùng Google Gemini, chuyển câu hỏi tiếng Việt
  tự nhiên thành câu lệnh SQL (chỉ cho phép `SELECT` thuần, chặn mọi
  từ khóa có thể sửa/xóa dữ liệu) để tra cứu dữ liệu giám sát.

## 2. Cài đặt

```bash
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Linux/Mac

pip install -r requirements.txt
```

Nếu máy không có GPU, hệ thống mặc định `device_mode = "cpu"` trong
`globals.py` — không cần chỉnh gì thêm, tốc độ xử lý sẽ chậm hơn.

## 3. Chuẩn bị model

- `models/yolov8s.pt` — model detect xe. Nếu chưa có, Ultralytics sẽ
  tự động tải về (~22MB) trong lần chạy đầu tiên.
- `models/plate_yolov8n.pt` — model detect biển số xe. **Chưa hoàn
  thiện trong phiên bản hiện tại** (cần train riêng hoặc tải model có
  sẵn cho biển số Việt Nam, ví dụ tìm trên Roboflow Universe với từ
  khóa "vietnamese license plate"). Nếu chưa có model này, chạy với
  `enable_plate_reading=False` để hệ thống vẫn hoạt động bình thường
  (bỏ qua bước đọc biển số).

## 4. Cấu hình API key cho Chatbot (Google Gemini)

Tạo file `.env` ở thư mục gốc dự án:

```
GOOGLE_API_KEY=your_gemini_api_key_here
```

`utils/chatbot_engine.py` dùng `python-dotenv` để tự động load biến
môi trường này khi khởi động.

## 5. Chạy thử nhanh (không qua web, chỉ xử lý video)

```bash
python main.py
```

Nhấn `q` để dừng. Chỉnh `G.video_path` trong `globals.py` hoặc sửa
biến `source` trong `main.py` để trỏ tới file video của bạn.

## 6. Chạy đầy đủ qua web dashboard

```bash
python app.py
```

Mở trình duyệt tại `http://localhost:5000`

- **Dashboard**: xem video trực tiếp, cấu hình line/ROI trên canvas,
  thống kê số xe theo ROI, danh sách vi phạm gần nhất.
- **Phân tích** (`/analytics`): biểu đồ xu hướng lưu lượng/vi phạm
  theo giờ, bộ lọc thời gian, xuất báo cáo Word/Excel.
- **Trợ lý AI** (`/chatbot`): hỏi đáp dữ liệu giám sát bằng tiếng
  Việt tự nhiên.

## 7. Cấu trúc thư mục

```
project/
├── app.py                     # Flask web server
├── main.py                    # Vòng lặp xử lý video chính (YOLO + ByteTrack)
├── globals.py                 # Biến trạng thái toàn cục (lines, roi_zones...)
├── custom_bytetrack.yaml      # Cấu hình tracker riêng (track_buffer, ngưỡng...)
├── utils/
│   ├── geometry.py            # point_in_roi, side_of_line, point_projects_onto_segment
│   ├── drawing.py             # Vẽ overlay line/ROI/thống kê lên frame
│   ├── license_plate.py       # Detect + OCR biển số (đang phát triển)
│   ├── violation_engine.py    # Logic ghi nhận vi phạm vùng cấm (ROI)
│   ├── behavior_analysis.py   # Phát hiện hành vi bất thường
│   ├── database.py            # SQLite: violations, traffic_stats, roi_occupancy, behavior_events
│   ├── chatbot_engine.py      # Trợ lý AI Text-to-SQL (Google Gemini)
│   └── report_generator.py    # Xuất báo cáo phân tích Word/Excel tích hợp AI
├── templates/
│   ├── dashboard.html
│   ├── analytics.html         # Trang biểu đồ phân tích + xuất báo cáo
│   └── chatbot.html
├── static/style.css
├── models/                    # Đặt file .pt model ở đây (không đưa lên git)
├── data/traffic.db            # Tự tạo khi chạy lần đầu (không đưa lên git)
├── archives/                  # Lưu dữ liệu mỗi lần chạy trước khi xóa để chạy mới
├── reports/                   # Báo cáo Word/Excel xuất ra (không đưa lên git)
├── evidence/                  # Ảnh bằng chứng vi phạm tự động lưu (không đưa lên git)
│
├── extract_frames.py          # Trích frame từ video để làm dữ liệu train
├── auto_label.py              # Tự gán nhãn sơ bộ bằng yolov8s (cần soát lại thủ công)
├── data.yaml                  # Cấu hình dataset cho train.py
├── train.py                   # Fine-tune model riêng cho đặc điểm xe Việt Nam
└── dataset/                   # Ảnh + nhãn train/val (không đưa lên git)
```

## 8. Huấn luyện model riêng cho xe Việt Nam (đang phát triển)

Model YOLOv8 gốc train trên tập COCO quốc tế đôi khi nhầm lẫn loại xe
(đặc biệt xe tải/xe buýt) do đặc điểm phương tiện Việt Nam khác biệt.
Quy trình cải thiện (đang thực hiện):

1. `python extract_frames.py` — trích frame từ video thực tế.
2. `python auto_label.py` — tự gán nhãn sơ bộ bằng model hiện có.
3. Soát và sửa lại nhãn thủ công (khuyến nghị dùng
   [makesense.ai](https://www.makesense.ai/), không cần cài đặt).
4. `python train.py` — fine-tune từ `yolov8s.pt` với 4 lớp
   car/motorbike/bus/truck.

## 9. Hạn chế và hướng phát triển

- Chưa hoàn thiện module đọc biển số xe (License Plate Recognition).
- Chưa kết nối trực tiếp với camera giám sát thực tế quy mô lớn
  (RTSP đa camera) — hiện chạy trên video tải lên/offline.
- Chạy trên CPU phổ thông, chưa tối ưu cho phần cứng GPU chuyên dụng.
- Đang xây dựng pipeline huấn luyện model riêng cho đặc điểm phương
  tiện Việt Nam (xem mục 8).
