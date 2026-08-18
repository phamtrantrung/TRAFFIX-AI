# LICENSE PLATE DETECTION & OCR MODULE
#
# Yêu cầu:
#   pip install ultralytics easyocr
#
# Model detect biển số (plate_model) CẦN được train riêng hoặc tải model
# có sẵn cho biển số Việt Nam (tìm trên Roboflow Universe / HuggingFace
# với từ khóa "vietnamese license plate yolo"). Model YOLO xe (vehicle
# model) là model bạn đã dùng để đếm xe từ trước, KHÔNG dùng chung
# checkpoint với plate_model.

import re
import cv2
import numpy as np


class PlateReader:
    def __init__(self, plate_model_path, use_gpu=False):
        from ultralytics import YOLO
        import easyocr

        self.plate_model = YOLO(plate_model_path)
        # EasyOCR: 'en' đọc tốt biển số dạng chữ-số Latin của VN.
        self.ocr_reader = easyocr.Reader(['en'], gpu=use_gpu)

    def detect_plate_boxes(self, vehicle_crop, conf=0.4):
        """Chạy model detect biển số trên ảnh đã crop theo bbox xe.
        Trả về danh sách bbox [(x1,y1,x2,y2), ...] tương đối trong vehicle_crop."""
        if vehicle_crop is None or vehicle_crop.size == 0:
            return []

        results = self.plate_model.predict(
            vehicle_crop, conf=conf, verbose=False
        )
        boxes = []
        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                boxes.append((int(x1), int(y1), int(x2), int(y2)))
        return boxes

    @staticmethod
    def _preprocess_plate(plate_img):
        """Tiền xử lý ảnh biển số để tăng độ chính xác OCR:
        chuyển xám, tăng tương phản (CLAHE), khử nhiễu nhẹ."""
        gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        denoised = cv2.bilateralFilter(enhanced, 5, 50, 50)
        return denoised

    @staticmethod
    def _normalize_plate_text(raw_text):
        """Chuẩn hóa text đọc được về định dạng biển số VN, loại ký tự lạ.
        VD: '51f-123.45' -> '51F-12345'."""
        cleaned = re.sub(r"[^A-Za-z0-9]", "", raw_text).upper()
        return cleaned

    def read_plate(self, vehicle_crop, plate_bbox):
        """Đọc text biển số từ 1 bbox cụ thể. Trả về (text, confidence)
        hoặc (None, 0.0) nếu không đọc được."""
        x1, y1, x2, y2 = plate_bbox
        plate_img = vehicle_crop[max(0, y1):y2, max(0, x1):x2]

        if plate_img.size == 0:
            return None, 0.0

        processed = self._preprocess_plate(plate_img)
        results = self.ocr_reader.readtext(processed)

        if not results:
            return None, 0.0

        # Ghép các đoạn text được OCR nhận diện (biển số VN có thể chia 2 dòng)
        # theo thứ tự tọa độ y (dòng trên trước, dòng dưới sau)
        results_sorted = sorted(results, key=lambda r: r[0][0][1])
        full_text = "".join([r[1] for r in results_sorted])
        avg_conf = sum([r[2] for r in results_sorted]) / len(results_sorted)

        normalized = self._normalize_plate_text(full_text)
        if len(normalized) < 5:  # quá ngắn, khả năng cao là đọc sai/nhiễu
            return None, 0.0

        return normalized, round(float(avg_conf), 3)

    def process_vehicle(self, frame, vehicle_bbox, conf=0.4):
        """Pipeline đầy đủ: crop xe từ frame -> detect biển số -> OCR.
        Trả về dict {'plate_text', 'confidence', 'plate_bbox_abs'} hoặc None."""
        vx1, vy1, vx2, vy2 = vehicle_bbox
        vehicle_crop = frame[max(0, vy1):vy2, max(0, vx1):vx2]

        plate_boxes = self.detect_plate_boxes(vehicle_crop, conf=conf)
        if not plate_boxes:
            return None

        # Chọn bbox biển số lớn nhất (thường là rõ nét nhất)
        best_box = max(plate_boxes, key=lambda b: (b[2]-b[0])*(b[3]-b[1]))
        text, confidence = self.read_plate(vehicle_crop, best_box)

        if text is None:
            return None

        px1, py1, px2, py2 = best_box
        abs_bbox = (vx1 + px1, vy1 + py1, vx1 + px2, vy1 + py2)

        return {
            "plate_text": text,
            "confidence": confidence,
            "plate_bbox_abs": abs_bbox,
        }
