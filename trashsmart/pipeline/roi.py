# camera_roi.py
"""
ROI cố định ở giữa frame, vùng ngoài tối mờ.
Phím tắt:
    Q / ESC — Thoát
"""

import logging
import os
import sys
import threading
import time
from pathlib import Path

import cv2
import numpy as np
import torch

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from trashsmart.pipeline.utils import preprocess_batch_for_resnet

from trashsmart.pipeline.config import (
    CAMERA_FPS,
    CAMERA_HEIGHT,
    CAMERA_INDEX,
    CAMERA_WIDTH,
    CLASS_NAMES,
    CONF_THRESHOLD,
    NUM_CLASSES,
    OUTPUT_DIR,
    RESNET_MODEL_PATH,
    WINDOW_NAME,
    YOLO_OBJECT_LABEL,
    YOLO_IMG_SIZE,
    YOLO_MODEL_PATH,
)

from trashsmart.pipeline.models import load_resnet_model, load_yolo_model
from trashsmart.pipeline.utils import build_detection_label, get_class_labels_batch

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ROI cố định — hình vuông căn giữa frame, kích thước = YOLO_IMG_SIZE
ROI_SIZE      = YOLO_IMG_SIZE          # 640px — đúng kích thước đầu vào YOLO
OVERLAY_ALPHA = 0.65                   # Độ tối vùng ngoài ROI (0=trong suốt, 1=đen hoàn toàn)

# Bbox stabilization
BBOX_STABILITY_THRESHOLD = 20     # px — chỉ cập nhật tọa độ khi lệch > ngưỡng
BBOX_DISAPPEAR_FRAMES    = 16           # frame — giữ bbox khi model bỏ sót



# ========================
# ROI cố định — tính toán 1 lần
# ========================

def compute_fixed_roi(frame_w: int, frame_h: int) -> tuple:
    """
    Tính ROI vuông căn giữa frame.
    Trả về (x1, y1, x2, y2).
    """
    size = min(ROI_SIZE, frame_w, frame_h)
    x1   = (frame_w - size) // 2
    y1   = (frame_h - size) // 2
    return x1, y1, x1 + size, y1 + size


def draw_overlay(frame: np.ndarray, roi: tuple) -> np.ndarray:
    """
    Vẽ overlay tối mờ bên ngoài ROI giống giao diện quét QR.
    Vùng trong ROI giữ nguyên sáng.
    """
    x1, y1, x2, y2 = roi
    overlay = frame.copy()

    # Tô đen toàn bộ frame trên overlay
    overlay[:] = 0

    # Khôi phục vùng ROI về ảnh gốc
    overlay[y1:y2, x1:x2] = frame[y1:y2, x1:x2]

    # Blend overlay (vùng ngoài tối) với frame gốc
    result = cv2.addWeighted(overlay, 1.0, frame, OVERLAY_ALPHA, 0)

    # Giữ vùng ROI hoàn toàn sáng (không bị blend)
    result[y1:y2, x1:x2] = frame[y1:y2, x1:x2]

    # Vẽ viền ROI
    cv2.rectangle(result, (x1, y1), (x2, y2), (255, 255, 255), 2)

    # 4 góc nổi bật hơn (giống UI quét QR)
    corner_len = 24
    corner_t   = 4
    color_corner = (255, 255, 255)
    for cx, cy, dx, dy in [
        (x1, y1,  1,  1),
        (x2, y1, -1,  1),
        (x1, y2,  1, -1),
        (x2, y2, -1, -1),
    ]:
        cv2.line(result, (cx, cy), (cx + dx * corner_len, cy), color_corner, corner_t)
        cv2.line(result, (cx, cy), (cx, cy + dy * corner_len), color_corner, corner_t)

    return result

# ========================
# Bbox Stabilizer
# ========================

class BboxStabilizer:
    def __init__(self):
        self._stable: list = []

    def update(self, new_detections: list):
        matched_stable = set()
        matched_new    = set()

        for ni, nd in enumerate(new_detections):
            best_iou, best_si = 0, -1
            for si, sd in enumerate(self._stable):
                iou = _iou(nd["bbox"], sd["bbox"])
                if iou > best_iou:
                    best_iou, best_si = iou, si

            if best_iou > 0.3:
                matched_stable.add(best_si)
                matched_new.add(ni)
                if _bbox_shift(nd["bbox"], self._stable[best_si]["bbox"]) > BBOX_STABILITY_THRESHOLD:
                    self._stable[best_si]["bbox"] = nd["bbox"]
                self._stable[best_si].update({
                    "label":        nd["label"],
                    "object_label": nd["object_label"],
                    "material_label": nd["material_label"],
                    "label_resnet": nd["label_resnet"],
                    "conf_resnet":  nd["conf_resnet"],
                    "label_yolo":   nd["label_yolo"],
                    "raw_yolo_label": nd["raw_yolo_label"],
                    "miss_count":   0,
                })

        for si in range(len(self._stable)):
            if si not in matched_stable:
                self._stable[si]["miss_count"] += 1

        self._stable = [s for s in self._stable if s["miss_count"] < BBOX_DISAPPEAR_FRAMES]

        for ni, nd in enumerate(new_detections):
            if ni not in matched_new:
                self._stable.append({**nd, "miss_count": 0})

    def get_stable(self) -> list:
        return list(self._stable)

    def clear(self):
        self._stable = []


def _iou(b1, b2) -> float:
    x1, y1 = max(b1[0], b2[0]), max(b1[1], b2[1])
    x2, y2 = min(b1[2], b2[2]), min(b1[3], b2[3])
    inter  = max(0, x2 - x1) * max(0, y2 - y1)
    if inter == 0:
        return 0.0
    a1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
    a2 = (b2[2] - b2[0]) * (b2[3] - b2[1])
    return inter / (a1 + a2 - inter)


def _bbox_shift(b1, b2) -> float:
    cx1, cy1 = (b1[0] + b1[2]) / 2, (b1[1] + b1[3]) / 2
    cx2, cy2 = (b2[0] + b2[2]) / 2, (b2[1] + b2[3]) / 2
    return ((cx1 - cx2) ** 2 + (cy1 - cy2) ** 2) ** 0.5


# ========================
# Inference
# ========================

def get_detections(frame, roi, yolo_model, resnet_model) -> list:
    """
    Chạy YOLO trong vùng ROI, ResNet classify từng crop.
    bbox trả về là tọa độ trong frame gốc.
    """
    x1r, y1r, x2r, y2r = roi
    region = frame[y1r:y2r, x1r:x2r]   # Crop vùng ROI để đưa vào YOLO

    results = yolo_model(region, imgsz=YOLO_IMG_SIZE, conf=CONF_THRESHOLD, verbose=False)

    valid_items = []
    for result in results:
        if result.boxes is None or len(result.boxes) == 0:
            continue
        for box in result.boxes:
            yolo_conf = box.conf.item()
            if yolo_conf < CONF_THRESHOLD:
                continue
            yolo_cls_id = int(box.cls.item())
            raw_yolo_label = result.names.get(yolo_cls_id, "unknown")

            # Offset tọa độ về frame gốc
            bx1, by1, bx2, by2 = map(int, box.xyxy[0])
            bx1, by1 = bx1 + x1r, by1 + y1r
            bx2, by2 = bx2 + x1r, by2 + y1r

            crop = frame[by1:by2, bx1:bx2]
            if crop.size == 0:
                continue

            valid_items.append({
                "crop":       crop,
                "label_yolo": YOLO_OBJECT_LABEL,
                "raw_yolo_label": raw_yolo_label,
                "conf_yolo":  yolo_conf,
                "bbox":       (bx1, by1, bx2, by2),
            })

    if not valid_items:
        return []

    crops = [item["crop"] for item in valid_items]
    batch = preprocess_batch_for_resnet(crops)

    with torch.no_grad():
        outputs = resnet_model(batch)

    resnet_results = get_class_labels_batch(outputs, CLASS_NAMES)

    detections = []
    for item, (resnet_label, resnet_conf) in zip(valid_items, resnet_results):
        combined_label = build_detection_label(item["label_yolo"], resnet_label)
        detections.append({
            "bbox":         item["bbox"],
            "label":        combined_label,
            "object_label": item["label_yolo"],
            "material_label": resnet_label,
            "label_resnet": resnet_label,
            "conf_resnet":  resnet_conf,
            "label_yolo":   item["label_yolo"],
            "raw_yolo_label": item["raw_yolo_label"],
            "conf_yolo":    item["conf_yolo"],
        })

    return detections


def draw_detections(frame: np.ndarray, detections: list):
    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        label = det["label"]
        conf  = det["conf_resnet"]
        color = (0, 255, 0) if det["label_resnet"] != "unknown" else (0, 0, 255)

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        text = f"{label} {conf:.0%}"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(frame, (x1, y1 - th - 10), (x1 + tw + 6, y1), color, -1)
        cv2.putText(frame, text, (x1 + 3, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)


# ========================
# Inference Thread
# ========================

class InferenceThread:
    def __init__(self, yolo_model, resnet_model, roi):
        self._yolo       = yolo_model
        self._resnet     = resnet_model
        self._roi        = roi
        self._lock       = threading.Lock()
        self._stop_event = threading.Event()
        self._input_frame = None
        self._detections  = []
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def push_frame(self, frame: np.ndarray):
        with self._lock:
            self._input_frame = frame.copy()

    def get_detections(self) -> list:
        with self._lock:
            return list(self._detections)

    def stop(self):
        self._stop_event.set()
        self._thread.join(timeout=3)

    def _run(self):
        while not self._stop_event.is_set():
            with self._lock:
                frame = self._input_frame
                self._input_frame = None

            if frame is None:
                time.sleep(0.005)
                continue

            dets = get_detections(frame, self._roi, self._yolo, self._resnet)

            with self._lock:
                self._detections = dets


# ========================
# Main Loop
# ========================

def run_camera():
    os.makedirs(str(OUTPUT_DIR), exist_ok=True)

    logger.info("Đang load models...")
    yolo_model   = load_yolo_model(YOLO_MODEL_PATH)
    resnet_model = load_resnet_model(RESNET_MODEL_PATH, NUM_CLASSES)

    cap = cv2.VideoCapture(CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  CAMERA_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS,          CAMERA_FPS)

    if not cap.isOpened():
        logger.error(f"Không mở được camera index={CAMERA_INDEX}")
        return

    # Đọc 1 frame để lấy kích thước thực tế
    ret, init_frame = cap.read()
    if not ret:
        logger.error("Không đọc được frame từ camera")
        return

    fh, fw = init_frame.shape[:2]
    roi    = compute_fixed_roi(fw, fh)
    logger.info(f"Camera {fw}x{fh} | ROI {ROI_SIZE}x{ROI_SIZE} tại {roi}")

    stabilizer       = BboxStabilizer()
    inference_thread = InferenceThread(yolo_model, resnet_model, roi)

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

    fps_ts      = time.time()
    fps_val     = 0.0
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            logger.warning("Mất kết nối camera")
            break

        frame_count += 1

        # FPS
        if frame_count % 30 == 0:
            elapsed = time.time() - fps_ts
            fps_val = 30 / elapsed if elapsed > 0 else 0
            fps_ts  = time.time()

        # Đẩy frame để inference thread xử lý
        inference_thread.push_frame(frame)

        # Cập nhật stabilizer từ kết quả mới nhất
        new_dets = inference_thread.get_detections()
        stabilizer.update(new_dets)

        # Vẽ overlay tối ngoài ROI
        display = draw_overlay(frame, roi)

        # Vẽ bbox ổn định
        draw_detections(display, stabilizer.get_stable())

        # HUD
        cv2.putText(display, f"FPS: {fps_val:.1f}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 0), 2)
        cv2.putText(display, "Q=Thoat",
                    (10, fh - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)

        cv2.imshow(WINDOW_NAME, display)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            break

    inference_thread.stop()
    cap.release()
    cv2.destroyAllWindows()
    logger.info("Đã thoát.")


if __name__ == "__main__":
    run_camera()
