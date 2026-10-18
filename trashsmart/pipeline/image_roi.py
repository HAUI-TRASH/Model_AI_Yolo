# image_roi.py - Phân loại rác trên ảnh tĩnh với ROI vẽ tay
"""
Cách dùng:
    python image_roi.py --image path/to/image.jpg

Hướng dẫn:
    1. Ảnh hiện ra — kéo chuột vẽ ô vuông quanh vật cần phân loại
    2. Nhấn SPACE hoặc ENTER để chạy phân loại
    3. Nhấn R để vẽ lại ROI
    4. Nhấn S để lưu kết quả
    5. Nhấn Q / ESC để thoát
"""

import argparse
import logging
import os
import sys
import uuid
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from trashsmart.pipeline.utils import preprocess_batch_for_resnet

from trashsmart.pipeline.config import (
    CLASS_NAMES,
    CONF_THRESHOLD,
    CROPS_DIR,
    DEVICE,
    IMAGENET_MEAN,
    IMAGENET_STD,
    MIN_CROP_SIZE,
    NUM_CLASSES,
    OUTPUT_DIR,
    RESNET_INPUT_SIZE,
    RESNET_MODEL_PATH,
    RESNET_RESIZE,
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

WINDOW_NAME = "Image ROI Classifier"

# Hiển thị ảnh tối đa bao nhiêu px (để vừa màn hình)
MAX_DISPLAY_SIZE = 900

# ========================
# ROI Drawing State
# ========================

class ROIDrawer:
    """
    Cho phép người dùng vẽ nhiều ô ROI trên ảnh bằng chuột.
    Mỗi lần kéo chuột = 1 ROI mới được thêm vào danh sách.
    """

    def __init__(self, image: np.ndarray):
        self.image      = image.copy()     # Ảnh gốc — không bao giờ bị sửa
        self.rois: list = []               # List (x1, y1, x2, y2)
        self._drawing   = False
        self._start     = None
        self._current   = None            # Điểm chuột hiện tại khi đang kéo

    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self._drawing = True
            self._start   = (x, y)
            self._current = (x, y)

        elif event == cv2.EVENT_MOUSEMOVE and self._drawing:
            self._current = (x, y)

        elif event == cv2.EVENT_LBUTTONUP:
            self._drawing = False
            x1 = min(self._start[0], x)
            y1 = min(self._start[1], y)
            x2 = max(self._start[0], x)
            y2 = max(self._start[1], y)
            w, h = x2 - x1, y2 - y1
            if w > MIN_CROP_SIZE and h > MIN_CROP_SIZE:
                self.rois.append((x1, y1, x2, y2))
                logger.info(f"ROI #{len(self.rois)} thêm: ({x1},{y1}) → ({x2},{y2})")
            self._start   = None
            self._current = None

    def clear(self):
        self.rois = []
        self._drawing = False
        self._start   = None
        self._current = None

    def get_display(self) -> np.ndarray:
        """
        Trả về ảnh hiển thị với tất cả ROI đã vẽ
        và ô đang kéo (nét đứt).
        """
        vis = self.image.copy()

        # Vẽ các ROI đã xác nhận
        for i, (x1, y1, x2, y2) in enumerate(self.rois, start=1):
            cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 200, 255), 2)
            cv2.putText(vis, f"ROI {i}", (x1 + 4, y1 + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)

        # Vẽ ô đang kéo (nét đứt)
        if self._drawing and self._start and self._current:
            x1 = min(self._start[0], self._current[0])
            y1 = min(self._start[1], self._current[1])
            x2 = max(self._start[0], self._current[0])
            y2 = max(self._start[1], self._current[1])
            _draw_dashed_rect(vis, (x1, y1), (x2, y2), (255, 255, 255), 2)

        return vis


def _draw_dashed_rect(img, pt1, pt2, color, thickness, gap=10):
    """Vẽ hình chữ nhật nét đứt."""
    x1, y1 = pt1
    x2, y2 = pt2
    for x in range(x1, x2, gap * 2):
        cv2.line(img, (x, y1), (min(x + gap, x2), y1), color, thickness)
        cv2.line(img, (x, y2), (min(x + gap, x2), y2), color, thickness)
    for y in range(y1, y2, gap * 2):
        cv2.line(img, (x1, y), (x1, min(y + gap, y2)), color, thickness)
        cv2.line(img, (x2, y), (x2, min(y + gap, y2)), color, thickness)


# ========================
# Inference
# ========================

def run_inference_on_rois(
    image: np.ndarray,
    rois: list,
    yolo_model,
    resnet_model,
) -> list:
    """
    Với mỗi ROI người dùng vẽ:
      1. Crop vùng ROI
      2. Chạy YOLO detect bên trong ROI
      3. ResNet classify từng detection

    Nếu YOLO không detect được gì trong ROI →
    dùng chính vùng ROI đó làm input cho ResNet trực tiếp.

    Returns:
        list detection dicts với tọa độ trong ảnh gốc
    """
    all_detections = []

    for roi_idx, (rx1, ry1, rx2, ry2) in enumerate(rois, start=1):
        region = image[ry1:ry2, rx1:rx2]
        if region.size == 0:
            continue

        logger.info(f"[ROI {roi_idx}] Chạy YOLO ({rx2-rx1}x{ry2-ry1}px)...")
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
                bx1, by1, bx2, by2 = map(int, box.xyxy[0])
                # Offset về ảnh gốc
                bx1, by1 = bx1 + rx1, by1 + ry1
                bx2, by2 = bx2 + rx1, by2 + ry1
                crop = image[by1:by2, bx1:bx2]
                if crop.size == 0 or min(crop.shape[:2]) < MIN_CROP_SIZE:
                    continue
                valid_items.append({
                    "crop":       crop,
                    "label_yolo": YOLO_OBJECT_LABEL,
                    "raw_yolo_label": raw_yolo_label,
                    "conf_yolo":  yolo_conf,
                    "bbox":       (bx1, by1, bx2, by2),
                    "roi_idx":    roi_idx,
                })

        # YOLO không detect được → dùng toàn bộ vùng ROI
        if not valid_items:
            logger.info(f"[ROI {roi_idx}] YOLO không detect → classify toàn ROI")
            valid_items.append({
                "crop":       region,
                "label_yolo": "n/a",
                "raw_yolo_label": "n/a",
                "conf_yolo":  0.0,
                "bbox":       (rx1, ry1, rx2, ry2),
                "roi_idx":    roi_idx,
            })

        pil_images = []
        for item in valid_items:
            pil = Image.fromarray(cv2.cvtColor(item["crop"], cv2.COLOR_BGR2RGB))
            pil_images.append(pil)

        # Gọi đúng — truyền list crops (numpy BGR), không phải PIL
        crops = [item["crop"] for item in valid_items]
        batch = preprocess_batch_for_resnet(crops)  # ← truyền list numpy array

        with torch.no_grad():
            outputs = resnet_model(batch)

        resnet_results = get_class_labels_batch(outputs, CLASS_NAMES)

        os.makedirs(str(CROPS_DIR), exist_ok=True)
        for item, (resnet_label, resnet_conf) in zip(valid_items, resnet_results):
            combined_label = build_detection_label(item["label_yolo"], resnet_label)
            crop_path = os.path.join(
                str(CROPS_DIR),
                f"roi{roi_idx}_{combined_label}_{uuid.uuid4().hex[:6]}.jpg",
            )
            cv2.imwrite(crop_path, item["crop"])
            logger.info(f"[ROI {roi_idx}] → {combined_label} ({resnet_conf:.0%})")

            all_detections.append({
                "bbox":         item["bbox"],
                "label":        combined_label,
                "object_label": item["label_yolo"],
                "material_label": resnet_label,
                "label_resnet": resnet_label,
                "conf_resnet":  resnet_conf,
                "label_yolo":   item["label_yolo"],
                "raw_yolo_label": item["raw_yolo_label"],
                "conf_yolo":    item["conf_yolo"],
                "roi_idx":      item["roi_idx"],
                "crop_path":    crop_path,
            })

    return all_detections


def draw_results(image: np.ndarray, detections: list) -> np.ndarray:
    """Vẽ kết quả phân loại lên ảnh."""
    vis = image.copy()
    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        label = det["label"]
        conf  = det["conf_resnet"]
        color = (0, 255, 0) if det["label_resnet"] != "unknown" else (0, 0, 255)

        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)

        text = f"{label} {conf:.0%}"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
        cv2.rectangle(vis, (x1, y1 - th - 12), (x1 + tw + 8, y1), color, -1)
        cv2.putText(vis, text, (x1 + 4, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 2)

    return vis


# ========================
# Display Helper
# ========================

def resize_for_display(image: np.ndarray, max_size: int = MAX_DISPLAY_SIZE):
    """
    Scale ảnh xuống để vừa màn hình, trả về (ảnh đã scale, scale_factor).
    Dùng scale_factor để chuyển đổi tọa độ chuột → tọa độ ảnh gốc.
    """
    h, w = image.shape[:2]
    scale = min(max_size / w, max_size / h, 1.0)
    if scale < 1.0:
        new_w, new_h = int(w * scale), int(h * scale)
        return cv2.resize(image, (new_w, new_h)), scale
    return image.copy(), 1.0


# ========================
# Main
# ========================

def main(image_path: str):
    os.makedirs(str(OUTPUT_DIR), exist_ok=True)

    # Load ảnh
    image = cv2.imread(image_path)
    if image is None:
        logger.error(f"Không đọc được ảnh: {image_path}")
        return

    logger.info(f"Ảnh: {image_path} ({image.shape[1]}x{image.shape[0]})")

    # Load models
    logger.info("Đang load models...")
    yolo_model   = load_yolo_model(YOLO_MODEL_PATH)
    resnet_model = load_resnet_model(RESNET_MODEL_PATH, NUM_CLASSES)

    # Scale ảnh để hiển thị vừa màn hình
    display_img, scale = resize_for_display(image)
    dh, dw = display_img.shape[:2]

    # ROIDrawer làm việc trên ảnh display
    # Tọa độ ROI sẽ được scale ngược về ảnh gốc khi inference
    drawer = ROIDrawer(display_img)

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, dw, dh)
    cv2.setMouseCallback(WINDOW_NAME, drawer.mouse_callback)

    detections   = []
    result_image = None
    mode         = "draw"   # "draw" | "result"

    logger.info("Kéo chuột vẽ ROI → SPACE/ENTER chạy | R vẽ lại | S lưu | Q thoát")

    while True:
        if mode == "draw":
            frame = drawer.get_display()

            # Hướng dẫn
            n = len(drawer.rois)
            hint = f"ROI: {n}  |  SPACE/ENTER=Chay  R=Ve lai  Q=Thoat"
            cv2.putText(frame, hint, (10, dh - 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
            if n == 0:
                msg = "Keo chuot de ve vung ROI"
                (tw, _), _ = cv2.getTextSize(msg, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
                cv2.putText(frame, msg, ((dw - tw) // 2, dh // 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        else:  # result
            frame = result_image.copy()
            hint = f"S=Luu  R=Ve lai  Q=Thoat"
            cv2.putText(frame, hint, (10, dh - 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)

        cv2.imshow(WINDOW_NAME, frame)
        key = cv2.waitKey(20) & 0xFF

        # Thoát
        if key in (ord("q"), 27):
            break

        # Chạy inference
        elif key in (ord(" "), 13) and mode == "draw":
            if not drawer.rois:
                logger.warning("Chưa vẽ ROI nào!")
                continue

            # Scale tọa độ ROI về ảnh gốc
            original_rois = [
                (
                    int(x1 / scale), int(y1 / scale),
                    int(x2 / scale), int(y2 / scale),
                )
                for x1, y1, x2, y2 in drawer.rois
            ]

            logger.info(f"Chạy inference trên {len(original_rois)} ROI...")
            detections = run_inference_on_rois(
                image, original_rois, yolo_model, resnet_model
            )

            # Scale tọa độ bbox kết quả về display để vẽ
            display_detections = []
            for det in detections:
                x1, y1, x2, y2 = det["bbox"]
                display_detections.append({
                    **det,
                    "bbox": (
                        int(x1 * scale), int(y1 * scale),
                        int(x2 * scale), int(y2 * scale),
                    ),
                })

            result_image = draw_results(display_img, display_detections)
            mode = "result"
            logger.info(f"Hoàn tất: {len(detections)} detection")

        # Vẽ lại ROI
        elif key == ord("r"):
            drawer.clear()
            detections   = []
            result_image = None
            mode         = "draw"
            logger.info("Reset — vẽ lại ROI")

        # Lưu kết quả
        elif key == ord("s") and mode == "result":
            out_path = os.path.join(str(OUTPUT_DIR), f"result_{uuid.uuid4().hex[:8]}.jpg")
            # Lưu ảnh gốc kích thước đầy đủ với bbox
            original_detections_for_save = []
            for det in detections:
                original_detections_for_save.append(det)
            saved = draw_results(image, original_detections_for_save)
            cv2.imwrite(out_path, saved)
            logger.info(f"Đã lưu: {out_path}")

    cv2.destroyAllWindows()
    logger.info("Đã thoát.")


# ========================
# Entry Point
# ========================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Image ROI Classifier")
    parser.add_argument(
        "--image", "-i",
        type=str,
        required=True,
        help="Đường dẫn tới ảnh cần phân loại",
    )
    args = parser.parse_args()
    main(args.image)
