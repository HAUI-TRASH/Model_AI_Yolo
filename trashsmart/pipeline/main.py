# main.py
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import numpy as np
import torch
import requests
import cloudinary
import cloudinary.uploader
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from pydantic import BaseModel, HttpUrl

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from trashsmart.pipeline.config import (
    CLASS_NAMES,
    CONF_THRESHOLD,
    CROPS_DIR,
    FIX_ESRGAN,
    INPUT_IMAGE,
    NUM_CLASSES,
    OUTPUT_DIR,
    RESNET_MODEL_PATH,
    YOLO_OBJECT_LABEL,
    YOLO_IMG_SIZE,
    YOLO_MODEL_PATH,
)
from trashsmart.pipeline.models import load_resnet_model, load_yolo_model
from trashsmart.pipeline.utils import (
    build_detection_label,
    build_vietnamese_detection_label,
    draw_label_box,
    enhance_for_display,
    get_class_labels_batch,
    preprocess_batch_for_resnet,
    quick_enhance_for_resnet,
)

# ========================
# Logging Setup
# ========================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ========================
# FastAPI App
# ========================
app = FastAPI(title="TrashSmart Detection API")

LOW_CONF_THRESHOLD = 0.5

cloudinary.config(
    cloud_name=os.getenv("CLOUDINARY_CLOUD_NAME"),
    api_key=os.getenv("CLOUDINARY_API_KEY"),
    api_secret=os.getenv("CLOUDINARY_API_SECRET"),
    secure=True,
)

# ========================
# Lazy Model Loading
# ========================
_yolo_model = None
_resnet_model = None


def get_models():
    global _yolo_model, _resnet_model
    if _yolo_model is None:
        _yolo_model = load_yolo_model(YOLO_MODEL_PATH)
    if _resnet_model is None:
        _resnet_model = load_resnet_model(RESNET_MODEL_PATH, NUM_CLASSES)
    return _yolo_model, _resnet_model


# ========================
# Schemas
# ========================
class PredictUrlRequest(BaseModel):
    image_url: HttpUrl
    conf: float = 0.25
    iou: float = 0.6
    request_id: Optional[int] = None


# ========================
# Cloudinary Helpers
# ========================
def clamp_bbox(x1: int, y1: int, x2: int, y2: int, width: int, height: int):
    x1 = max(0, min(x1, width - 1))
    y1 = max(0, min(y1, height - 1))
    x2 = max(0, min(x2, width))
    y2 = max(0, min(y2, height))
    return x1, y1, x2, y2


def download_image_to_cv2(url: str) -> np.ndarray:
    r = requests.get(url, timeout=15)
    r.raise_for_status()
    np_arr = np.frombuffer(r.content, np.uint8)
    return cv2.imdecode(np_arr, cv2.IMREAD_COLOR)


def upload_image_bytes_to_cloudinary(
    image_bytes: bytes,
    folder: str,
    public_id: Optional[str] = None,
    overwrite: bool = False,
) -> str:
    res = cloudinary.uploader.upload(
        image_bytes,
        resource_type="image",
        folder=folder,
        public_id=public_id,
        overwrite=overwrite,
    )
    return res.get("secure_url")


def upload_crop_to_cloudinary(
    crop_img: np.ndarray,
    folder: str,
    public_id: Optional[str] = None,
) -> Optional[str]:
    if crop_img is None or crop_img.size == 0:
        return None
    ok, crop_jpg = cv2.imencode(".jpg", crop_img, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
    if not ok:
        return None
    res = cloudinary.uploader.upload(
        crop_jpg.tobytes(),
        resource_type="image",
        folder=folder,
        public_id=public_id,
        overwrite=True if public_id else False,
    )
    return res.get("secure_url")


# ========================
# Core Pipeline
# ========================
def process_image(
    image: np.ndarray,
    yolo_model,
    resnet_model,
    request_id: Optional[int] = None,
    original_image_url: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Pipeline chính:
        1. YOLO detect → lấy bounding boxes
        2. Batch enhance + ResNet classify tất cả crops cùng lúc
        3. ESRGAN upscale từng crop để xuất ảnh đẹp
        4. Upload annotated image + crops (needsConfirm) lên Cloudinary

    Returns:
        dict theo format API (requestId, annotatedUrl, detections, ...)
    """
    response_image_url = original_image_url
    h, w = image.shape[:2]

    # --- YOLO Detection ---
    results = yolo_model(image, imgsz=YOLO_IMG_SIZE)
    os.makedirs(str(CROPS_DIR), exist_ok=True)

    # Thu thập tất cả valid crops + metadata
    valid_items = []

    for result in results:
        boxes = result.boxes
        if boxes is None or len(boxes) == 0:
            continue

        for box in boxes:
            yolo_conf = box.conf.item()
            if yolo_conf < CONF_THRESHOLD:
                continue

            yolo_cls_id = int(box.cls.item())
            raw_yolo_label = result.names.get(yolo_cls_id, "unknown")
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            x1, y1, x2, y2 = clamp_bbox(x1, y1, x2, y2, w, h)

            crop = image[y1:y2, x1:x2]
            if crop.size == 0:
                continue

            valid_items.append({
                "crop":           crop,
                "yolo_label":     YOLO_OBJECT_LABEL,
                "raw_yolo_label": raw_yolo_label,
                "yolo_conf":      yolo_conf,
                "bbox":           (x1, y1, x2, y2),
            })

    if not valid_items:
        logger.info("Không phát hiện đối tượng hợp lệ.")
        # Upload ảnh gốc không có annotation
        annotated = image.copy()
        ok, out_jpg = cv2.imencode(".jpg", annotated, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
        annotated_url = None
        if ok:
            annotated_public_id = f"req_{request_id}" if request_id else None
            annotated_url = upload_image_bytes_to_cloudinary(
                out_jpg.tobytes(),
                folder="hauitrash/annotated",
                public_id=annotated_public_id,
                overwrite=True if request_id else False,
            )
        return {
            "requestId":            request_id,
            "imageUrl":             response_image_url,
            "annotatedUrl":         annotated_url,
            "count":                0,
            "confidenceAvg":        0.0,
            "requiresConfirmation": False,
            "detections":           [],
        }

    logger.info(f"Phát hiện {len(valid_items)} đối tượng → Batch ResNet inference...")

    # --- Batch ResNet Inference ---
    enhanced_crops = [quick_enhance_for_resnet(item["crop"].copy()) for item in valid_items]
    batch_tensor   = preprocess_batch_for_resnet(enhanced_crops)

    with torch.no_grad():
        batch_output = resnet_model(batch_tensor)

    resnet_results = get_class_labels_batch(batch_output, CLASS_NAMES)

    # --- Vẽ bounding box với cả YOLO + ResNet label lên ảnh gốc ---
    annotated = image.copy()
    for item, (resnet_label, resnet_conf) in zip(valid_items, resnet_results):
        x1, y1, x2, y2 = item["bbox"]
        display_label = build_vietnamese_detection_label(item["yolo_label"], resnet_label)
        label_text = f"{display_label} {resnet_conf:.0%}"
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label_y = y1 - 30 if y1 >= 30 else y1 + 4
        draw_label_box(
            annotated,
            label_text,
            x1,
            label_y,
            font_size=20,
            bg_color=(0, 255, 0),
            text_color=(0, 0, 0),
        )

    ok, out_jpg = cv2.imencode(".jpg", annotated, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
    if not ok:
        raise RuntimeError("Failed to encode annotated image")

    annotated_public_id = f"req_{request_id}" if request_id else None
    annotated_url = upload_image_bytes_to_cloudinary(
        out_jpg.tobytes(),
        folder="hauitrash/annotated",
        public_id=annotated_public_id,
        overwrite=True if request_id else False,
    )

    # --- Xuất ảnh crop + build response ---
    detections: List[Dict[str, Any]] = []
    conf_list:  List[float] = []

    for i, (item, (resnet_label, resnet_conf)) in enumerate(
        zip(valid_items, resnet_results), start=1
    ):
        crop  = item["crop"]
        x1, y1, x2, y2 = item["bbox"]
        combined_label = build_detection_label(item["yolo_label"], resnet_label)
        display_label = build_vietnamese_detection_label(item["yolo_label"], resnet_label)

        # ESRGAN upscale để xuất ảnh đẹp
        if FIX_ESRGAN:
            logger.info(f"  [{i}/{len(valid_items)}] ESRGAN upscale: {resnet_label}")
            crop_final = enhance_for_display(crop)
        else:
            crop_final = crop.copy()

        # Vẽ label lên crop
        label_text = f"{display_label} ({resnet_conf:.0%})"
        draw_label_box(
            crop_final,
            label_text,
            10,
            10,
            font_size=20,
            bg_color=(0, 255, 0),
            text_color=(0, 0, 0),
        )

        needs_confirm = resnet_conf < LOW_CONF_THRESHOLD
        crop_url = None

        if needs_confirm:
            crop_public_id = f"req_{request_id}_crop_{i}" if request_id else None
            try:
                crop_url = upload_crop_to_cloudinary(
                    crop_final,
                    folder="hauitrash/crops",
                    public_id=crop_public_id,
                )
            except Exception:
                crop_url = None

        conf_list.append(resnet_conf)

        detection: Dict[str, Any] = {
            "label": resnet_label,
            "labelDisplay":  display_label,
            "confidence":    round(resnet_conf, 4),
            "x1":            x1,
            "y1":            y1,
            "x2":            x2,
            "y2":            y2,
            "annotatedUrl":  annotated_url,
            "needsConfirm":  needs_confirm,
        }
        if crop_url:
            detection["cropUrl"] = crop_url

        detections.append(detection)

    confidence_avg        = round(sum(conf_list) / len(conf_list), 4) if conf_list else 0.0
    requires_confirmation = any(d.get("needsConfirm", False) for d in detections)

    logger.info(f"Hoàn tất: {len(detections)} đối tượng được phân loại.")

    return {
        "requestId":            request_id,
        "imageUrl":             response_image_url,
        "annotatedUrl":         annotated_url,
        "count":                len(detections),
        "confidenceAvg":        confidence_avg,
        "requiresConfirmation": requires_confirmation,
        "detections":           detections,
    }


# ========================
# API Endpoints
# ========================
@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict-image")
async def predict_image(file: UploadFile = File(...)):
    img_bytes = await file.read()
    np_arr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    if img is None:
        return JSONResponse({"error": "Invalid image"}, status_code=400)

    yolo_model, resnet_model = get_models()
    try:
        result = process_image(img, yolo_model, resnet_model)
        return JSONResponse(content=result, status_code=200)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@app.post("/predict-image-url")
def predict_image_url(req: PredictUrlRequest) -> Dict[str, Any]:
    try:
        img = download_image_to_cv2(str(req.image_url))
    except Exception as e:
        return JSONResponse(
            {"error": f"Failed to download/decode image: {str(e)}"},
            status_code=400,
        )

    if img is None:
        return JSONResponse({"error": "Invalid image data"}, status_code=400)

    yolo_model, resnet_model = get_models()
    try:
        result = process_image(
            img,
            yolo_model,
            resnet_model,
            request_id=req.request_id,
            original_image_url=str(req.image_url),
        )
        return result
    except Exception as e:
        return JSONResponse({"error": f"Predict/upload failed: {str(e)}"}, status_code=500)


# ========================
# Entry Point (CLI)
# ========================
def main():
    os.makedirs(str(OUTPUT_DIR), exist_ok=True)

    yolo_model   = load_yolo_model(YOLO_MODEL_PATH)
    resnet_model = load_resnet_model(RESNET_MODEL_PATH, NUM_CLASSES)

    logger.info("Bắt đầu xử lý ảnh...")
    image = cv2.imread(str(INPUT_IMAGE))
    if image is None:
        logger.error(f"Không đọc được ảnh: {INPUT_IMAGE}")
        return

    response = process_image(image, yolo_model, resnet_model)

    json_path = os.path.join(str(OUTPUT_DIR), "response.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(response, f, ensure_ascii=False, indent=2)

    logger.info(f"Kết quả JSON: {json_path}")
    logger.info(f"Tổng đối tượng phát hiện: {len(response['detections'])}")
@app.post("/predict-image-json")
async def predict_image_json(
    file: UploadFile = File(...),
    request_id: Optional[int] = None
):
    try:
        # Đọc ảnh upload
        img_bytes = await file.read()
        np_arr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if img is None:
            return JSONResponse(
                {"error": "Invalid image"},
                status_code=400
            )

        # Load YOLO + ResNet
        yolo_model, resnet_model = get_models()

        # Chạy pipeline YOLO -> crop -> ResNet -> trả JSON
        result = process_image(
            image=img,
            yolo_model=yolo_model,
            resnet_model=resnet_model,
            request_id=request_id
        )

        return JSONResponse(content=result, status_code=200)

    except Exception as e:
        logger.exception("Predict image json failed")
        return JSONResponse(
            {"error": f"Predict failed: {str(e)}"},
            status_code=500
        )


if __name__ == "__main__":
    main()
