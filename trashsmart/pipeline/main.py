# main.py
import json
import logging
import os
import sys
import threading
import anyio
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import numpy as np
import torch
import cloudinary
import cloudinary.uploader
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from pydantic import BaseModel, HttpUrl, Field

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from trashsmart.pipeline.config import (
    CLASS_NAMES,
    CONF_THRESHOLD,
    CROPS_DIR,
    FIX_ESRGAN,
    INPUT_IMAGE,
    NUM_CLASSES,
    OUTPUT_DIR,
    RESNET_MODEL_PATH,
    RESNET_CONF_THRESHOLD,
    YOLO_OBJECT_LABEL,
    YOLO_IMG_SIZE,
    YOLO_MODEL_PATH,
)
from trashsmart.pipeline.models import load_resnet_model, load_yolo_model
from trashsmart.pipeline.input_limits import download_image, read_upload, ImageInputError
from trashsmart.pipeline.admission import PredictionAdmission
from trashsmart.pipeline.utils import (
    build_detection_label,
    build_vietnamese_detection_label,
    draw_label_box,
    enhance_for_display,
    get_class_labels_batch,
    preprocess_batch_for_resnet,
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
app.add_middleware(PredictionAdmission)


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
_model_lock = threading.Lock()
_inference_lock = threading.Lock()
_health_limiter = anyio.CapacityLimiter(1)


def get_models():
    global _yolo_model, _resnet_model
    with _model_lock:
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
    conf: float = Field(default=CONF_THRESHOLD, ge=0, le=1)
    iou: float = Field(default=0.6, ge=0, le=1)
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
    return download_image(url)


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
        timeout=15,
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
        timeout=15,
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
    conf: float = CONF_THRESHOLD,
    iou: float = 0.6,
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
    with _inference_lock:
        results = yolo_model(image, imgsz=YOLO_IMG_SIZE, conf=conf, iou=iou)
    os.makedirs(str(CROPS_DIR), exist_ok=True)

    # Thu thập tất cả valid crops + metadata
    valid_items = []

    for result in results:
        boxes = result.boxes
        if boxes is None or len(boxes) == 0:
            continue

        for box in boxes:
            yolo_conf = box.conf.item()
            if yolo_conf < conf:
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
    batch_tensor = preprocess_batch_for_resnet([item["crop"] for item in valid_items])

    with _inference_lock, torch.no_grad():
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

        needs_confirm = resnet_conf < RESNET_CONF_THRESHOLD
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
async def health():
    try:
        await anyio.to_thread.run_sync(get_models, limiter=_health_limiter)
    except Exception:
        logger.exception("Model readiness failed")
        return JSONResponse({"status": "unavailable", "modelsReady": False}, status_code=503)
    return {"status": "ok", "modelsReady": True}


@app.get("/live")
async def live():
    return {"status": "ok"}


@app.post("/predict-image")
def predict_image(file: UploadFile = File(...)):
    return predict_uploaded_image(file)


def predict_uploaded_image(file: UploadFile, request_id: Optional[int] = None):
    """Sync endpoints run decoding, inference and uploads in FastAPI's thread pool."""
    try:
        img = read_upload(file.file)
        yolo_model, resnet_model = get_models()
        result = process_image(img, yolo_model, resnet_model, request_id=request_id)
        return JSONResponse(content=result, status_code=200)
    except ImageInputError as e:
        return JSONResponse({"error": str(e)}, status_code=e.status_code)
    except Exception:
        logger.exception("Prediction/upload failed")
        return JSONResponse({"error": "Prediction/upload failed"}, status_code=500)


@app.post("/predict-image-url")
def predict_image_url(req: PredictUrlRequest) -> Dict[str, Any]:
    try:
        img = download_image_to_cv2(str(req.image_url))
    except ImageInputError as e:
        return JSONResponse({"error": str(e)}, status_code=e.status_code)
    except Exception:
        logger.exception("Image URL download failed")
        return JSONResponse(
            {"error": "Failed to download/decode image"},
            status_code=400,
        )

    if img is None:
        return JSONResponse({"error": "Invalid image data"}, status_code=400)

    try:
        yolo_model, resnet_model = get_models()
        result = process_image(
            img,
            yolo_model,
            resnet_model,
            request_id=req.request_id,
            original_image_url=str(req.image_url),
            conf=req.conf,
            iou=req.iou,
        )
        return result
    except Exception:
        logger.exception("URL prediction/upload failed")
        return JSONResponse({"error": "Predict/upload failed"}, status_code=500)


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
def predict_image_json(
    file: UploadFile = File(...),
    request_id: Optional[int] = None
):
    return predict_uploaded_image(file, request_id=request_id)


if __name__ == "__main__":
    main()
