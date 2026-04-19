from fastapi import FastAPI, UploadFile, File
from fastapi.responses import Response, JSONResponse
from ultralytics import YOLO
import numpy as np
import cv2
import os
import tempfile

from pydantic import BaseModel, HttpUrl
from typing import Optional, List, Dict, Any
import requests

import cloudinary
import cloudinary.uploader

app = FastAPI(title="Trash Detection API")

MODEL_PATH = "runs/detect/yolo_resumed_training/weights/best.pt"
model = YOLO(MODEL_PATH)

LOW_CONF_THRESHOLD = 0.5

cloudinary.config(
    cloud_name="dtbyrc1ae",
    api_key="418518358217964",
    api_secret="Y2yRe1sy8C1AIvIF7eyUlVwGy20",
    secure=True
)


@app.get("/health")
def health():
    return {"status": "ok"}


class PredictUrlRequest(BaseModel):
    image_url: HttpUrl
    conf: float = 0.25
    iou: float = 0.6
    request_id: Optional[int] = None


def download_image_to_cv2(url: str) -> np.ndarray:
    r = requests.get(url, timeout=15)
    r.raise_for_status()
    np_arr = np.frombuffer(r.content, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    return img


def clamp_bbox(x1: int, y1: int, x2: int, y2: int, width: int, height: int):
    x1 = max(0, min(x1, width - 1))
    y1 = max(0, min(y1, height - 1))
    x2 = max(0, min(x2, width))
    y2 = max(0, min(y2, height))
    return x1, y1, x2, y2


def upload_image_bytes_to_cloudinary(
    image_bytes: bytes,
    folder: str,
    public_id: Optional[str] = None,
    overwrite: bool = False
) -> str:
    res = cloudinary.uploader.upload(
        image_bytes,
        resource_type="image",
        folder=folder,
        public_id=public_id,
        overwrite=overwrite
    )
    return res.get("secure_url")


def upload_crop_to_cloudinary(crop_img: np.ndarray, folder: str, public_id: Optional[str] = None) -> Optional[str]:
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
        overwrite=True if public_id else False
    )
    return res.get("secure_url")


def build_detection_response(
    img: np.ndarray,
    req_conf: float,
    req_iou: float,
    request_id: Optional[int] = None,
    original_image_url: Optional[str] = None
) -> Dict[str, Any]:
    results = model.predict(img, conf=req_conf, iou=req_iou, verbose=False)
    r0 = results[0]

    # 1) Ảnh annotated
    annotated = r0.plot()
    ok, out_jpg = cv2.imencode(".jpg", annotated, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
    if not ok:
        raise RuntimeError("Failed to encode output image")

    annotated_public_id = f"req_{request_id}" if request_id else None
    annotated_url = upload_image_bytes_to_cloudinary(
        out_jpg.tobytes(),
        folder="hauitrash/annotated",
        public_id=annotated_public_id,
        overwrite=True if request_id else False
    )

    # 2) Parse detections + crop low confidence
    detections: List[Dict[str, Any]] = []
    conf_list: List[float] = []
    h, w = img.shape[:2]

    if r0.boxes is not None and len(r0.boxes) > 0:
        names = r0.names

        for idx, b in enumerate(r0.boxes):
            cls_id = int(b.cls.item())
            label_raw = names.get(cls_id, str(cls_id))
            conf_score = float(b.conf.item())

            xyxy = b.xyxy[0].tolist()
            x1, y1, x2, y2 = map(int, xyxy)
            x1, y1, x2, y2 = clamp_bbox(x1, y1, x2, y2, w, h)

            crop_url = None
            needs_confirm = conf_score < LOW_CONF_THRESHOLD

            if needs_confirm and x2 > x1 and y2 > y1:
                crop_img = img[y1:y2, x1:x2]
                crop_public_id = f"req_{request_id}_crop_{idx}" if request_id else None
                try:
                    crop_url = upload_crop_to_cloudinary(
                        crop_img,
                        folder="hauitrash/crops",
                        public_id=crop_public_id
                    )
                except Exception:
                    crop_url = None

            conf_list.append(conf_score)
            detections.append({
                "label": label_raw,
                "confidence": round(conf_score, 4),
                "annotatedUrl": annotated_url,
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2,
                "cropUrl": crop_url,
                "needsConfirm": needs_confirm
            })

    confidence_avg = round(sum(conf_list) / len(conf_list), 4) if conf_list else 0.0
    requires_confirmation = any(d["needsConfirm"] for d in detections)

    return {
        "requestId": request_id,
        "imageUrl": original_image_url,
        "annotatedUrl": annotated_url,
        "params": {"conf": req_conf, "iou": req_iou},
        "count": len(detections),
        "confidenceAvg": confidence_avg,
        "requiresConfirmation": requires_confirmation,
        "detections": detections
    }


@app.post("/predict-image")
async def predict_image(
    file: UploadFile = File(...),
    conf: float = 0.25,
    iou: float = 0.6
):
    img_bytes = await file.read()
    np_arr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    if img is None:
        return JSONResponse({"error": "Invalid image"}, status_code=400)

    try:
        result = build_detection_response(
            img=img,
            req_conf=conf,
            req_iou=iou,
            request_id=None,
            original_image_url=None
        )
        return JSONResponse(content=result, status_code=200)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@app.post("/predict-image-json")
async def predict_image_json(file: UploadFile = File(...), conf: float = 0.25, iou: float = 0.6):
    img_bytes = await file.read()
    np_arr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    if img is None:
        return JSONResponse({"error": "Invalid image"}, status_code=400)

    results = model.predict(img, conf=conf, iou=iou, verbose=False)
    r0 = results[0]

    detections: List[Dict[str, Any]] = []
    if r0.boxes is not None and len(r0.boxes) > 0:
        names = r0.names
        for b in r0.boxes:
            cls_id = int(b.cls.item())
            label_raw = names.get(cls_id, str(cls_id))
            conf_score = float(b.conf.item())
            xyxy = b.xyxy[0].tolist()

            detections.append({
                "label": label_raw,
                "confidence": round(conf_score, 4),
                "labelDisplay": label_raw,
                "x1": int(xyxy[0]),
                "y1": int(xyxy[1]),
                "x2": int(xyxy[2]),
                "y2": int(xyxy[3])
            })

    return {"detections": detections}


@app.post("/predict-image-url")
def predict_image_url(req: PredictUrlRequest) -> Dict[str, Any]:
    try:
        img = download_image_to_cv2(str(req.image_url))
    except Exception as e:
        return JSONResponse(
            {"error": f"Failed to download/decode image: {str(e)}"},
            status_code=400
        )

    if img is None:
        return JSONResponse({"error": "Invalid image data"}, status_code=400)

    try:
        result = build_detection_response(
            img=img,
            req_conf=req.conf,
            req_iou=req.iou,
            request_id=req.request_id,
            original_image_url=str(req.image_url)
        )
        return result
    except Exception as e:
        return JSONResponse({"error": f"Predict/upload failed: {str(e)}"}, status_code=500)