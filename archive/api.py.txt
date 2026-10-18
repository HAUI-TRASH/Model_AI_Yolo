
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import Response, JSONResponse
from ultralytics import YOLO
import numpy as np
import cv2
import os

from pydantic import BaseModel, HttpUrl
from typing import Optional, List, Dict, Any
import requests
from pathlib import Path

import cloudinary
import cloudinary.uploader
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent / ".env")

app = FastAPI(title="Trash Detection API")

MODEL_PATH = "runs/detect/multiTrash/best.pt"
model = YOLO(MODEL_PATH)

#  Khuyên dùng ENV thay vì hardcode secret
cloudinary.config(
    cloud_name=os.getenv("CLOUDINARY_CLOUD_NAME"),
    api_key=os.getenv("CLOUDINARY_API_KEY"),
    api_secret=os.getenv("CLOUDINARY_API_SECRET"),
    secure=True
)



@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict-image")
async def predict_image(file: UploadFile = File(...), conf: float = 0.25, iou: float = 0.6):
    img_bytes = await file.read()
    np_arr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    if img is None:
        return JSONResponse({"error": "Invalid image"}, status_code=400)

    results = model.predict(img, conf=conf, iou=iou, verbose=False)
    annotated = results[0].plot()

    ok, out_jpg = cv2.imencode(".jpg", annotated)
    if not ok:
        return JSONResponse({"error": "Failed to encode output image"}, status_code=500)

    return Response(content=out_jpg.tobytes(), media_type="image/jpeg")
    

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
        names = r0.names  # dict: class_id -> class_name
        for b in r0.boxes:
            cls_id = int(b.cls.item())
            label_raw = names.get(cls_id, str(cls_id))
            conf_score = float(b.conf.item())
            xyxy = b.xyxy[0].tolist()
            
            detections.append({
                "label": label_raw,
                "confidence": round(conf_score, 4),
                "x1": int(xyxy[0]),
                "y1": int(xyxy[1]),
                "x2": int(xyxy[2]),
                "y2": int(xyxy[3])
            })

    return {"detections": detections}


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


@app.post("/predict-image-url")
def predict_image_url(req: PredictUrlRequest) -> Dict[str, Any]:
    # 1) Download ảnh từ URL
    try:
        img = download_image_to_cv2(str(req.image_url))
    except Exception as e:
        return JSONResponse({"error": f"Failed to download/decode image: {str(e)}"}, status_code=400)

    if img is None:
        return JSONResponse({"error": "Invalid image data"}, status_code=400)

    # 2) Predict
    results = model.predict(img, conf=req.conf, iou=req.iou, verbose=False)
    r0 = results[0]

    # 3) Vẽ box + label lên ảnh (dù count=0 vẫn ra ảnh)
    annotated = r0.plot()

    # 4) Encode ảnh -> jpg bytes
    ok, out_jpg = cv2.imencode(".jpg", annotated, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
    if not ok:
        return JSONResponse({"error": "Failed to encode output image"}, status_code=500)

    # 5) Upload Cloudinary
    try:
        upload_res = cloudinary.uploader.upload(
            out_jpg.tobytes(),
            resource_type="image",
            folder="hauitrash/annotated",
            public_id=f"req_{req.request_id}" if req.request_id else None,
            overwrite=True if req.request_id else False
        )
        annotated_url = upload_res.get("secure_url")
    except Exception as e:
        return JSONResponse({"error": f"Cloudinary upload failed: {str(e)}"}, status_code=500)

    # 6) Parse detections: label/labelDisplay/confidence + annotatedUrl
    detections: List[Dict[str, Any]] = []
    conf_list: List[float] = []

    if r0.boxes is not None and len(r0.boxes) > 0:
        names = r0.names  # dict: class_id -> class_name

        for b in r0.boxes:
            cls_id = int(b.cls.item())
            label_raw = names.get(cls_id, str(cls_id))
            conf_score = float(b.conf.item())

            conf_list.append(conf_score)
            detections.append({
                "label": label_raw,
                "confidence": round(conf_score, 4),
                "annotatedUrl": annotated_url,
            })

    # 7) confidenceAvg
    confidence_avg = round(sum(conf_list) / len(conf_list), 4) if conf_list else 0.0

    return {
        "requestId": req.request_id,
        "imageUrl": str(req.image_url),
        "annotatedUrl": annotated_url,          #  link ảnh đã vẽ box
        "params": {"conf": req.conf, "iou": req.iou},
        "count": len(detections),
        # "confidenceAvg": confidence_avg,
        "detections": detections
    }
