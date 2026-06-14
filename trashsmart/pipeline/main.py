
import logging



import sys
from pathlib import Path
from typing import Dict, Optional

import cv2

import numpy as np

import requests
import torch
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from pydantic import BaseModel, HttpUrl

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from trashsmart.pipeline.config import CLASS_NAMES, NUM_CLASSES, RESNET_MODEL_PATH
from trashsmart.pipeline.models import load_resnet_model
from trashsmart.pipeline.utils import (
    get_class_label,
    preprocess_for_resnet,
)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

app = FastAPI(title="TrashSmart Material Classification API")


_resnet_model = None


class ClassifyUrlRequest(BaseModel):
    image_url: HttpUrl
    request_id: Optional[int] = None


class PredictUrlRequest(ClassifyUrlRequest):
    conf: float = 0.25
    iou: float = 0.6


def get_model():
    global _resnet_model
    if _resnet_model is None:
        _resnet_model = load_resnet_model(RESNET_MODEL_PATH, NUM_CLASSES)
    return _resnet_model


def download_image_to_cv2(url: str) -> np.ndarray:
    response = requests.get(url, timeout=15)
    response.raise_for_status()
    np_arr = np.frombuffer(response.content, np.uint8)
    return cv2.imdecode(np_arr, cv2.IMREAD_COLOR)


def decode_upload_to_cv2(file_bytes: bytes) -> Optional[np.ndarray]:
    np_arr = np.frombuffer(file_bytes, np.uint8)
    return cv2.imdecode(np_arr, cv2.IMREAD_COLOR)


def normalize_material_label(label: str) -> str:
    normalized = str(label or "").strip().lower()
    if normalized == "mental":
        return "metal"
    return normalized


def classify_image(

    image: np.ndarray,

    request_id: Optional[int] = None,
) -> Dict[str, str]:
    if image is None or image.size == 0:
        raise ValueError("Invalid image data")

    model = get_model()
    tensor = preprocess_for_resnet(image)

    with torch.no_grad():
        output = model(tensor)

    raw_label, confidence = get_class_label(output, CLASS_NAMES)
    label = normalize_material_label(raw_label)

    logger.info(
        "Classified request_id=%s as label=%s confidence=%.4f",
        request_id,
        label,
        confidence,
    )

    return {"label": label}


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model": "resnet50",
        "classes": CLASS_NAMES,
    }


@app.post("/classify-image")
async def classify_uploaded_image(
    file: UploadFile = File(...),
    request_id: Optional[int] = None,
):
    img = decode_upload_to_cv2(await file.read())
    if img is None:
        return JSONResponse({"error": "Invalid image"}, status_code=400)


    try:
        return JSONResponse(classify_image(img, request_id=request_id), status_code=200)
    except Exception as exc:
        logger.exception("Classify uploaded image failed")
        return JSONResponse({"error": str(exc)}, status_code=500)


@app.post("/classify-image-url")
def classify_image_url(req: ClassifyUrlRequest):
    try:
        img = download_image_to_cv2(str(req.image_url))
    except Exception as exc:
        return JSONResponse(
            {"error": f"Failed to download/decode image: {exc}"},
            status_code=400,
        )

    if img is None:
        return JSONResponse({"error": "Invalid image data"}, status_code=400)


    try:
        return classify_image(


            img,

            request_id=req.request_id,
        )
    except Exception as exc:
        logger.exception("Classify image url failed")
        return JSONResponse({"error": f"Classify failed: {exc}"}, status_code=500)


@app.post("/predict-image-json")
async def predict_image_json(file: UploadFile = File(...)):
    img = decode_upload_to_cv2(await file.read())
    if img is None:
        return JSONResponse({"error": "Invalid image"}, status_code=400)

    try:
        return JSONResponse(classify_image(img), status_code=200)
    except Exception as exc:
        logger.exception("Legacy predict image json failed")
        return JSONResponse({"error": str(exc)}, status_code=500)


@app.post("/predict-image")
async def predict_image(file: UploadFile = File(...)):
    return await predict_image_json(file)


@app.post("/predict-image-url")
def predict_image_url(req: PredictUrlRequest):
    return classify_image_url(req)


def main():
    logger.info("TrashSmart ResNet50 classifier is ready.")
    logger.info("Classes: %s", ", ".join(CLASS_NAMES))


if __name__ == "__main__":
    main()
