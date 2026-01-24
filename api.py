from fastapi import FastAPI, UploadFile, File
from fastapi.responses import Response, JSONResponse
from ultralytics import YOLO
import numpy as np
import cv2

app = FastAPI(title="Trash Detection API")

# ✅ Load model 1 lần khi khởi động server
MODEL_PATH = "runs/detect/huy3/weights/best.pt"  # sửa đúng đường dẫn của bạn
model = YOLO(MODEL_PATH)

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict-image")
async def predict_image(file: UploadFile = File(...), conf: float = 0.25, iou: float = 0.6):
    # 1) Read bytes -> np array -> BGR image
    img_bytes = await file.read()
    np_arr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    if img is None:
        return JSONResponse({"error": "Invalid image"}, status_code=400)

    # 2) Predict
    results = model.predict(img, conf=conf, iou=iou, verbose=False)

    # 3) Vẽ box + label lên ảnh
    annotated = results[0].plot()  # trả về ảnh BGR đã vẽ

    # 4) Encode ảnh -> bytes để trả response
    ok, out_jpg = cv2.imencode(".jpg", annotated)
    if not ok:
        return JSONResponse({"error": "Failed to encode output image"}, status_code=500)

    return Response(content=out_jpg.tobytes(), media_type="image/jpeg")
