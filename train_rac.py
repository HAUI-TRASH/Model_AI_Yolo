from ultralytics import YOLO

model = YOLO("runs/detect/huy3/weights/best.pt")
# results = model.train(data="data_rac/data.yaml", epochs=100, imgsz=640, name="huy")

results = model.predict(
    source="battery74.jpg",        # webcam
    conf=0.3,
    iou=0.6,
    show=True
)