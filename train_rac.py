from ultralytics import YOLO

model = YOLO("yolo11n.pt")
results = model.train(data="dataset/data.yaml", epochs=100, imgsz=640, name="thao")

# results = model.predict(
#     source="battery74.jpg",        # webcam
#     conf=0.3,
#     iou=0.6,
#     show=True
# )