from ultralytics import YOLO

model = YOLO("yolo11n.pt")
# results = model.train(data="dataset/data.yaml", epochs=100, imgsz=640, name="thao")
result = YOLO("runs/detect/yolo_model/weights/best.pt")
results = model.predict(
    source="test.jpg",        # webcam
    conf=0.3,
    iou=0.6,
    show=True
)
for r in results:
    boxes = r.boxes
    for box in boxes:
        cls_id = int(box.cls[0])           # class id
        conf = float(box.conf[0])          # confidence
        xyxy = box.xyxy[0].tolist()        # tọa độ bbox

        class_name = model.names[cls_id]   # tên class

        print(f"Detected: {class_name} | Conf: {conf:.2f} | Box: {xyxy}")