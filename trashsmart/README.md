# TrashSmart — một luồng YOLO best_new.pt

## Model chính thức

YOLO chỉ dùng `trashsmart/models/best_new.pt`. Đường dẫn tuyệt đối định nghĩa
một lần trong `trashsmart/shared/paths.py`, dùng chung cho training, đánh giá,
thử ảnh, webcam và API. `YOLO_PRETRAINED` và `YOLO_MODEL_PATH` cũ không còn
điều khiển model, kể cả khi được đặt trong môi trường server.

ResNet50 dùng checkpoint riêng `best_resnet50_model1.pth`: YOLO định vị chai,
ResNet phân loại glass/metal/paper/plastic. Dataset YOLO hiện có một lớp Bottle,
tương ứng `single_cls=True`; không phải dataset phân loại ResNet.

## Chuẩn bị môi trường

Chạy từ thư mục gốc dự án, dùng Python 3.10:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Nếu Python chưa ở PATH, dùng Python đã cài trên máy này:

```powershell
& "$env:LOCALAPPDATA\Python\pythoncore-3.10-64\python.exe" -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Không cần activate nếu thay `python` trong các lệnh dưới bằng
`.\.venv\Scripts\python.exe`. GPU chỉ được dùng khi PyTorch hỗ trợ CUDA.
Dockerfile cài PyTorch CPU. Real-ESRGAN tắt mặc định (`FIX_ESRGAN=0`).

## Đánh giá model hiện tại

```powershell
python -m trashsmart.yolo_training.evaluate_yolo
python -m trashsmart.yolo_training.evaluate_yolo --split test
```

In đường dẫn model, Precision, Recall, mAP50 và mAP50-95 theo phần trăm.
Dataset lấy từ `YOLO_DATA_YAML` trong `.env`; cần có ảnh và label.
Đánh giá trước và sau training trên cùng tập dữ liệu để so sánh.

## Fine-tune và cập nhật model

```powershell
python -m trashsmart.yolo_training.train_yolo
```

1. Kiểm tra `best_new.pt` và chuẩn bị dataset.
2. Sao lưu model cũ vào `runs/waste_detection_yolo11_roboflow/backups/`.
3. Nạp `best_new.pt`, bắt đầu fine-tune mới (không resume optimizer cũ).
4. Lưu kết quả vào `runs/waste_detection_yolo11_roboflow/best_new_finetune*/`.
5. Đánh giá checkpoint tốt nhất của lần chạy.
6. Nếu training và đánh giá thành công, cập nhật `trashsmart/models/best_new.pt`
   bằng file tạm và replace trên cùng filesystem.

Ultralytics vẫn tạo `weights/best.pt` và `weights/last.pt` bên trong thư mục
training. Đây là artifact của lần chạy. Model chính thức sau bước 6 luôn là
`trashsmart/models/best_new.pt`. Nếu training/đánh giá lỗi, model chính thức
không bị thay thế. Đánh giá thành công không đảm bảo model mới tốt hơn model
cũ; giữ backup để đối chiếu và khôi phục. Chỉ chạy một job training cập nhật
model tại một thời điểm.

Các biến cấu hình training:

```dotenv
YOLO_EPOCHS=100
YOLO_BATCH_SIZE=8
YOLO_IMG_SIZE=640
YOLO_PATIENCE=20
YOLO_LEARNING_RATE=0.0001
YOLO_RUN_NAME=best_new_finetune
```

## Chạy API và demo

```powershell
python -m uvicorn trashsmart.pipeline.main:app --host 0.0.0.0 --port 8000
python -m trashsmart.yolo_training.test_yolo --image sample.jpg --save
python -m trashsmart.pipeline.roi
python -m trashsmart.pipeline.image_roi --image sample.jpg
```

API: `/health`, `/predict-image`, `/predict-image-json`, `/predict-image-url`.
Ảnh kết quả upload lên Cloudinary: cần cấu hình Cloudinary trong `.env` hoặc
môi trường server. `/health` chưa kiểm tra model/Cloudinary. `conf`/`iou` của
API URL được truyền vào YOLO. Endpoint upload dùng cùng pipeline YOLO + ResNet.

Sau training, khởi động lại API để nạp model mới. Với Docker từ GHCR, cần
build/push image có checkpoint mới và triển khai lại; training local không
tự cập nhật container trên server.

`api.py`, `apiv2.py`, `app.py` là alias cho cùng FastAPI. `train_rac.py` gọi
cùng script fine-tune. Demo giao thông Streamlit và API cũ được giữ dưới dạng
`.py.txt` trong `archive/`, không còn là entry point chạy. `models/best.pt`,
`yolo11n.pt` và cấu hình trong `runs/` là file cũ, không dùng làm model đầu vào
cho các entry point hiện tại.

## Training ResNet và kiểm thử

```powershell
python -m trashsmart.training.main
python -m unittest discover -s tests -v
```

ResNet cần `DATASET_PATH` trỏ tới ImageFolder theo chất liệu. Kiểm thử dùng
model giả và thư mục tạm để kiểm tra sao lưu, cập nhật thành công và giữ model
khi training/validation lỗi; không đo accuracy thật.

