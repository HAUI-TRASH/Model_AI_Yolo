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

## Tiền xử lý ResNet dùng chung

Validation, lệnh `python -m trashsmart.training.evaluate`, API và ROI/webcam
dùng `shared/preprocessing.py`: tăng sáng/làm nét theo cấu hình, letterbox
giữ toàn ảnh, BGR -> RGB, tensor và chuẩn hóa ImageNet. Dataset validation
giữ tensor trên CPU để tương thích DataLoader worker Windows. API chuyển
tensor sang thiết bị suy luận sau bước tiền xử lý. Augmentation training và
các biến thể TTA vẫn riêng; TTA kết thúc bằng cùng tiền xử lý này.

Chỉ số ResNet vẫn đo trên ảnh có nhãn, không bao gồm lỗi phát hiện/cắt crop
của YOLO. Vì validation trước đây dùng CenterCrop, cần đo lại checkpoint cũ
bằng cách mới để so sánh công bằng với checkpoint training sau thay đổi.

```powershell
python -m unittest discover -s tests -p "test_resnet_preprocessing.py" -v
```

So sánh ba phương án trên cùng split validation và checkpoint (không sửa cấu hình):

```powershell
python -m trashsmart.training.compare_preprocessing
```

Kết quả và dự đoán từng ảnh lưu tại `outputs/resnet_preprocessing_comparison.json`.
Thăm dò crop YOLO trên bộ ảnh chai có thư mục `giay`, `kim loai`, `nhua`, `thuy tinh`:

```powershell
python -m trashsmart.training.probe_yolo_crops --data D:/thacsi/DOAN/HAUI-TRASH/data/dataset_new_chai
```

Chỉ lấy ảnh có đúng một detection, kế thừa nhãn chất liệu từ thư mục. Kết quả
ở `outputs/resnet_yolo_crop_probe.json` là thăm dò, không phải benchmark end-to-end:
chưa kiểm chứng box thủ công, chưa đo ảnh bị bỏ qua và bộ dữ liệu nhỏ, lệch lớp.

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

API: `/live`, `/health`, `/predict-image`, `/predict-image-json`, `/predict-image-url`.
Ảnh kết quả upload lên Cloudinary: cần cấu hình Cloudinary trong `.env` hoặc
môi trường server. `/live` kiểm tra ứng dụng; `/health` nạp/kiểm tra model và
trả 503 nếu không nạp được. Health chưa gọi Cloudinary. `conf`/`iou` của
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

Training dừng trước khi xây dựng model nếu thứ tự lớp dataset không khớp
`TRASHSMART_CLASSES`. Early stopping đếm cả loss không đổi; chỉ cải thiện lớn
hơn `min_delta` mới đặt lại bộ đếm. `needsConfirm` và nhãn `unknown` cùng dùng
`RESNET_CONF_THRESHOLD`.

Nếu loss/accuracy là NaN hoặc Infinity, training báo lỗi trước khi cập nhật
history, checkpoint hoặc early stopping. Accuracy phải nằm trong [0, 1].
Loss từng batch được kiểm tra trước backward/optimizer step (training) và
trước cộng dồn (validation). Checkpoint hợp lệ đã lưu trước đó được giữ nguyên.

Checkpoint ResNet mới là dictionary gồm `format_version`, `architecture`,
`class_names`, `num_classes`, `val_accuracy`, `model_state_dict`. Khi nạp,
API và công cụ đánh giá đối chiếu metadata với cấu hình, từ chối sai thứ tự
lớp dù số lớp bằng nhau. Định dạng này vẫn đọc được bằng `weights_only=True`.

Checkpoint cũ chỉ có state_dict không thể tự chứng minh thứ tự nhãn. Loader
giữ tương thích theo thứ tự lịch sử của dự án: glass, metal, paper, plastic,
ghi warning và từ chối dùng khi cấu hình khác thứ tự này. Checkpoint cũ không
bị tự động sửa hoặc gán metadata phỏng đoán; lần training tiếp theo sẽ tạo
checkpoint có metadata. Khi dùng checkpoint cũ từ nguồn khác, phải xác minh
thứ tự nhãn gốc trước.

Các endpoint upload chạy đồng bộ trong thread pool FastAPI để không chặn
event loop. Nạp model và suy luận dùng khóa riêng bảo vệ model dùng chung.
Webcam ghép box theo IoU giảm dần, mỗi box chỉ ghép một lần và chỉ cập nhật
bộ ổn định khi có kết quả suy luận mới. Đây là ổn định box, chưa phải tracking
danh tính khi vật thể che khuất hoặc cắt ngang nhau.

Dependency trực tiếp và gián tiếp được resolve/khóa riêng cho CPU:
`requirements-runtime-linux-py310.lock` (Docker),
`requirements-test-linux-py310.lock` (test cùng runtime Linux),
`requirements-test-windows-py314.lock` (training/test Windows).
Lock Linux test giữ các phiên bản runtime qua constraints. Khi sửa requirement,
cần resolve lại lock bằng uv trước khi build, không chỉ sửa file requirement gốc.
Giữ `--extra-index-url https://download.pytorch.org/whl/cpu` trong lock để pip tìm
đúng wheel `+cpu`. Các lock này không dành cho training CUDA.

```powershell
python -m pip install --no-deps -r requirements-test-windows-py314.lock
python -m unittest discover -s tests -v
```

Kiểm thử API dùng model/Cloudinary giả; kiểm thử tiền xử lý dùng tensor thật.
Không mở camera thật, training hoặc gửi ảnh lên Cloudinary trong các test.

## Giới hạn ảnh và tải URL

API chỉ tải HTTP/HTTPS ở cổng chuẩn, không có credentials trong URL. Tất cả
IP DNS trả về phải là public; mỗi redirect được kiểm tra lại (tối đa 3).
Kết nối ghim vào IP đã kiểm tra để tránh DNS rebinding; HTTPS vẫn xác minh
certificate theo hostname gốc và gửi SNI. Không dùng proxy từ môi trường.
Có thể đặt `IMAGE_URL_ALLOWED_HOSTS=res.cloudinary.com` để chỉ cho phép các
hostname cụ thể (phân cách dấu phẩy). URL ảnh nội bộ sẽ bị từ chối.

Mặc định ảnh tối đa 10 MiB, 12 triệu pixel, mỗi cạnh tối đa 8192 pixel. Header
ảnh được kiểm tra trước OpenCV decode; định dạng hỗ trợ JPEG/PNG/WEBP/BMP.
Body HTTP được giới hạn trước multipart parsing kể cả khi không có
Content-Length. Multipart có thêm 1 MiB cho phần bao và trường dữ liệu.
Upload và tải URL đều trả 413 khi vượt giới hạn. HTTP gzip không được nhận.

```dotenv
MAX_IMAGE_BYTES=10485760
MAX_IMAGE_PIXELS=12000000
MAX_IMAGE_SIDE=8192
MAX_CONCURRENT_PREDICTIONS=2
```

Giới hạn đồng thời áp dụng trên mỗi worker, trước khi đọc body và vào thread
pool; quá tải trả 503 cùng Retry-After: 1. `/live` chạy async độc lập với pool;
`/health` không bị tính vào quota dự đoán và có limiter riêng để nạp model,
không dùng capacity của pool dự đoán. Cloudinary upload có timeout 15 giây.
Thư viện không tự cài dependency lúc inference (`YOLO_AUTOINSTALL=false`).

## Kiểm tra trước triển khai

Docker có target `test` và `production`. CI phải build/chạy test Linux và
nạp model thật trong container thành công trước khi push image/triển khai.
Chưa chạy được Docker trên máy không có Docker CLI; lần chạy CI mới xác nhận
được phần Linux thực tế. Có thể kiểm tra thủ công trên máy có Docker:

```powershell
docker build --target test -t trashsmart-test .
docker build --target production -t trashsmart-api .
docker run --rm trashsmart-api python -c "from trashsmart.pipeline.main import get_models; get_models()"
```

Khi commit cần đưa cả module mới, tests và các lock vào cùng thay đổi. Không
commit `.env`, thư mục `.tools`, output thử nghiệm hoặc checkpoint backup.

