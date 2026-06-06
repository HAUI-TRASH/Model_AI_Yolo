FROM python:3.10-slim

WORKDIR /app

# libglib cần cho OpenCV (đã dùng opencv-python-headless nên không cần libgl).
# Nếu bật lại ESRGAN (cài từ GitHub) thì thêm "git" vào danh sách dưới.
RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Cài torch bản CPU trước (nhẹ hơn nhiều so với bản CUDA mặc định)
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu

COPY trashsmart/pipeline/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source + model weights (best_new.pt, *.pth đã được commit trong repo)
COPY . .

EXPOSE 8000

CMD ["uvicorn", "trashsmart.pipeline.main:app", "--host", "0.0.0.0", "--port", "8000"]
