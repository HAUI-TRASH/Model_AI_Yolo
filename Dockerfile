FROM python:3.10-slim

WORKDIR /app

# Runtime libraries required by the OpenCV wheel installed with Ultralytics.
# Nếu bật lại ESRGAN (cài từ GitHub) thì thêm "git" vào danh sách dưới.
RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libgl1 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libxcb1 \
    && rm -rf /var/lib/apt/lists/*

# Dùng PyPI cho các dependency phụ; PyTorch index chỉ chứa wheel torch CPU.
RUN python -m pip install --no-cache-dir --upgrade pip \
    && python -m pip install --no-cache-dir torch torchvision \
        --index-url https://download.pytorch.org/whl/cpu \
        --extra-index-url https://pypi.org/simple

COPY trashsmart/pipeline/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source + model weights (best_new.pt, *.pth đã được commit trong repo)
COPY . .

EXPOSE 8000

CMD ["uvicorn", "trashsmart.pipeline.main:app", "--host", "0.0.0.0", "--port", "8000"]
