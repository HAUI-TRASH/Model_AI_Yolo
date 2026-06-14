FROM python:3.10-slim

WORKDIR /app

# libglib is required by OpenCV.
RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install CPU torch first to keep the image lighter.
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu

COPY trashsmart/pipeline/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source and model weights.
COPY . .

EXPOSE 8000

CMD ["uvicorn", "trashsmart.pipeline.main:app", "--host", "0.0.0.0", "--port", "8000"]
