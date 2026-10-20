FROM python:3.10-slim AS runtime

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

# Install the resolved CPU runtime, including pinned transitive dependencies.
COPY requirements-runtime-linux-py310.lock .
RUN python -m pip install --no-cache-dir --upgrade pip \
    && python -m pip install --no-cache-dir --no-deps -r requirements-runtime-linux-py310.lock \
    && python -m pip check

# Copy source + model weights (best_new.pt, *.pth đã được commit trong repo)
COPY trashsmart/ ./trashsmart/
COPY api.py apiv2.py app.py ./

ENV YOLO_AUTOINSTALL=false

EXPOSE 8000

CMD ["uvicorn", "trashsmart.pipeline.main:app", "--host", "0.0.0.0", "--port", "8000"]

FROM runtime AS test
COPY requirements-test-linux-py310.lock .
RUN python -m pip install --no-cache-dir --no-deps -r requirements-test-linux-py310.lock \
    && python -m pip check
COPY tests/ ./tests/
RUN python -m unittest discover -s tests -q

FROM runtime AS production
