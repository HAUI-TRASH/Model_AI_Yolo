import os
from pathlib import Path

import torch
from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(PROJECT_ROOT / ".env")


ROBOFLOW_KEY = os.getenv("ROBOFLOW_KEY")
ROBOFLOW_URL = (
    os.getenv("ROBOFLOW_URL")
    or (f"https://universe.roboflow.com/ds/iZRb61TtaQ?key={ROBOFLOW_KEY}" if ROBOFLOW_KEY else None)
)

DATASET_DIR = Path(os.getenv("YOLO_DATASET_DIR", PROJECT_ROOT / "datasets" / "HauiTrash-2"))
DATA_YAML_PATH = Path(os.getenv("YOLO_DATA_YAML", DATASET_DIR / "data.yaml"))

MODEL_PRETRAINED = os.getenv("YOLO_PRETRAINED", "yolo11m.pt")
EPOCHS = int(os.getenv("YOLO_EPOCHS", "100"))
BATCH_SIZE = int(os.getenv("YOLO_BATCH_SIZE", "8"))
IMG_SIZE = int(os.getenv("YOLO_IMG_SIZE", "640"))
PATIENCE = int(os.getenv("YOLO_PATIENCE", "20"))
DEVICE = "0" if torch.cuda.is_available() else "cpu"

PROJECT_NAME = os.getenv("YOLO_PROJECT_NAME", str(PROJECT_ROOT / "runs" / "waste_detection_yolo11_roboflow"))
RUN_NAME = os.getenv("YOLO_RUN_NAME", "run_640")
