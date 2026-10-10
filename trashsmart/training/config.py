import os
from pathlib import Path

import torch
from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(PROJECT_ROOT / ".env")

PACKAGE_ROOT = PROJECT_ROOT / "trashsmart"
MODELS_DIR = PROJECT_ROOT / "models"

DIR_PATH = Path(os.getenv("DATASET_PATH", PROJECT_ROOT / "dataset_new"))
TARGET_SIZE = (224, 224)
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "32"))
EPOCHS = int(os.getenv("EPOCHS", "100"))
STAGE1_EPOCHS = int(os.getenv("STAGE1_EPOCHS", "20"))
LEARNING_RATE = float(os.getenv("LEARNING_RATE", "0.0001"))
INPUT_SHAPE = (3, 224, 224)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _discover_classes(dataset_dir: Path) -> list[str]:
    if not dataset_dir.exists():
        return []
    return sorted([p.name for p in dataset_dir.iterdir() if p.is_dir()])


class_folders = _discover_classes(DIR_PATH)
NUM_CLASSES = len(class_folders)
WASTE_LABELS = {folder: idx for idx, folder in enumerate(class_folders)}
INDEX_TO_NAME = {v: k for k, v in WASTE_LABELS.items()}

MODEL_SAVE_PATH = MODELS_DIR / "resnet50_optimized_model1.pth"
HISTORY_SAVE_PATH = MODELS_DIR / "resnet50_training_history1.pkl"
BEST_MODEL_PATH = MODELS_DIR / "best_resnet50_model1.pth"
