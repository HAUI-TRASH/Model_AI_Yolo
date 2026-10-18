"""Canonical YOLO checkpoint shared by training, evaluation and inference."""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
YOLO_MODEL_PATH = PROJECT_ROOT / "trashsmart" / "models" / "best_new.pt"
