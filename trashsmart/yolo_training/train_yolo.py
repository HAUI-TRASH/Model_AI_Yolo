import os
import subprocess
import sys
import zipfile
import shutil
import tempfile
from datetime import datetime
from pathlib import Path

import torch
from ultralytics import YOLO

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from trashsmart.yolo_training.config import (
    BATCH_SIZE,
    DATASET_DIR,
    DATA_YAML_PATH,
    DEVICE,
    EPOCHS,
    IMG_SIZE,
    LEARNING_RATE,
    PATIENCE,
    PROJECT_NAME,
    ROBOFLOW_URL,
    RUN_NAME,
)
from trashsmart.shared.paths import YOLO_MODEL_PATH


def download_and_extract_dataset():
    if DATA_YAML_PATH.exists():
        print(f"Dataset already exists: {DATA_YAML_PATH}")
        return

    if not ROBOFLOW_URL:
        raise ValueError(
            "Roboflow URL is not configured. Set ROBOFLOW_KEY or ROBOFLOW_URL, "
            "or set YOLO_DATA_YAML to an existing data.yaml."
        )

    DATASET_DIR.parent.mkdir(parents=True, exist_ok=True)
    zip_path = DATASET_DIR.parent / "roboflow.zip"

    print("Downloading dataset from Roboflow...")
    subprocess.run(["curl", "-L", ROBOFLOW_URL, "-o", str(zip_path)], check=True)

    print("Extracting dataset...")
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(DATASET_DIR.parent)

    os.remove(zip_path)

    if not DATA_YAML_PATH.exists():
        dirs = [p.name for p in DATASET_DIR.parent.iterdir() if p.is_dir()]
        raise FileNotFoundError(
            f"data.yaml not found at {DATA_YAML_PATH}. Existing directories: {dirs}"
        )


def main():
    if not YOLO_MODEL_PATH.is_file():
        raise FileNotFoundError(f"Missing canonical YOLO checkpoint: {YOLO_MODEL_PATH}")
    print("=" * 70)
    print("FINE-TUNE YOLO CHECKPOINT")
    print("=" * 70)
    print(f"Dataset YAML: {DATA_YAML_PATH}")
    print(f"Pretrained model: {YOLO_MODEL_PATH}")
    print(f"IMG_SIZE: {IMG_SIZE}")
    print("=" * 70)

    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    else:
        print("Training on CPU.")

    download_and_extract_dataset()

    backup_dir = Path(PROJECT_NAME) / "backups"
    backup_dir.mkdir(parents=True, exist_ok=True)
    backup_path = backup_dir / f"best_new_{datetime.now():%Y%m%d_%H%M%S_%f}.pt"
    shutil.copy2(YOLO_MODEL_PATH, backup_path)
    print(f"Original checkpoint backup: {backup_path}")

    model = YOLO(str(YOLO_MODEL_PATH))
    results = model.train(
        data=str(DATA_YAML_PATH),
        single_cls=True,
        epochs=EPOCHS,
        batch=BATCH_SIZE,
        imgsz=IMG_SIZE,
        patience=PATIENCE,
        device=DEVICE,
        project=PROJECT_NAME,
        name=RUN_NAME,
        exist_ok=False,
        pretrained=True,
        resume=False,
        optimizer="AdamW",
        lr0=LEARNING_RATE,
        weight_decay=0.0005,
        augment=True,
        cache="ram" if torch.cuda.is_available() else False,
        plots=True,
        save=True,
        iou=0.7,
        conf=0.001,
        box=7.5,
        cls=0.5,
        dfl=1.5,
    )

    print("\n" + "=" * 70)
    print("TRAINING COMPLETED")
    best_model_path = Path(model.trainer.save_dir) / "weights" / "best.pt"
    print(f"Best model: {best_model_path}")
    print("=" * 70)

    metrics = YOLO(str(best_model_path)).val(
        data=str(DATA_YAML_PATH), single_cls=True, imgsz=IMG_SIZE,
        batch=BATCH_SIZE, device=DEVICE,
    )
    print(f"Precision: {metrics.box.mp:.4f}")
    print(f"Recall: {metrics.box.mr:.4f}")
    print(f"mAP50-95: {metrics.box.map:.4f}")
    print(f"mAP50: {metrics.box.map50:.4f}")
    print(f"mAP75: {metrics.box.map75:.4f}")

    # Publish only after training and validation succeed. Keep the active file
    # intact until a complete replacement is ready on the same filesystem.
    fd, temporary_path = tempfile.mkstemp(suffix=".pt", prefix="best_new_", dir=YOLO_MODEL_PATH.parent)
    os.close(fd)
    try:
        shutil.copy2(best_model_path, temporary_path)
        os.replace(temporary_path, YOLO_MODEL_PATH)
    finally:
        if os.path.exists(temporary_path):
            os.remove(temporary_path)
    print(f"Updated API checkpoint: {YOLO_MODEL_PATH}")
    print("Restart the API to load the new checkpoint.")
    return results


if __name__ == "__main__":
    main()
