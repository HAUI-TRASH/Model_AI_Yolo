import os
import subprocess
import sys
import zipfile
from pathlib import Path

import torch
from ultralytics import YOLO

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from yolo_training.config import (
    BATCH_SIZE,
    DATASET_DIR,
    DATA_YAML_PATH,
    DEVICE,
    EPOCHS,
    IMG_SIZE,
    MODEL_PRETRAINED,
    PATIENCE,
    PROJECT_NAME,
    ROBOFLOW_URL,
    RUN_NAME,
)


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
    print("=" * 70)
    print("TRAIN YOLO11")
    print("=" * 70)
    print(f"Dataset YAML: {DATA_YAML_PATH}")
    print(f"IMG_SIZE: {IMG_SIZE}")
    print("=" * 70)

    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    else:
        print("Training on CPU.")

    download_and_extract_dataset()

    model = YOLO(MODEL_PRETRAINED)
    results = model.train(
        data=str(DATA_YAML_PATH),
        epochs=EPOCHS,
        batch=BATCH_SIZE,
        imgsz=IMG_SIZE,
        patience=PATIENCE,
        device=DEVICE,
        project=PROJECT_NAME,
        name=RUN_NAME,
        exist_ok=True,
        pretrained=True,
        optimizer="AdamW",
        lr0=0.001,
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
    print(f"Best model: {Path(PROJECT_NAME) / RUN_NAME / 'weights' / 'best.pt'}")
    print("=" * 70)

    metrics = model.val()
    print(f"mAP50-95: {metrics.box.map:.4f}")
    print(f"mAP50: {metrics.box.map50:.4f}")
    print(f"mAP75: {metrics.box.map75:.4f}")
    return results


if __name__ == "__main__":
    main()
