"""Evaluate the same best_new.pt checkpoint used by the API."""
import argparse
from ultralytics import YOLO
from trashsmart.shared.paths import YOLO_MODEL_PATH
from trashsmart.yolo_training.config import DATA_YAML_PATH, IMG_SIZE, BATCH_SIZE, DEVICE


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=("val", "test"), default="val")
    args = parser.parse_args()
    if not YOLO_MODEL_PATH.is_file():
        raise FileNotFoundError(YOLO_MODEL_PATH)
    if not DATA_YAML_PATH.is_file():
        raise FileNotFoundError(DATA_YAML_PATH)
    print(f"Evaluating: {YOLO_MODEL_PATH}")
    metrics = YOLO(str(YOLO_MODEL_PATH)).val(
        data=str(DATA_YAML_PATH), imgsz=IMG_SIZE, batch=BATCH_SIZE,
        device=DEVICE, single_cls=True, split=args.split,
    )
    print(f"Precision: {metrics.box.mp:.4%}")
    print(f"Recall: {metrics.box.mr:.4%}")
    print(f"mAP50: {metrics.box.map50:.4%}")
    print(f"mAP50-95: {metrics.box.map:.4%}")


if __name__ == "__main__":
    main()
