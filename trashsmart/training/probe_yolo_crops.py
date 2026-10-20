"""Small exploratory probe: material folder labels inherited by a single YOLO crop."""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
import torch
from sklearn.metrics import accuracy_score, classification_report
from ultralytics import YOLO

from trashsmart.pipeline.config import CLASS_NAMES, CONF_THRESHOLD, YOLO_IMG_SIZE
from trashsmart.shared.paths import YOLO_MODEL_PATH
from trashsmart.shared.model import load_resnet_checkpoint
from trashsmart.training.config import BEST_MODEL_PATH, DEVICE, BATCH_SIZE
from trashsmart.training.compare_preprocessing import build_modes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    args = parser.parse_args()
    materials = {"giay": "paper", "kim loai": "metal", "nhua": "plastic", "thuy tinh": "glass"}
    detector = YOLO(str(YOLO_MODEL_PATH))
    classifier = load_resnet_checkpoint(BEST_MODEL_PATH, len(CLASS_NAMES), DEVICE)
    crops, labels, paths, skipped = [], [], [], []
    for folder, material in materials.items():
        for path in sorted((args.data / folder).iterdir()):
            if path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".webp"}:
                continue
            with Image.open(path) as image:
                bgr = cv2.cvtColor(np.asarray(image.convert("RGB")), cv2.COLOR_RGB2BGR)
            result = detector.predict(bgr, imgsz=YOLO_IMG_SIZE, conf=CONF_THRESHOLD,
                                      iou=0.6, device=str(DEVICE), verbose=False)[0]
            if result.boxes is None or len(result.boxes) != 1:
                skipped.append({"path": str(path), "reason": "not exactly one detection"})
                continue
            x1, y1, x2, y2 = map(int, result.boxes.xyxy[0].tolist())
            crop = bgr[max(0, y1):min(bgr.shape[0], y2), max(0, x1):min(bgr.shape[1], x2)]
            if crop.size == 0:
                skipped.append({"path": str(path), "reason": "empty crop"})
                continue
            crops.append(Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)))
            labels.append(CLASS_NAMES.index(material))
            paths.append(str(path))
    if not crops:
        raise ValueError("No single-object crops found")
    report = {"dataset": str(args.data), "samples": len(crops), "skipped": skipped,
              "label_source": "Inherited material folder; boxes not manually verified",
              "limitation": "Small class-imbalanced probe, conditional on exactly one detection",
              "modes": {}}
    print(f"Probe crops: {len(crops)}, skipped: {len(skipped)}", flush=True)
    for name, transform in build_modes().items():
        predicted = []
        with torch.inference_mode():
            for start in range(0, len(crops), BATCH_SIZE):
                batch = torch.stack([transform(image) for image in crops[start:start + BATCH_SIZE]])
                predicted.extend(classifier(batch.to(DEVICE)).argmax(1).cpu().tolist())
        metrics = classification_report(labels, predicted, labels=list(range(len(CLASS_NAMES))),
                                        target_names=CLASS_NAMES, output_dict=True, zero_division=0)
        accuracy = accuracy_score(labels, predicted)
        report["modes"][name] = {"accuracy": accuracy, "report": metrics,
                                  "predictions": [{"path": p, "true": CLASS_NAMES[t],
                                                   "predicted": CLASS_NAMES[y]}
                                                  for p, t, y in zip(paths, labels, predicted)]}
        print(f"{name}: accuracy={accuracy:.2%}, F1={metrics['weighted avg']['f1-score']:.2%}", flush=True)
    output = Path(__file__).resolve().parents[2] / "outputs" / "resnet_yolo_crop_probe.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Details: {output}", flush=True)


if __name__ == "__main__":
    main()
