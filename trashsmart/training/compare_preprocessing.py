"""Compare three preprocessing methods on the same validation images and checkpoint."""
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from sklearn.metrics import accuracy_score, classification_report
from torchvision import transforms
from torch.utils.data import DataLoader

from trashsmart.shared.constants import IMAGENET_MEAN, IMAGENET_STD
from trashsmart.shared.model import load_resnet_checkpoint
from trashsmart.shared.preprocessing import letterbox_to_square, ResNetEvalTransform
from trashsmart.shared.resnet_config import RESNET_INPUT_SIZE, FIX_LOW_LIGHT, FIX_MOTION_BLUR
from trashsmart.training.config import BEST_MODEL_PATH, BATCH_SIZE, DEVICE, DIR_PATH, NUM_CLASSES
from trashsmart.training.data_utils import get_data_loaders


class LetterboxOnly:
    def __init__(self):
        self.normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])

    def __call__(self, image):
        bgr = cv2.cvtColor(np.asarray(image.convert("RGB")), cv2.COLOR_RGB2BGR)
        square = letterbox_to_square(bgr, RESNET_INPUT_SIZE)
        return self.normalize(Image.fromarray(cv2.cvtColor(square, cv2.COLOR_BGR2RGB)))


def build_modes():
    return {
        "center_crop": transforms.Compose([
            transforms.Resize(256), transforms.CenterCrop(RESNET_INPUT_SIZE),
            transforms.ToTensor(), transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]),
        "letterbox_only": LetterboxOnly(),
        "letterbox_enhanced": ResNetEvalTransform(),
    }


def main():
    _, validation, classes = get_data_loaders()
    subset = validation.dataset
    model = load_resnet_checkpoint(BEST_MODEL_PATH, NUM_CLASSES, DEVICE)
    modes = build_modes()
    result = {
        "checkpoint": str(BEST_MODEL_PATH),
        "checkpoint_sha256": hashlib.sha256(Path(BEST_MODEL_PATH).read_bytes()).hexdigest(),
        "dataset": str(DIR_PATH), "split_seed": 42,
        "samples": len(subset), "classes": classes,
        "input_size": RESNET_INPUT_SIZE,
        "fix_low_light": FIX_LOW_LIGHT, "fix_motion_blur": FIX_MOTION_BLUR,
        "modes": {},
    }
    print(f"Checkpoint: {BEST_MODEL_PATH}; validation: {len(subset)} images", flush=True)
    for name, transform in modes.items():
        subset.dataset.transform = transform
        loader = DataLoader(subset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
        truth, predicted = [], []
        with torch.inference_mode():
            for images, labels in loader:
                predictions = model(images.to(DEVICE)).argmax(dim=1).cpu().tolist()
                truth.extend(labels.tolist())
                predicted.extend(predictions)
        report = classification_report(truth, predicted, labels=list(range(len(classes))),
                                       target_names=classes, output_dict=True, zero_division=0)
        accuracy = accuracy_score(truth, predicted)
        weighted = report["weighted avg"]
        result["modes"][name] = {
            "accuracy": accuracy, "correct": sum(a == b for a, b in zip(truth, predicted)),
            "report": report,
            "predictions": [{"path": subset.dataset.samples[index][0], "true": classes[label],
                             "predicted": classes[pred]}
                            for index, label, pred in zip(subset.indices, truth, predicted)],
        }
        print(f"{name}: Accuracy={accuracy:.2%}, Precision={weighted['precision']:.2%}, "
              f"Recall={weighted['recall']:.2%}, F1={weighted['f1-score']:.2%}", flush=True)
    output = Path(__file__).resolve().parents[2] / "outputs" / "resnet_preprocessing_comparison.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Details: {output}", flush=True)


if __name__ == "__main__":
    main()
