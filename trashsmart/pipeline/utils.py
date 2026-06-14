
import logging
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from trashsmart.pipeline.config import (
    DEVICE,
    IMAGENET_MEAN,
    IMAGENET_STD,
    RESNET_CONF_THRESHOLD,
    RESNET_INPUT_SIZE,
)


logger = logging.getLogger(__name__)


def _build_resnet_transform() -> transforms.Compose:
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


_resnet_transform = _build_resnet_transform()


def preprocess_for_resnet(image_crop: np.ndarray) -> torch.Tensor:
    image_crop = cv2.resize(
        image_crop,
        (RESNET_INPUT_SIZE, RESNET_INPUT_SIZE),
        interpolation=cv2.INTER_AREA,
    )
    pil_image = Image.fromarray(cv2.cvtColor(image_crop, cv2.COLOR_BGR2RGB))
    return _resnet_transform(pil_image).unsqueeze(0).to(DEVICE)


def get_class_label(
    prediction: torch.Tensor,
    class_names: list,
    conf_threshold: float = RESNET_CONF_THRESHOLD,
) -> tuple[str, float]:
    probs = torch.nn.functional.softmax(prediction, dim=1)
    confidence, predicted_idx = torch.max(probs, 1)

    conf_val = confidence.item()
    label = class_names[predicted_idx.item()]

    if conf_val < conf_threshold:
        logger.debug("[ResNet] Low confidence %.2f for %s -> unknown", conf_val, label)
        return "unknown", conf_val

    return label, conf_val
