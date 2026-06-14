import logging
import sys
from pathlib import Path


import torch.nn as nn


if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from trashsmart.pipeline.config import DEVICE, NUM_CLASSES
from trashsmart.shared.model import load_resnet_checkpoint


logger = logging.getLogger(__name__)






def load_resnet_model(model_path: str, num_classes: int = NUM_CLASSES) -> nn.Module:
    logger.info(f"[ResNet] Loading checkpoint: {model_path}")
    model = load_resnet_checkpoint(model_path, num_classes=num_classes, device=DEVICE)
    logger.info(f"[ResNet] Model loaded (device={DEVICE})")
    return model
