import os
import sys
from pathlib import Path

import torch

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from trashsmart.shared.constants import IMAGENET_MEAN, IMAGENET_STD

# Re-export for convenience
__all__ = [
    "IMAGENET_MEAN",
    "IMAGENET_STD",
    "DEVICE",
    "CLASS_NAMES",
    "NUM_CLASSES",
    "RESNET_MODEL_PATH",
    "RESNET_INPUT_SIZE",
    "RESNET_CONF_THRESHOLD",
]

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = PROJECT_ROOT / "trashsmart" / "models"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CLASS_NAMES = os.getenv("TRASHSMART_CLASSES", "glass,metal,paper,plastic").split(",")
CLASS_NAMES = [name.strip() for name in CLASS_NAMES if name.strip()]
NUM_CLASSES = len(CLASS_NAMES)







RESNET_MODEL_PATH = Path(os.getenv("RESNET_MODEL_PATH", MODELS_DIR / "best_resnet50_model1.pth"))








RESNET_INPUT_SIZE = int(os.getenv("RESNET_INPUT_SIZE", "224"))


RESNET_CONF_THRESHOLD = float(os.getenv("RESNET_CONF_THRESHOLD", "0.5"))


