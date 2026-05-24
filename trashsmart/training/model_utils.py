import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from shared.model import TrashResNet50, build_trash_resnet50
from training.config import DEVICE, NUM_CLASSES




def build_model():
    return build_trash_resnet50(num_classes=NUM_CLASSES, device=DEVICE, pretrained=True)


def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total:,}")
    print(f"Trainable parameters: {trainable:,}")
    return total, trainable
