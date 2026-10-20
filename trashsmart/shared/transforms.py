import sys
from pathlib import Path

from torchvision import transforms

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from trashsmart.shared.constants import IMAGENET_MEAN, IMAGENET_STD
from trashsmart.shared.preprocessing import ResNetEvalTransform
from trashsmart.shared.resnet_config import RESNET_INPUT_SIZE


def get_train_transform(input_size: int = 224) -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize(256),
        transforms.RandomRotation(20),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), shear=(-10, 10)),
        transforms.RandomResizedCrop(input_size, scale=(0.8, 1.0)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def get_eval_transform(input_size: int = RESNET_INPUT_SIZE) -> ResNetEvalTransform:
    """Use the same enhancement, letterbox and normalization as the API."""
    return ResNetEvalTransform(input_size=input_size)
