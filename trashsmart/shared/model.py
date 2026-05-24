import logging
from pathlib import Path
from typing import Union

import torch
import torch.nn as nn
from torchvision import models

logger = logging.getLogger(__name__)


class TrashResNet50(nn.Module):
    """ResNet50 classifier shared by training and inference."""

    def __init__(self, num_classes: int, pretrained: bool = True):
        super().__init__()
        self._freeze_backbone_bn = True

        weights = models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        resnet = models.resnet50(weights=weights)

        # Freeze TOÀN BỘ base model trước khi tạo self.base_model
        for param in resnet.parameters():
            param.requires_grad = False
        logger.info("[TrashResNet50] Backbone frozen.")

        num_features = resnet.fc.in_features
        self.base_model = nn.Sequential(*list(resnet.children())[:-1])
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(num_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

    def train(self, mode: bool = True):
        super().train(mode)
        if mode and self._freeze_backbone_bn:
            self._set_backbone_batchnorm_eval()
        return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self._validate_input(x)
        x = self.base_model(x)
        return self.classifier(x)

    def unfreeze_base_model(self, freeze_bn: bool = True, clear_gradients: bool = True) -> None:
        self._freeze_backbone_bn = freeze_bn

        if len(self.base_model) <= 7:
            raise RuntimeError("Unexpected ResNet backbone layout; cannot unfreeze layer3/layer4.")

        for layer in (self.base_model[6], self.base_model[7]):
            for param in layer.parameters():
                param.requires_grad = True
                if clear_gradients:
                    param.grad = None

        if freeze_bn:
            self._set_backbone_batchnorm_eval()

    def _set_backbone_batchnorm_eval(self) -> None:
        for module in self.base_model.modules():
            if isinstance(module, nn.BatchNorm2d):
                module.eval()

    @staticmethod
    def _validate_input(x: torch.Tensor) -> None:
        if not isinstance(x, torch.Tensor):
            raise TypeError(f"Expected input to be torch.Tensor, got {type(x).__name__}.")
        if x.ndim != 4:
            raise ValueError(f"Expected input shape [batch, 3, height, width], got {tuple(x.shape)}.")
        if x.size(1) != 3:
            raise ValueError(f"Expected 3 input channels, got {x.size(1)}.")
        if x.size(2) < 32 or x.size(3) < 32:
            raise ValueError(
                "Expected input height and width to be at least 32 pixels, "
                f"got {x.size(2)}x{x.size(3)}."
            )


def build_trash_resnet50(num_classes: int, device: torch.device, pretrained: bool = True) -> TrashResNet50:
    return TrashResNet50(num_classes=num_classes, pretrained=pretrained).to(device)


def load_resnet_checkpoint(
    model_path: Union[str, Path],
    num_classes: int,
    device: torch.device,
    pretrained: bool = False,
) -> TrashResNet50:
    model = build_trash_resnet50(num_classes=num_classes, device=device, pretrained=pretrained)
    checkpoint = load_checkpoint(model_path=model_path, device=device)

    if isinstance(checkpoint, nn.Module):
        state_dict = checkpoint.state_dict()
    elif isinstance(checkpoint, dict):
        state_dict = checkpoint
        for key in ("model_state_dict", "state_dict", "model"):
            if key in checkpoint:
                state_dict = checkpoint[key].state_dict() if isinstance(checkpoint[key], nn.Module) else checkpoint[key]
                break
    else:
        raise TypeError(f"Unsupported checkpoint type: {type(checkpoint).__name__}")

    cleaned_state_dict = {
        key[7:] if key.startswith("module.") else key: value
        for key, value in state_dict.items()
    }

    model.load_state_dict(cleaned_state_dict)
    model.eval()
    return model


def load_checkpoint(
    model_path: Union[str, Path],
    device: torch.device,
) -> dict:
    model_path = Path(model_path)

    if not model_path.exists():
        raise FileNotFoundError(f"[load_checkpoint] Khong tim thay file checkpoint: {model_path}")

    if model_path.suffix not in {".pth", ".pt"}:
        logger.warning(
            "[load_checkpoint] Extension khong phai .pth/.pt: %s - van thu load.",
            model_path.suffix,
        )

    logger.info("[load_checkpoint] Loading: %s -> device=%s", model_path, device)

    first_error_message = None
    try:
        checkpoint = torch.load(model_path, map_location=device, weights_only=True)
        logger.info("[load_checkpoint] Loaded thanh cong (weights_only=True).")
        return checkpoint
    except Exception as first_error:
        first_error_message = repr(first_error)
        logger.warning(
            "[load_checkpoint] weights_only=True that bai (%s). "
            "Fallback ve weights_only=False - chi load file tin cay!",
            first_error_message,
        )

    try:
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        logger.info("[load_checkpoint] Loaded thanh cong (weights_only=False).")
        return checkpoint
    except Exception as second_error:
        raise RuntimeError(
            f"[load_checkpoint] Khong the load checkpoint: {model_path}\n"
            f"  Loi weights_only=True : {first_error_message}\n"
            f"  Loi weights_only=False: {second_error!r}"
        ) from second_error
