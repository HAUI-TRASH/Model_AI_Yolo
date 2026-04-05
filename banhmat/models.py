# models.py
from ultralytics import YOLO
import torch
from torchvision import models
import torch.nn as nn
from config import (
    DEVICE, NUM_CLASSES, RESNET_FEATURES,
    RESNET_HIDDEN_1, RESNET_HIDDEN_2, DROPOUT_RATE
)


def load_yolo_model(model_path):
    """
    Load the trained YOLO model.
    :param model_path: Path to the YOLO model weights.
    :return: Loaded YOLO model.
    """
    model = YOLO(model_path)
    model.eval()  # Set to evaluation mode
    return model


def load_resnet_model(model_path, num_classes=NUM_CLASSES):
    """
    Load the trained ResNet model.
    :param model_path: Path to the ResNet model weights.
    :param num_classes: Number of classes in your classification task.
    :return: Loaded ResNet model.
    """
    # Load checkpoint để kiểm tra cấu trúc
    checkpoint = torch.load(model_path, map_location=DEVICE, weights_only=False)

    # Kiểm tra xem có phải là state_dict thuần hay wrapped model
    if 'base_model.0.weight' in checkpoint:
        # Model được wrap với base_model và classifier
        class CustomResNet(nn.Module):
            def __init__(self, num_classes=NUM_CLASSES):
                super(CustomResNet, self).__init__()
                # Load base ResNet50
                resnet = models.resnet50(weights=None)
                # Lấy tất cả layers trừ fc
                self.base_model = nn.Sequential(*list(resnet.children())[:-1])

                # Classifier với cấu trúc tương tự model đã train
                self.classifier = nn.Sequential(
                    nn.Flatten(),
                    nn.Linear(RESNET_FEATURES, RESNET_HIDDEN_1),
                    nn.BatchNorm1d(RESNET_HIDDEN_1),
                    nn.ReLU(),
                    nn.Dropout(DROPOUT_RATE),
                    nn.Linear(RESNET_HIDDEN_1, RESNET_HIDDEN_2),
                    nn.BatchNorm1d(RESNET_HIDDEN_2),
                    nn.ReLU(),
                    nn.Dropout(DROPOUT_RATE),
                    nn.Linear(RESNET_HIDDEN_2, num_classes)
                )

            def forward(self, x):
                x = self.base_model(x)
                x = self.classifier(x)
                return x

        model = CustomResNet(num_classes)
        model.load_state_dict(checkpoint)
    else:
        # Model ResNet chuẩn
        model = models.resnet50(weights=None)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        model.load_state_dict(checkpoint)

    model.to(DEVICE)
    model.eval()  # Set to evaluation mode
    return model