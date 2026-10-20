"""Canonical CPU preprocessing for PIL RGB images and OpenCV BGR crops."""
import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from trashsmart.shared.constants import IMAGENET_MEAN, IMAGENET_STD
from trashsmart.shared.resnet_config import (
    RESNET_INPUT_SIZE, FIX_LOW_LIGHT, FIX_MOTION_BLUR,
    LOW_LIGHT_BRIGHTNESS_THRESHOLD, BLUR_LAPLACIAN_THRESHOLD,
)

def fix_low_light(image: np.ndarray) -> np.ndarray:
    """
    Cải thiện độ sáng bằng Gamma Correction + CLAHE.
    Chỉ áp dụng khi ảnh thực sự tối (mean L < threshold).
    """
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    if np.mean(l) < LOW_LIGHT_BRIGHTNESS_THRESHOLD:
        # Gamma correction (gamma = 1/1.5 → làm sáng)
        table = np.array(
            [(i / 255.0) ** (1.0 / 1.5) * 255 for i in range(256)],
            dtype=np.uint8,
        )
        l = cv2.LUT(l, table)
        # CLAHE để cân bằng tương phản cục bộ
        l = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(l)
        image = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)

    return image


def fix_motion_blur(image: np.ndarray) -> np.ndarray:
    """
    Làm nét nhẹ bằng Unsharp Mask nếu ảnh bị mờ.
    Chỉ áp dụng khi Laplacian variance < threshold.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if cv2.Laplacian(gray, cv2.CV_64F).var() < BLUR_LAPLACIAN_THRESHOLD:
        gaussian = cv2.GaussianBlur(image, (0, 0), sigmaX=1.0)
        image = cv2.addWeighted(image, 2.5, gaussian, -1.5, 0)
    return image


def letterbox_to_square(image: np.ndarray, size: int, pad_color=(255, 255, 255)) -> np.ndarray:
    h, w = image.shape[:2]
    scale = size / max(h, w)
    new_w, new_h = max(1, int(w * scale)), max(1, int(h * scale))
    interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_LANCZOS4
    resized = cv2.resize(image, (new_w, new_h), interpolation=interp)
    canvas = np.full((size, size, 3), pad_color, dtype=np.uint8)
    top = (size - new_h) // 2
    left = (size - new_w) // 2
    canvas[top:top + new_h, left:left + new_w] = resized
    return canvas


def quick_enhance_for_resnet(image: np.ndarray) -> np.ndarray:
    """
    Pipeline xử lý nhanh trước khi đưa vào ResNet.
    Không dùng ESRGAN ở bước này để tối ưu tốc độ.
    """
    if FIX_LOW_LIGHT:
        image = fix_low_light(image)
    if FIX_MOTION_BLUR:
        image = fix_motion_blur(image)
    return image


_normalize = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])


def preprocess_resnet_crop(image_bgr: np.ndarray, input_size: int = RESNET_INPUT_SIZE) -> torch.Tensor:
    """BGR uint8 crop -> enhanced, letterboxed, ImageNet-normalized CPU CHW tensor."""
    if (not isinstance(image_bgr, np.ndarray) or image_bgr.ndim != 3
            or image_bgr.shape[2] != 3 or image_bgr.size == 0):
        raise ValueError("Expected a non-empty BGR image with 3 channels.")
    if input_size <= 0:
        raise ValueError("ResNet input size must be positive.")
    image = quick_enhance_for_resnet(image_bgr.copy())
    image = letterbox_to_square(image, input_size)
    return _normalize(Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)))


class ResNetEvalTransform:
    """Picklable ImageFolder transform; keep tensors on CPU for DataLoader workers."""
    def __init__(self, input_size: int = RESNET_INPUT_SIZE):
        self.input_size = input_size

    def __call__(self, image: Image.Image) -> torch.Tensor:
        image_bgr = cv2.cvtColor(np.asarray(image.convert("RGB")), cv2.COLOR_RGB2BGR)
        return preprocess_resnet_crop(image_bgr, self.input_size)


