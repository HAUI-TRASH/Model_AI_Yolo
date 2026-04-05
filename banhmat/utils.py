# utils.py  (updated – tích hợp Real-ESRGAN)
import cv2
import numpy as np
from PIL import Image, ImageFilter
import torch
from torchvision import transforms
from config import (
    DEVICE, RESNET_RESIZE, RESNET_INPUT_SIZE,
    IMAGENET_MEAN, IMAGENET_STD,
    FIX_SMALL_CROP, FIX_LOW_LIGHT, FIX_MOTION_BLUR, FIX_RESIZE_BLUR,
    FIX_ESRGAN,                        # ← thêm mới
    SMALL_CROP_THRESHOLD, LOW_LIGHT_BRIGHTNESS_THRESHOLD,
    BLUR_LAPLACIAN_THRESHOLD,
    ESRGAN_SMART_MODE,                 # ← thêm mới
)
from esrgan_enhancer import upscale_with_esrgan, smart_esrgan_enhance


# ========================
# Enhancement Functions
# ========================

def fix_small_crop(image):
    """(1) Upscale crop nhỏ bằng LANCZOS4."""
    h, w = image.shape[:2]
    if min(h, w) < SMALL_CROP_THRESHOLD:
        scale = SMALL_CROP_THRESHOLD / min(h, w)
        image = cv2.resize(image,
                           (int(w * scale), int(h * scale)),
                           interpolation=cv2.INTER_LANCZOS4)
    return image


def fix_low_light(image):
    """(2) Gamma correction + CLAHE nếu ảnh tối."""
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    if np.mean(l) < LOW_LIGHT_BRIGHTNESS_THRESHOLD:
        # Gamma
        table = np.array([(i / 255.0) ** (1.0 / 1.5) * 255
                          for i in range(256)]).astype("uint8")
        l = cv2.LUT(l, table)
        # CLAHE
        l = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(l)
        image = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)
    return image


def fix_motion_blur(image):
    """(3) Unsharp mask nếu ảnh mờ (Laplacian variance thấp)."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if cv2.Laplacian(gray, cv2.CV_64F).var() < BLUR_LAPLACIAN_THRESHOLD:
        gaussian = cv2.GaussianBlur(image, (0, 0), sigmaX=1.0)
        image = cv2.addWeighted(image, 2.5, gaussian, -1.5, 0)
    return image


def fix_resize_blur(pil_image, size):
    """(4) LANCZOS resize + nhẹ unsharp mask."""
    w, h = pil_image.size
    new_h, new_w = (size, int(w * size / h)) if h < w else (int(h * size / w), size)
    pil_image = pil_image.resize((new_w, new_h), Image.LANCZOS)
    return pil_image.filter(
        ImageFilter.UnsharpMask(radius=0.5, percent=50, threshold=0))


# ========================
# enhance_crop – Pipeline chính
# ========================

def enhance_crop(image: np.ndarray) -> np.ndarray:
    """
    Chạy tuần tự các bước enhance trước khi đưa vào ResNet.

    Thứ tự ưu tiên:
        FIX_ESRGAN = True   → Real-ESRGAN (thay thế fix_small_crop
                               và fix_motion_blur, mạnh hơn nhiều)
        FIX_ESRGAN = False  → pipeline cũ (fix_small_crop, fix_low_light,
                               fix_motion_blur)

    Input / Output: numpy BGR (OpenCV format)
    """
    # --- Bước ESRGAN (nếu bật) ---
    if FIX_ESRGAN:
        if ESRGAN_SMART_MODE:
            # Chỉ ESRGAN khi ảnh nhỏ HOẶC mờ → tiết kiệm thời gian
            image = smart_esrgan_enhance(
                image,
                size_threshold=SMALL_CROP_THRESHOLD,
                blur_threshold=BLUR_LAPLACIAN_THRESHOLD,
            )
        else:
            # Luôn chạy ESRGAN với mọi crop
            image = upscale_with_esrgan(image)

        # Vẫn chạy fix_low_light sau ESRGAN (ESRGAN không sửa độ sáng)
        if FIX_LOW_LIGHT:
            image = fix_low_light(image)

    else:
        # --- Pipeline cũ ---
        if FIX_SMALL_CROP:
            image = fix_small_crop(image)
        if FIX_LOW_LIGHT:
            image = fix_low_light(image)
        if FIX_MOTION_BLUR:
            image = fix_motion_blur(image)

    return image


# ========================
# ResNet Transform
# ========================

def build_resnet_transform():
    if FIX_RESIZE_BLUR:
        return transforms.Compose([
            transforms.Lambda(lambda img: fix_resize_blur(img, RESNET_RESIZE)),
            transforms.CenterCrop(RESNET_INPUT_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])
    return transforms.Compose([
        transforms.Resize(RESNET_RESIZE),
        transforms.CenterCrop(RESNET_INPUT_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


resnet_transform = build_resnet_transform()


def preprocess_for_resnet(image_crop: np.ndarray) -> torch.Tensor:
    pil_image = Image.fromarray(cv2.cvtColor(image_crop, cv2.COLOR_BGR2RGB))
    return resnet_transform(pil_image).unsqueeze(0).to(DEVICE)


def get_class_label(prediction: torch.Tensor, class_names: list):
    probs = torch.nn.functional.softmax(prediction, dim=1)
    confidence, predicted_idx = torch.max(probs, 1)
    return class_names[predicted_idx.item()], confidence.item()