# utils.py
import logging
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from torchvision import transforms

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from trashsmart.pipeline.config import (
    BLUR_LAPLACIAN_THRESHOLD,
    DEVICE,
    ESRGAN_SMART_MODE,
    ESRGAN_TARGET_SIZE,
    FIX_LOW_LIGHT,
    FIX_MOTION_BLUR,
    IMAGENET_MEAN,
    IMAGENET_STD,
    LOW_LIGHT_BRIGHTNESS_THRESHOLD,
    RESNET_CONF_THRESHOLD,
    RESNET_INPUT_SIZE,
)
from trashsmart.pipeline.esrgan_enhancer import smart_esrgan_enhance, upscale_with_esrgan

logger = logging.getLogger(__name__)

LABELS_VI = {
    "chai": "Chai",
    "bottle": "Chai",
    "glass": "thủy tinh",
    "metal": "kim loại",
    "mental": "kim loại",
    "paper": "giấy",
    "plastic": "nhựa",
    "unknown": "không xác định",
    "n/a": "",
}

_FONT_CACHE = {}


def translate_label_vi(label: str) -> str:
    key = str(label or "").strip().lower()
    if not key:
        return ""
    return LABELS_VI.get(key, str(label).replace("_", " ").strip())


def build_vietnamese_detection_label(object_label: str, material_label: str) -> str:
    object_vi  = translate_label_vi(object_label)
    material_vi = translate_label_vi(material_label)
    if not object_vi:
        return _capitalize_label(material_vi)
    if not material_vi:
        return _capitalize_label(object_vi)
    return f"{_capitalize_label(object_vi)} {material_vi}".strip()


def draw_label_box(
    image_bgr: np.ndarray,
    text: str,
    x: int,
    y: int,
    font_size: int = 20,
    bg_color=(0, 255, 0),
    text_color=(0, 0, 0),
    padding: int = 4,
) -> np.ndarray:
    """
    Draw Unicode text with a filled background. OpenCV Hershey fonts do not
    render Vietnamese accents reliably, so labels are drawn through PIL.
    """
    if image_bgr is None or image_bgr.size == 0 or not text:
        return image_bgr

    h, w = image_bgr.shape[:2]
    pil_image = Image.fromarray(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_image)
    font = _load_unicode_font(font_size)
    bbox = draw.textbbox((0, 0), text, font=font)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]
    box_w = text_w + padding * 2
    box_h = text_h + padding * 2

    x = max(0, min(int(x), max(0, w - box_w)))
    y = max(0, min(int(y), max(0, h - box_h)))

    draw.rectangle(
        [x, y, x + box_w, y + box_h],
        fill=_bgr_to_rgb(bg_color),
    )
    draw.text(
        (x + padding, y + padding - bbox[1]),
        text,
        font=font,
        fill=_bgr_to_rgb(text_color),
    )
    image_bgr[:] = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    return image_bgr


def _capitalize_label(label: str) -> str:
    if not label:
        return ""
    return label[:1].upper() + label[1:]


def _bgr_to_rgb(color: tuple) -> tuple:
    return int(color[2]), int(color[1]), int(color[0])


def _load_unicode_font(font_size: int):
    if font_size in _FONT_CACHE:
        return _FONT_CACHE[font_size]

    font_candidates = [
        r"C:\Windows\Fonts\arial.ttf",
        r"C:\Windows\Fonts\segoeui.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
        "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
    ]
    for font_path in font_candidates:
        if Path(font_path).exists():
            _FONT_CACHE[font_size] = ImageFont.truetype(font_path, font_size)
            return _FONT_CACHE[font_size]

    _FONT_CACHE[font_size] = ImageFont.load_default()
    return _FONT_CACHE[font_size]


def build_detection_label(object_label: str, material_label: str) -> str:
    """
    Combine YOLO object type with ResNet material label.
    Example: chai + plastic -> chai_plastic.
    """
    if not object_label or object_label == "n/a":
        return material_label
    if not material_label:
        return object_label
    return f"{object_label}_{material_label}"


# =========================================================
# 1. TIỀN XỬ LÝ NHANH (cho ResNet inference)
# =========================================================

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
    new_w, new_h = int(w * scale), int(h * scale)
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


# =========================================================
# 2. XUẤT ẢNH CHẤT LƯỢNG CAO (cho display/lưu file)
# =========================================================

def enhance_for_display(image: np.ndarray) -> np.ndarray:
    """
    Nâng cấp ảnh bằng Real-ESRGAN để xuất ra file chất lượng cao.
    """
    h, w = image.shape[:2]
    image = letterbox_to_square(image, max(h, w))
    if ESRGAN_SMART_MODE:
        return smart_esrgan_enhance(image, target_size=ESRGAN_TARGET_SIZE)
    return upscale_with_esrgan(image, target_size=ESRGAN_TARGET_SIZE)


# =========================================================
# 3. TRANSFORM & INFERENCE
# =========================================================

def _build_resnet_transform() -> transforms.Compose:
    """Transform pipeline cho ResNet — letterbox đã xử lý trước ở numpy stage."""
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


# Transform khởi tạo một lần, tái sử dụng cho tất cả inference
_resnet_transform = _build_resnet_transform()


def preprocess_for_resnet(image_crop: np.ndarray) -> torch.Tensor:
    """
    Chuyển numpy BGR crop → Tensor chuẩn ImageNet cho ResNet.

    Args:
        image_crop: numpy array BGR

    Returns:
        Tensor shape [1, 3, 224, 224] trên DEVICE
    """
    image_crop = letterbox_to_square(image_crop, RESNET_INPUT_SIZE)
    pil_image = Image.fromarray(cv2.cvtColor(image_crop, cv2.COLOR_BGR2RGB))
    return _resnet_transform(pil_image).unsqueeze(0).to(DEVICE)


def preprocess_batch_for_resnet(crops: list) -> torch.Tensor:
    """
    Chuyển list numpy BGR crops → batch Tensor cho ResNet.
    Hiệu quả hơn gọi preprocess_for_resnet từng cái một.

    Args:
        crops: list of numpy array BGR

    Returns:
        Tensor shape [N, 3, 224, 224] trên DEVICE
    """
    tensors = [
        preprocess_for_resnet(crop).squeeze(0) for crop in crops
    ]
    return torch.stack(tensors, dim=0)


def get_class_label(
    prediction: torch.Tensor,
    class_names: list,
    conf_threshold: float = RESNET_CONF_THRESHOLD,
) -> tuple[str, float]:
    """
    Trích xuất nhãn và confidence từ output của ResNet.

    Args:
        prediction    : raw logits tensor [1, num_classes]
        class_names   : danh sách tên class
        conf_threshold: dưới ngưỡng này trả về "unknown"

    Returns:
        (label, confidence) — label = "unknown" nếu confidence thấp
    """
    probs = torch.nn.functional.softmax(prediction, dim=1)
    confidence, predicted_idx = torch.max(probs, 1)

    conf_val  = confidence.item()
    label     = class_names[predicted_idx.item()]

    if conf_val < conf_threshold:
        logger.debug(
            f"[ResNet] Confidence thấp ({conf_val:.2f}) cho '{label}' → unknown"
        )
        return "unknown", conf_val

    return label, conf_val


def get_class_labels_batch(
    predictions: torch.Tensor,
    class_names: list,
    conf_threshold: float = RESNET_CONF_THRESHOLD,
) -> list[tuple[str, float]]:
    """
    Phiên bản batch của get_class_label.

    Args:
        predictions: raw logits tensor [N, num_classes]
        class_names: danh sách tên class

    Returns:
        list of (label, confidence)
    """
    probs       = torch.nn.functional.softmax(predictions, dim=1)
    confidences, indices = torch.max(probs, dim=1)

    results = []
    for conf, idx in zip(confidences.tolist(), indices.tolist()):
        label = class_names[idx] if conf >= conf_threshold else "unknown"
        results.append((label, conf))

    return results
