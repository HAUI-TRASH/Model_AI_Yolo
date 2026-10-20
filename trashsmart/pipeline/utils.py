# utils.py
import logging
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from trashsmart.pipeline.config import (
    DEVICE,
    ESRGAN_SMART_MODE,
    ESRGAN_TARGET_SIZE,
    RESNET_CONF_THRESHOLD,
    RESNET_INPUT_SIZE,
)
from trashsmart.pipeline.esrgan_enhancer import smart_esrgan_enhance, upscale_with_esrgan
from trashsmart.shared.preprocessing import (
    fix_low_light, fix_motion_blur, letterbox_to_square,
    quick_enhance_for_resnet, preprocess_resnet_crop,
)

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

def preprocess_for_resnet(image_crop: np.ndarray) -> torch.Tensor:
    """
    Chuyển numpy BGR crop → Tensor chuẩn ImageNet cho ResNet.

    Args:
        image_crop: numpy array BGR

    Returns:
        Tensor shape [1, 3, 224, 224] trên DEVICE
    """
    return preprocess_resnet_crop(image_crop, RESNET_INPUT_SIZE).unsqueeze(0).to(DEVICE)


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
