# esrgan_enhancer.py - Real-ESRGAN Super Resolution cho pipeline YOLO + ResNet
"""
Cài đặt trước khi dùng:
    pip install basicsr facexlib gfpgan
    pip install git+https://github.com/xinntao/Real-ESRGAN.git

Download model weights:
    # Real-ESRGAN x4 (general purpose — khuyên dùng)
    wget https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth

    # Real-ESRGAN x2 (upscale ít hơn)
    wget https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth

    # Anime version
    wget https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth
"""

# ========================
# Patch torchvision >= 0.16
# functional_tensor đã bị xóa nhưng basicsr vẫn dùng API cũ
# ========================
import sys
import types
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

try:
    import torchvision.transforms.functional as _F
    _patch = types.ModuleType("torchvision.transforms.functional_tensor")
    _patch.rgb_to_grayscale = _F.rgb_to_grayscale
    _patch.to_grayscale     = _F.rgb_to_grayscale
    sys.modules["torchvision.transforms.functional_tensor"] = _patch
except ImportError as exc:
    import logging
    logging.getLogger(__name__).warning("Torchvision compatibility patch skipped: %s", exc)

import cv2
import logging
import numpy as np
import os
import threading
import torch
import warnings

warnings.filterwarnings("ignore")

from pipeline.config import (
    ESRGAN_MODEL_PATH,
    ESRGAN_SCALE,
    ESRGAN_TILE,
    ESRGAN_TILE_PAD,
    ESRGAN_PRE_PAD,
    ESRGAN_FP32,
    ESRGAN_TARGET_SIZE,
    ESRGAN_ANIME_MODE,
    SMALL_CROP_THRESHOLD,
    BLUR_LAPLACIAN_THRESHOLD,
    DEVICE,
)

logger = logging.getLogger(__name__)

# ========================
# Thread-safe Singleton Loader
# ========================
_esrgan_upsampler = None
_lock = threading.Lock()


def _build_model():
    """
    Khởi tạo RealESRGANer. Chỉ gọi từ bên trong _lock.
    """
    try:
        from basicsr.archs.rrdbnet_arch import RRDBNet
        from realesrgan import RealESRGANer

        num_block = 6 if ESRGAN_ANIME_MODE else 23
        model = RRDBNet(
            num_in_ch=3,
            num_out_ch=3,
            num_feat=64,
            num_block=num_block,
            num_grow_ch=32,
            scale=ESRGAN_SCALE,
        )

        logger.info(
            f"[ESRGAN] Loading model on {DEVICE} "
            f"(scale=x{ESRGAN_SCALE}, anime={ESRGAN_ANIME_MODE}, fp32={ESRGAN_FP32})"
        )

        upsampler = RealESRGANer(
            scale=ESRGAN_SCALE,
            model_path=str(ESRGAN_MODEL_PATH),
            model=model,
            tile=ESRGAN_TILE,
            tile_pad=ESRGAN_TILE_PAD,
            pre_pad=ESRGAN_PRE_PAD,
            half=not ESRGAN_FP32,
            device=DEVICE,
        )
        logger.info("[ESRGAN] Model loaded ✓")
        return upsampler

    except ImportError as e:
        logger.error(f"[ESRGAN] ImportError: {e}")
        logger.error("[ESRGAN] Chạy: pip install basicsr realesrgan")
        return None
    except Exception as e:
        logger.error(f"[ESRGAN] Lỗi load model: {e}", exc_info=True)
        return None

_NOT_LOADED = object()   # sentinel — phân biệt "chưa load" vs "load thất bại"
_esrgan_upsampler = _NOT_LOADED

def _load_esrgan_model():
    """
    Trả về RealESRGANer singleton, thread-safe (double-checked locking).
    """
    global _esrgan_upsampler

    if _esrgan_upsampler is not _NOT_LOADED:
        return _esrgan_upsampler

    with _lock:
        if _esrgan_upsampler is _NOT_LOADED:
            _esrgan_upsampler = _build_model()

    return _esrgan_upsampler


# ========================
# Core Enhancement
# ========================

def upscale_with_esrgan(
    image_bgr: np.ndarray,
    target_size: int = ESRGAN_TARGET_SIZE,
) -> np.ndarray:
    """
    Upscale ảnh bằng Real-ESRGAN rồi center-crop về target_size.

    Args:
        image_bgr  : numpy array BGR (OpenCV)
        target_size: kích thước output vuông (px)

    Returns:
        numpy array BGR đã upscale + resize
    """
    upsampler = _load_esrgan_model()

    if upsampler is None:
        logger.warning("[ESRGAN] Model không khả dụng → fallback LANCZOS")
        return _lanczos_fallback(image_bgr, target_size)

    try:
        output_bgr, _ = upsampler.enhance(image_bgr, outscale=ESRGAN_SCALE)
        h, w = output_bgr.shape[:2]
        if h != target_size or w != target_size:
            output_bgr = _resize_and_center_crop(output_bgr, target_size)
        return output_bgr

    except RuntimeError as e:
        if "CUDA out of memory" in str(e):
            logger.warning("[ESRGAN] CUDA OOM → giảm ESRGAN_TILE trong config.py")
        else:
            logger.error(f"[ESRGAN] RuntimeError: {e}", exc_info=True)
        return _lanczos_fallback(image_bgr, target_size)

    except Exception as e:
        logger.error(f"[ESRGAN] Lỗi enhance: {e}", exc_info=True)
        return _lanczos_fallback(image_bgr, target_size)


# ========================
# Smart ESRGAN
# ========================

def smart_esrgan_enhance(
    image_bgr: np.ndarray,
    size_threshold: int = SMALL_CROP_THRESHOLD,
    blur_threshold: float = BLUR_LAPLACIAN_THRESHOLD,
    target_size: int = ESRGAN_TARGET_SIZE,
) -> np.ndarray:
    """
    Chỉ chạy ESRGAN khi ảnh nhỏ HOẶC mờ, tránh lãng phí tài nguyên.

    Args:
        image_bgr      : ảnh đầu vào BGR
        size_threshold : upscale nếu min(h, w) < threshold
        blur_threshold : upscale nếu Laplacian variance < threshold
        target_size    : kích thước output

    Returns:
        ảnh đã xử lý
    """
    h, w = image_bgr.shape[:2]
    need_upscale = min(h, w) < size_threshold

    if not need_upscale:
        gray    = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        need_upscale = lap_var < blur_threshold
        if need_upscale:
            logger.info(f"[ESRGAN] Ảnh mờ (Laplacian={lap_var:.1f}) → ESRGAN")

    if need_upscale:
        return upscale_with_esrgan(image_bgr, target_size)

    return _resize_and_center_crop(image_bgr, target_size)


# ========================
# Batch Processing
# ========================

def batch_esrgan(
    image_dir: str,
    output_dir: str,
    extensions: tuple = (".jpg", ".jpeg", ".png"),
) -> list:
    """
    Xử lý hàng loạt ảnh trong một thư mục.

    Args:
        image_dir  : thư mục chứa ảnh gốc
        output_dir : thư mục lưu ảnh đã xử lý
        extensions : định dạng ảnh cần xử lý

    Returns:
        list đường dẫn ảnh đã xử lý thành công
    """
    os.makedirs(output_dir, exist_ok=True)
    output_paths = []

    image_files = [
        f for f in os.listdir(image_dir)
        if os.path.splitext(f)[1].lower() in extensions
    ]

    logger.info(f"[ESRGAN Batch] Bắt đầu xử lý {len(image_files)} ảnh...")

    for i, filename in enumerate(image_files, 1):
        input_path  = os.path.join(image_dir, filename)
        output_path = os.path.join(output_dir, filename)

        image = cv2.imread(input_path)
        if image is None:
            logger.warning(f"  [{i}/{len(image_files)}] Bỏ qua (không đọc được): {filename}")
            continue

        enhanced = upscale_with_esrgan(image)
        cv2.imwrite(output_path, enhanced)
        output_paths.append(output_path)
        logger.info(f"  [{i}/{len(image_files)}] ✓ {filename}")

    logger.info(f"[ESRGAN Batch] Hoàn tất → {output_dir}")
    return output_paths


# ========================
# Internal Helpers
# ========================

def _resize_and_center_crop(image: np.ndarray, size: int) -> np.ndarray:
    """
    Resize cạnh ngắn về `size` rồi center crop thành hình vuông.
    Tương đương torchvision Resize + CenterCrop.
    """
    h, w = image.shape[:2]
    if h < w:
        new_h, new_w = size, int(w * size / h)
    else:
        new_h, new_w = int(h * size / w), size

    image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)

    h, w  = image.shape[:2]
    top   = (h - size) // 2
    left  = (w - size) // 2
    return image[top : top + size, left : left + size]


def _lanczos_fallback(image: np.ndarray, size: int) -> np.ndarray:
    """Resize đơn giản bằng LANCZOS khi ESRGAN không khả dụng."""
    return cv2.resize(image, (size, size), interpolation=cv2.INTER_LANCZOS4)
