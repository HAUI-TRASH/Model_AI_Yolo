# esrgan_enhancer.py - Real-ESRGAN Super Resolution cho pipeline YOLO+ResNet
"""
Cài đặt trước khi dùng:
    pip install basicsr facexlib gfpgan
    pip install git+https://github.com/xinntao/Real-ESRGAN.git

Download model weights:
    # Real-ESRGAN x4 (general purpose - khuyên dùng)
    wget https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth
    
    # Real-ESRGAN x2 (nếu muốn upscale ít hơn)
    wget https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth
    
    # Anime version (nếu ảnh có nét cartoon)
    wget https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth
"""

# ========================
# Patch torchvision >= 0.16
# (functional_tensor đã bị xóa, basicsr vẫn dùng API cũ)
# ========================
import sys
import types
try:
    import torchvision.transforms.functional as _F
    _patch = types.ModuleType('torchvision.transforms.functional_tensor')
    _patch.rgb_to_grayscale   = _F.rgb_to_grayscale
    _patch.to_grayscale       = _F.rgb_to_grayscale
    sys.modules['torchvision.transforms.functional_tensor'] = _patch
except Exception:
    pass

import cv2
import numpy as np
import torch
import os
import warnings
warnings.filterwarnings("ignore")


# ========================
# ESRGAN Config (thêm vào config.py nếu muốn)
# ========================
ESRGAN_MODEL_PATH   = r"C:\trash_classification\RealESRGAN_x4plus.pth"
ESRGAN_SCALE        = 4          # 2 hoặc 4 (phải khớp với model weights)
ESRGAN_TILE         = 256        # Chia tile để tiết kiệm VRAM (0 = không chia)
ESRGAN_TILE_PAD     = 10
ESRGAN_PRE_PAD      = 0
ESRGAN_FP32         = True       # True nếu CPU hoặc GPU cũ (không hỗ trợ fp16)
ESRGAN_TARGET_SIZE  = 512        # Resize về kích thước này sau khi upscale
ESRGAN_ANIME_MODE   = False      # True nếu dùng model anime


# ========================
# Lazy Loader (chỉ load model 1 lần)
# ========================
_esrgan_upsampler = None  # Global cache


def _load_esrgan_model():
    """
    Load Real-ESRGAN model (singleton pattern).
    Chỉ load lần đầu, tái dùng các lần sau.
    """
    global _esrgan_upsampler
    if _esrgan_upsampler is not None:
        return _esrgan_upsampler

    try:
        from basicsr.archs.rrdbnet_arch import RRDBNet
        from realesrgan import RealESRGANer

        if ESRGAN_ANIME_MODE:
            # Model anime: nhẹ hơn (6 blocks thay vì 23)
            model = RRDBNet(
                num_in_ch=3, num_out_ch=3,
                num_feat=64, num_block=6,
                num_grow_ch=32, scale=ESRGAN_SCALE
            )
        else:
            # Model general: chất lượng tốt hơn
            model = RRDBNet(
                num_in_ch=3, num_out_ch=3,
                num_feat=64, num_block=23,
                num_grow_ch=32, scale=ESRGAN_SCALE
            )

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"[ESRGAN] Loading model on {device}...")

        _esrgan_upsampler = RealESRGANer(
            scale=ESRGAN_SCALE,
            model_path=ESRGAN_MODEL_PATH,
            model=model,
            tile=ESRGAN_TILE,
            tile_pad=ESRGAN_TILE_PAD,
            pre_pad=ESRGAN_PRE_PAD,
            half=not ESRGAN_FP32,   # fp16 nếu GPU mạnh
            device=device
        )
        print(f"[ESRGAN] Model loaded ✓  (scale=x{ESRGAN_SCALE})")
        return _esrgan_upsampler

    except ImportError as e:
        print(f"[ESRGAN] ImportError: {e}")
        print("[ESRGAN] Chạy: pip install basicsr realesrgan")
        return None
    except Exception as e:
        print(f"[ESRGAN] Lỗi load model: {e}")
        return None


# ========================
# Core Enhancement Function
# ========================

def upscale_with_esrgan(image_bgr: np.ndarray,
                         target_size: int = ESRGAN_TARGET_SIZE) -> np.ndarray:
    """
    Upscale ảnh bằng Real-ESRGAN rồi resize về target_size.

    Args:
        image_bgr  : numpy array BGR (OpenCV format)
        target_size: kích thước output cuối cùng (px)

    Returns:
        numpy array BGR đã được upscale + resize
    """
    upsampler = _load_esrgan_model()

    if upsampler is None:
        # Fallback: LANCZOS resize nếu ESRGAN không load được
        print("[ESRGAN] Fallback → LANCZOS resize")
        return cv2.resize(image_bgr, (target_size, target_size),
                          interpolation=cv2.INTER_LANCZOS4)

    try:
        # Real-ESRGAN nhận BGR, trả về BGR
        output_bgr, _ = upsampler.enhance(image_bgr, outscale=ESRGAN_SCALE)

        # Resize về target_size sau khi upscale
        h, w = output_bgr.shape[:2]
        if h != target_size or w != target_size:
            # Giữ tỉ lệ, crop center
            output_bgr = _resize_and_center_crop(output_bgr, target_size)

        return output_bgr

    except RuntimeError as e:
        if "CUDA out of memory" in str(e):
            print("[ESRGAN] CUDA OOM → thử TILE nhỏ hơn hoặc CPU")
        else:
            print(f"[ESRGAN] RuntimeError: {e}")
        # Fallback
        return cv2.resize(image_bgr, (target_size, target_size),
                          interpolation=cv2.INTER_LANCZOS4)
    except Exception as e:
        print(f"[ESRGAN] Lỗi enhance: {e}")
        return cv2.resize(image_bgr, (target_size, target_size),
                          interpolation=cv2.INTER_LANCZOS4)


def _resize_and_center_crop(image: np.ndarray, size: int) -> np.ndarray:
    """
    Resize ảnh sao cho cạnh ngắn = size, rồi center crop vuông.
    Tương tự torchvision Resize + CenterCrop.
    """
    h, w = image.shape[:2]
    # Scale sao cho cạnh ngắn = size
    if h < w:
        new_h = size
        new_w = int(w * size / h)
    else:
        new_w = size
        new_h = int(h * size / w)

    image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)

    # Center crop
    h, w = image.shape[:2]
    top  = (h - size) // 2
    left = (w - size) // 2
    return image[top:top+size, left:left+size]


# ========================
# Smart ESRGAN (chỉ upscale khi cần)
# ========================

def smart_esrgan_enhance(image_bgr: np.ndarray,
                          size_threshold: int = 120,
                          blur_threshold: float = 100.0,
                          target_size: int = ESRGAN_TARGET_SIZE) -> np.ndarray:
    """
    Chỉ chạy ESRGAN khi ảnh nhỏ HOẶC mờ.
    Tránh lãng phí thời gian với ảnh đã đủ nét.

    Args:
        image_bgr      : ảnh đầu vào BGR
        size_threshold : upscale nếu min(h,w) < threshold
        blur_threshold : upscale nếu Laplacian variance < threshold
        target_size    : kích thước output

    Returns:
        ảnh đã xử lý
    """
    h, w = image_bgr.shape[:2]
    min_dim = min(h, w)

    # Kiểm tra kích thước
    need_upscale = min_dim < size_threshold

    # Kiểm tra blur (chỉ tính nếu chưa quyết định upscale)
    if not need_upscale:
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        need_upscale = lap_var < blur_threshold
        if need_upscale:
            print(f"[ESRGAN] Ảnh mờ (Laplacian={lap_var:.1f}) → ESRGAN")

    if need_upscale:
        return upscale_with_esrgan(image_bgr, target_size)
    else:
        # Ảnh đã đủ nét → chỉ resize thường
        return _resize_and_center_crop(image_bgr, target_size)


# ========================
# Batch Processing (cho nhiều ảnh)
# ========================

def batch_esrgan(image_dir: str,
                 output_dir: str,
                 extensions: tuple = (".jpg", ".jpeg", ".png")) -> list:
    """
    Xử lý hàng loạt ảnh trong một thư mục.

    Args:
        image_dir  : thư mục chứa ảnh gốc
        output_dir : thư mục lưu ảnh đã xử lý
        extensions : các định dạng ảnh cần xử lý

    Returns:
        list: đường dẫn các ảnh đã xử lý
    """
    os.makedirs(output_dir, exist_ok=True)
    output_paths = []

    image_files = [
        f for f in os.listdir(image_dir)
        if os.path.splitext(f)[1].lower() in extensions
    ]

    print(f"[ESRGAN Batch] Xử lý {len(image_files)} ảnh...")

    for i, filename in enumerate(image_files, 1):
        input_path  = os.path.join(image_dir, filename)
        output_path = os.path.join(output_dir, filename)

        image = cv2.imread(input_path)
        if image is None:
            print(f"  [{i}/{len(image_files)}] Bỏ qua (không đọc được): {filename}")
            continue

        enhanced = upscale_with_esrgan(image)
        cv2.imwrite(output_path, enhanced)
        output_paths.append(output_path)
        print(f"  [{i}/{len(image_files)}] ✓ {filename}")

    print(f"[ESRGAN Batch] Hoàn tất → {output_dir}")
    return output_paths