# config.py - Cấu hình tập trung cho Yolo+ResNet Pipeline
import os
import torch

# ========================
# Device Configuration
# ========================
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ========================
# Class Configuration
# ========================
CLASS_NAMES = ['glass', 'metal', 'paper', 'plastic']
NUM_CLASSES = len(CLASS_NAMES)

# ========================
# Model Paths
# ========================
YOLO_MODEL_PATH   = r"C:\trash_classification\best.pt"
RESNET_MODEL_PATH = r"C:\trash_classification\best_resnet50_model1.pth"

# ========================
# YOLO Detection Parameters
# ========================
CONF_THRESHOLD = 0.35
YOLO_IMG_SIZE  = 640
MIN_CROP_SIZE  = 5

# ========================
# ResNet Preprocessing
# ========================
RESNET_RESIZE      = 256
RESNET_INPUT_SIZE  = 224
IMAGENET_MEAN      = [0.485, 0.456, 0.406]
IMAGENET_STD       = [0.229, 0.224, 0.225]

# ========================
# ResNet Architecture
# ========================
RESNET_FEATURES = 2048
RESNET_HIDDEN_1 = 512
RESNET_HIDDEN_2 = 256
DROPOUT_RATE    = 0.5

# ========================
# Image Enhancement Flags
# ========================
FIX_SMALL_CROP  = True   # (1) Upscale crops nhỏ bằng LANCZOS
FIX_LOW_LIGHT   = True   # (2) Gamma + CLAHE cho ảnh tối
FIX_MOTION_BLUR = True   # (3) Unsharp mask cho ảnh mờ
FIX_RESIZE_BLUR = True   # (4) LANCZOS resize + unsharp nhẹ

# -------------------------------------------
# Real-ESRGAN Enhancement  ← MỚI
# -------------------------------------------
FIX_ESRGAN       = True  # Bật/tắt Real-ESRGAN
                          # True  → thay thế fix_small_crop + fix_motion_blur
                          # False → dùng pipeline cũ

ESRGAN_SMART_MODE = True  # True  → chỉ ESRGAN khi ảnh nhỏ/mờ (nhanh hơn)
                           # False → luôn ESRGAN với mọi crop (chất lượng tối đa)

# ========================
# Enhancement Thresholds
# ========================
SMALL_CROP_THRESHOLD         = 120   # px – upscale nếu min(h,w) < này
LOW_LIGHT_BRIGHTNESS_THRESHOLD = 80  # mean brightness trên kênh L
BLUR_LAPLACIAN_THRESHOLD     = 100.0 # Laplacian variance thấp hơn → mờ

# ========================
# Real-ESRGAN Config  ← MỚI
# ========================
ESRGAN_MODEL_PATH  = r"C:\trash_classification\RealESRGAN_x4plus.pth"
                     # Tải tại: https://github.com/xinntao/Real-ESRGAN/releases
ESRGAN_SCALE       = 4      # 2 hoặc 4 (phải khớp weights)
ESRGAN_TILE        = 256    # Chia tile để giảm VRAM (0 = không chia)
ESRGAN_TILE_PAD    = 10
ESRGAN_PRE_PAD     = 0
ESRGAN_FP32        = True   # True cho CPU hoặc GPU cũ
ESRGAN_TARGET_SIZE = 512    # Kích thước output sau upscale
ESRGAN_ANIME_MODE  = False  # True nếu dùng model anime

# ========================
# Output Configuration
# ========================
OUTPUT_DIR      = r"C:\trash_classification\source6\Yolo+ResNet(goc)\predict1"
CROPS_DIR       = os.path.join(OUTPUT_DIR, "crops")
OUTPUT_FILENAME = "result_multi.jpg"
INPUT_IMAGE     = r"C:\trash_classification\z7647974978123_17877677da13c5ed5d08f97dfbdbbac0.jpg"

# ========================
# Visualization
# ========================
BOX_COLOR      = (0, 255, 0)
BOX_THICKNESS  = 3
FONT_SCALE     = 0.9
FONT_THICKNESS = 2
TEXT_OFFSET_Y  = 10
MIN_TEXT_Y     = 25