import os
import sys
from pathlib import Path

import torch
from dotenv import load_dotenv

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(PROJECT_ROOT / ".env")

from trashsmart.shared.constants import IMAGENET_MEAN, IMAGENET_STD


MODELS_DIR = PROJECT_ROOT / "trashsmart" / "models"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CLASS_NAMES = os.getenv("TRASHSMART_CLASSES", "glass,metal,paper,plastic").split(",")
CLASS_NAMES = [name.strip() for name in CLASS_NAMES if name.strip()]
NUM_CLASSES = len(CLASS_NAMES)
YOLO_OBJECT_LABEL = os.getenv("YOLO_OBJECT_LABEL", "chai")

YOLO_MODEL_PATH = Path(os.getenv("YOLO_MODEL_PATH", MODELS_DIR / "best_new.pt"))
RESNET_MODEL_PATH = Path(os.getenv("RESNET_MODEL_PATH", MODELS_DIR / "best_resnet50_model1.pth"))
ESRGAN_MODEL_PATH = Path(os.getenv("ESRGAN_MODEL_PATH", MODELS_DIR / "RealESRGAN_x4plus.pth"))

CONF_THRESHOLD = float(os.getenv("YOLO_CONF_THRESHOLD", "0.35"))
YOLO_IMG_SIZE = int(os.getenv("YOLO_IMG_SIZE", "640"))

RESNET_RESIZE = int(os.getenv("RESNET_RESIZE", "256"))
RESNET_INPUT_SIZE = int(os.getenv("RESNET_INPUT_SIZE", "224"))
RESNET_CONF_THRESHOLD = float(os.getenv("RESNET_CONF_THRESHOLD", "0.5"))

FIX_LOW_LIGHT = os.getenv("FIX_LOW_LIGHT", "1") == "1"
FIX_MOTION_BLUR = os.getenv("FIX_MOTION_BLUR", "1") == "1"
FIX_RESIZE_BLUR = os.getenv("FIX_RESIZE_BLUR", "1") == "1"

# FIX_ESRGAN = os.getenv("FIX_ESRGAN", "1") == "1"
FIX_ESRGAN = os.getenv("FIX_ESRGAN", "0") == "1"
ESRGAN_SMART_MODE = os.getenv("ESRGAN_SMART_MODE", "1") == "1"
ESRGAN_ANIME_MODE = os.getenv("ESRGAN_ANIME_MODE", "0") == "1"
ESRGAN_TARGET_SIZE = int(os.getenv("ESRGAN_TARGET_SIZE", "512"))
ESRGAN_SCALE = int(os.getenv("ESRGAN_SCALE", "4"))
ESRGAN_TILE = int(os.getenv("ESRGAN_TILE", "256"))
ESRGAN_TILE_PAD = int(os.getenv("ESRGAN_TILE_PAD", "10"))
ESRGAN_PRE_PAD = int(os.getenv("ESRGAN_PRE_PAD", "0"))
ESRGAN_FP32 = os.getenv("ESRGAN_FP32", "0") == "1"

SMALL_CROP_THRESHOLD = int(os.getenv("SMALL_CROP_THRESHOLD", "120"))
LOW_LIGHT_BRIGHTNESS_THRESHOLD = int(os.getenv("LOW_LIGHT_BRIGHTNESS_THRESHOLD", "80"))
BLUR_LAPLACIAN_THRESHOLD = float(os.getenv("BLUR_LAPLACIAN_THRESHOLD", "100.0"))

OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", PROJECT_ROOT / "outputs" / "predict1"))
CROPS_DIR = OUTPUT_DIR / "crops"
INPUT_IMAGE = Path(os.getenv("INPUT_IMAGE", r"C:\trash_classification\source6\dataset_new\plastic\248_jpg.rf.3baf2dc89356f46790fe4be8b08a5837.jpg"))

CAMERA_INDEX = int(os.getenv("CAMERA_INDEX", "0"))
CAMERA_WIDTH = int(os.getenv("CAMERA_WIDTH", "1280"))
CAMERA_HEIGHT = int(os.getenv("CAMERA_HEIGHT", "720"))
CAMERA_FPS = int(os.getenv("CAMERA_FPS", "30"))
WINDOW_NAME = os.getenv("WINDOW_NAME", "Trash Classifier")

BOX_COLOR = (0, 255, 0)
BOX_THICKNESS = 3
FONT_SCALE = 0.9
FONT_THICKNESS = 2
