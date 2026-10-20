"""Preprocessing settings shared by ResNet evaluation and deployment."""
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[2] / ".env")

RESNET_INPUT_SIZE = int(os.getenv("RESNET_INPUT_SIZE", "224"))
FIX_LOW_LIGHT = os.getenv("FIX_LOW_LIGHT", "1") == "1"
FIX_MOTION_BLUR = os.getenv("FIX_MOTION_BLUR", "1") == "1"
LOW_LIGHT_BRIGHTNESS_THRESHOLD = int(os.getenv("LOW_LIGHT_BRIGHTNESS_THRESHOLD", "80"))
BLUR_LAPLACIAN_THRESHOLD = float(os.getenv("BLUR_LAPLACIAN_THRESHOLD", "100.0"))
