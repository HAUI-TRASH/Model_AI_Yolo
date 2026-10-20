"""ResNet class order is part of the checkpoint/API contract."""
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[2] / ".env")
CLASS_NAMES = [name.strip() for name in os.getenv(
    "TRASHSMART_CLASSES", "glass,metal,paper,plastic",
).split(",") if name.strip()]
if not CLASS_NAMES or len(CLASS_NAMES) != len(set(CLASS_NAMES)):
    raise ValueError("TRASHSMART_CLASSES must contain unique, non-empty class names.")

# Original order used by this project's pre-metadata ResNet checkpoints.
# This must never be inferred from the current environment or dataset folders.
LEGACY_CLASS_NAMES = ("glass", "metal", "paper", "plastic")


def validate_checkpoint_classes(checkpoint, num_classes):
    """Return True for metadata checkpoints, False for the supported legacy format."""
    if num_classes != len(CLASS_NAMES):
        raise ValueError("ResNet num_classes does not match configured class names.")
    if isinstance(checkpoint, dict) and "class_names" in checkpoint:
        names = checkpoint["class_names"]
        if (not isinstance(names, (list, tuple)) or not names
                or any(not isinstance(name, str) or not name for name in names)
                or len(set(names)) != len(names)):
            raise ValueError("Invalid class_names metadata in ResNet checkpoint.")
        if list(names) != CLASS_NAMES:
            raise ValueError(f"ResNet checkpoint class order {list(names)} does not match API {CLASS_NAMES}.")
        if checkpoint.get("num_classes", len(names)) != len(names):
            raise ValueError("ResNet checkpoint num_classes metadata is inconsistent.")
        if checkpoint.get("format_version", 1) != 1:
            raise ValueError("Unsupported ResNet checkpoint format_version.")
        if checkpoint.get("architecture", "TrashResNet50") != "TrashResNet50":
            raise ValueError("ResNet checkpoint architecture is not TrashResNet50.")
        return True
    if isinstance(checkpoint, dict) and "format_version" in checkpoint:
        raise ValueError("Versioned ResNet checkpoint is missing class_names metadata.")
    if CLASS_NAMES != list(LEGACY_CLASS_NAMES):
        raise ValueError(
            "Legacy ResNet checkpoint has no class metadata and only supports the original "
            f"class order {list(LEGACY_CLASS_NAMES)}. Cannot use a different order safely."
        )
    return False


def validate_class_order(dataset_classes):
    if list(dataset_classes) != CLASS_NAMES:
        raise ValueError(
            f"Dataset class order {list(dataset_classes)} does not match API/checkpoint "
            f"class order {CLASS_NAMES}. Training stopped before saving any checkpoint."
        )
