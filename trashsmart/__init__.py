"""TrashSmart package."""
import os

# Dependencies are installed at build/setup time, never implicitly at inference.
os.environ.setdefault("YOLO_AUTOINSTALL", "false")
