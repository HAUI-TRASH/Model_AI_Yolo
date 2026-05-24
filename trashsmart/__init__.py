"""TrashSmart package."""

from importlib import import_module
import sys


for _subpackage in ("shared", "training", "pipeline", "yolo_training"):
    sys.modules.setdefault(_subpackage, import_module(f"{__name__}.{_subpackage}"))
