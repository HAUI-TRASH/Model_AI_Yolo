"""Reject invalid training metrics before they affect model selection."""
import math


def require_finite(value, name):
    value = float(value)
    if not math.isfinite(value):
        raise FloatingPointError(f"{name} must be finite; got {value}. Checkpoint not updated.")
    return value


def require_accuracy(value, name="validation accuracy"):
    value = require_finite(value, name)
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be between 0 and 1; got {value}.")
    return value
