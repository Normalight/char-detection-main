"""Lean PP-OCRv6 ONNX inference components used by the application."""

from .detector import PPOcrV6Det, PPOcrV6Detector
from .recognizer import PPOcrV6RecHandle

__all__ = ["PPOcrV6Det", "PPOcrV6Detector", "PPOcrV6RecHandle"]
