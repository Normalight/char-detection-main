# -*- coding: utf-8 -*-
"""Direct ONNX runtime for PP-OCRv6 text detection.

The preprocessing and DB postprocessing parameters are read from the official
``inference.yml`` shipped with each exported model.  Runtime resize policy is
kept explicit because PaddleOCR's module API uses a 960-pixel maximum side,
while this project's production default follows the legacy DBNet semantics:
the ``short`` policy scales the shorter side to ``limit_side_len`` in BOTH
directions, so small pages are upscaled (legacy always did this) and large
scans keep the same detection resolution the legacy detector used.
"""

from dataclasses import dataclass, replace
import logging
import os
from pathlib import Path
from typing import Optional, Tuple, Union

import cv2
import numpy as np
import onnxruntime as ort
from PIL import Image
import yaml

from ocr.postprocess import SegDetectorRepresenter

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DetectorConfig:
    """DB postprocess and image-normalization values from ``inference.yml``."""

    thresh: float
    box_thresh: float
    unclip_ratio: float
    max_candidates: int
    mean: Tuple[float, float, float]
    std: Tuple[float, float, float]


def load_detector_config(yml_path: Union[str, os.PathLike]) -> DetectorConfig:
    """Load the official detector preprocessing/postprocessing configuration."""

    with Path(yml_path).open(encoding="utf-8") as stream:
        raw = yaml.safe_load(stream)

    try:
        postprocess = raw["PostProcess"]
        normalize = next(
            item["NormalizeImage"]
            for item in raw["PreProcess"]["transform_ops"]
            if "NormalizeImage" in item
        )
        return DetectorConfig(
            thresh=float(postprocess["thresh"]),
            box_thresh=float(postprocess["box_thresh"]),
            unclip_ratio=float(postprocess["unclip_ratio"]),
            max_candidates=int(postprocess["max_candidates"]),
            mean=tuple(float(value) for value in normalize["mean"]),
            std=tuple(float(value) for value in normalize["std"]),
        )
    except (KeyError, StopIteration, TypeError) as exc:
        raise ValueError(f"invalid PP-OCRv6 inference config: {yml_path}") from exc


def calculate_resized_shape(
    image_height: int,
    image_width: int,
    limit_side_len: int = 960,
    limit_type: str = "short",
    max_side_limit: int = 4000,
) -> Tuple[int, int]:
    """Calculate an aspect-preserving, 32-aligned detector input shape.

    Supported ``limit_type`` policies:

    - ``short``: scale so the shorter side equals ``limit_side_len`` (both
      upscaling and downscaling).  This matches the legacy DBNET behaviour the
      UI parameter "检测尺寸" historically had, so small pages are upscaled
      instead of silently losing detection resolution.
    - ``max``: cap the longer side at ``limit_side_len``, never upscale.
    - ``min``: upscale until the shorter side reaches ``limit_side_len``,
      never downscale.

    All policies are finally clamped by ``max_side_limit`` and aligned to 32.
    """

    if image_height <= 0 or image_width <= 0:
        raise ValueError("image dimensions must be positive")
    if limit_side_len <= 0 or max_side_limit <= 0:
        raise ValueError("side limits must be positive")

    if limit_type == "short":
        ratio = limit_side_len / min(image_height, image_width)
    elif limit_type == "max":
        ratio = min(1.0, limit_side_len / max(image_height, image_width))
    elif limit_type == "min":
        ratio = (
            max(1.0, limit_side_len / min(image_height, image_width))
            if min(image_height, image_width) < limit_side_len
            else 1.0
        )
    else:
        raise ValueError(f"unsupported limit_type: {limit_type}")

    resized_height = int(image_height * ratio)
    resized_width = int(image_width * ratio)
    if max(resized_height, resized_width) > max_side_limit:
        clamp_ratio = max_side_limit / max(resized_height, resized_width)
        resized_height = int(resized_height * clamp_ratio)
        resized_width = int(resized_width * clamp_ratio)

    resized_height = max(32, int(round(resized_height / 32.0)) * 32)
    resized_width = max(32, int(round(resized_width / 32.0)) * 32)
    aligned_limit = max(32, (max_side_limit // 32) * 32)
    if max(resized_height, resized_width) > aligned_limit:
        clamp_ratio = aligned_limit / max(resized_height, resized_width)
        resized_height = max(
            32, int(round(resized_height * clamp_ratio / 32.0)) * 32
        )
        resized_width = max(
            32, int(round(resized_width * clamp_ratio / 32.0)) * 32
        )
    return resized_height, resized_width


def _make_session_options() -> ort.SessionOptions:
    options = ort.SessionOptions()
    cpu_count = os.cpu_count() or 1
    options.intra_op_num_threads = max(1, cpu_count // 2)
    options.inter_op_num_threads = 1
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return options


def _build_session(model_path: Union[str, os.PathLike]) -> ort.InferenceSession:
    return ort.InferenceSession(
        str(model_path),
        sess_options=_make_session_options(),
        providers=["CPUExecutionProvider"],
    )


class PPOcrV6Detector:
    """PP-OCRv6 detector with official model metadata and explicit resize policy."""

    def __init__(
        self,
        model_path: Union[str, os.PathLike],
        *,
        yml_path: Optional[Union[str, os.PathLike]] = None,
        limit_side_len: int = 960,
        limit_type: str = "short",
        max_side_limit: int = 4000,
        thresh: Optional[float] = None,
        box_thresh: Optional[float] = None,
        unclip_ratio: Optional[float] = None,
        max_candidates: Optional[int] = None,
    ):
        model_path = Path(model_path)
        config_path = (
            Path(yml_path)
            if yml_path is not None
            else model_path.with_name("inference.yml")
        )
        config = load_detector_config(config_path)
        overrides = {
            name: value
            for name, value in {
                "thresh": thresh,
                "box_thresh": box_thresh,
                "unclip_ratio": unclip_ratio,
                "max_candidates": max_candidates,
            }.items()
            if value is not None
        }
        self.config = replace(config, **overrides)
        self.limit_side_len = int(limit_side_len)
        self.limit_type = limit_type
        self.max_side_limit = int(max_side_limit)
        # Validate policies before the expensive model load.
        calculate_resized_shape(
            32,
            32,
            self.limit_side_len,
            self.limit_type,
            self.max_side_limit,
        )

        self.session = _build_session(model_path)
        self.input_name = self.session.get_inputs()[0].name
        self.decoder = SegDetectorRepresenter(
            thresh=self.config.thresh,
            box_thresh=self.config.box_thresh,
            max_candidates=self.config.max_candidates,
            unclip_ratio=self.config.unclip_ratio,
        )
        self._mean = np.asarray(self.config.mean, dtype=np.float32)
        self._std = np.asarray(self.config.std, dtype=np.float32)

    def _preprocess(
        self,
        image_rgb: Union[np.ndarray, Image.Image],
        limit_side_len: Optional[int] = None,
    ) -> Tuple[np.ndarray, int, int]:
        if isinstance(image_rgb, Image.Image):
            rgb = np.asarray(image_rgb.convert("RGB"), dtype=np.uint8)
        else:
            rgb = np.asarray(image_rgb)
        if rgb.ndim != 3 or rgb.shape[2] != 3:
            raise ValueError(f"expected RGB image with 3 channels, got {rgb.shape}")

        original_height, original_width = rgb.shape[:2]
        resized_height, resized_width = calculate_resized_shape(
            original_height,
            original_width,
            self.limit_side_len if limit_side_len is None else int(limit_side_len),
            self.limit_type,
            self.max_side_limit,
        )
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        if (resized_height, resized_width) != (
            original_height,
            original_width,
        ):
            bgr = cv2.resize(
                bgr,
                (resized_width, resized_height),
                interpolation=cv2.INTER_LINEAR,
            )

        normalized = bgr.astype(np.float32) / 255.0
        normalized = (normalized - self._mean) / self._std
        tensor = np.ascontiguousarray(
            normalized.transpose(2, 0, 1)[np.newaxis]
        )
        return tensor, original_height, original_width

    def process(
        self,
        image_rgb: Union[np.ndarray, Image.Image],
        limit_side_len: Optional[int] = None,
    ):
        """Return ``(boxes, scores)`` in the original image coordinate system."""

        tensor, original_height, original_width = self._preprocess(
            image_rgb, limit_side_len
        )
        score_map = self.session.run(
            None, {self.input_name: tensor.astype(np.float32, copy=False)}
        )[0]
        if score_map.ndim == 4:
            score_map = score_map[:, 0]
        boxes, scores = self.decoder(
            score_map, original_height, original_width
        )
        if isinstance(boxes, np.ndarray) and len(boxes) > 0:
            valid = boxes.reshape(boxes.shape[0], -1).sum(axis=1) > 0
            boxes = boxes[valid]
            scores = np.asarray(scores)[valid]
        else:
            boxes, scores = [], []
        return boxes, scores

    def detect(self, image_rgb: Union[np.ndarray, Image.Image]):
        """Compatibility convenience method returning boxes only."""

        boxes, _scores = self.process(image_rgb)
        return boxes


PPOcrV6Det = PPOcrV6Detector

__all__ = [
    "DetectorConfig",
    "PPOcrV6Det",
    "PPOcrV6Detector",
    "calculate_resized_shape",
    "load_detector_config",
]
