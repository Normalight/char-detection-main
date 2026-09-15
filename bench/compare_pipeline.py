# -*- coding: utf-8 -*-
"""Decision-grade benchmark for character-cutting OCR pipeline candidates.

The benchmark compares complete detector → recognizer → character-box paths.
It combines real project images (structural metrics) with deterministic
synthetic Chinese lines (known text and box ground truth).
"""

from __future__ import annotations

import argparse
import copy
from dataclasses import asdict, dataclass
import json
import logging
from pathlib import Path
import random
import statistics
import time
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import onnxruntime as ort
from PIL import Image, ImageDraw, ImageFont

from crnn import CRNNHandle
from dbnet.dbnet_infer import DBNET
from ppocrv6 import PPOcrV6Detector, PPOcrV6RecHandle
from storage_utils import get_rotate_crop_image as get_storage_crop
from utils import get_rotate_crop_image as get_recognition_crop

logger = logging.getLogger(__name__)
ort.set_default_logger_severity(3)


@dataclass(frozen=True)
class Candidate:
    name: str
    det_model: str
    rec_model: str
    char_box_strategy: str


CANDIDATES = (
    Candidate("legacy", "legacy", "crnn", "legacy"),
    Candidate("hybrid-v6-rec", "legacy", "v6-small", "ctc-midpoint"),
    Candidate("full-v6-midpoint", "v6-small", "v6-small", "ctc-midpoint"),
    Candidate(
        "full-v6-projection",
        "v6-small",
        "v6-small",
        "projection-refined",
    ),
    Candidate(
        "full-v6-polarity",
        "v6-small",
        "v6-small",
        "polarity-aware",
    ),
)

DEFAULT_IMAGES = (
    "images/0.png",
    "images/1.png",
    "images/t1.png",
    "images/t2.png",
    "images/c1.jpg",
)
SYNTHETIC_SEED = 20260724


@dataclass(frozen=True)
class SyntheticCase:
    name: str
    image: np.ndarray
    text: str
    ground_truth_boxes: Tuple[np.ndarray, ...]
    orientation: str
    cell_size: int


@dataclass
class RowResult:
    line_box: np.ndarray
    score: float
    text: str
    characters: List[dict]


@dataclass
class PipelineResult:
    rows: List[RowResult]
    det_ms: float
    rec_ms: float

    @property
    def total_ms(self) -> float:
        return self.det_ms + self.rec_ms


def _polygon_area(box: np.ndarray) -> float:
    return abs(float(cv2.contourArea(np.asarray(box, dtype=np.float32))))


def polygon_iou(left: np.ndarray, right: np.ndarray) -> float:
    """Intersection-over-union for two convex quadrilaterals."""

    left = np.asarray(left, dtype=np.float32).reshape(-1, 2)
    right = np.asarray(right, dtype=np.float32).reshape(-1, 2)
    left_area = _polygon_area(left)
    right_area = _polygon_area(right)
    if left_area <= 0 or right_area <= 0:
        return 0.0
    intersection, _polygon = cv2.intersectConvexConvex(left, right)
    union = left_area + right_area - float(intersection)
    return float(intersection) / union if union > 0 else 0.0


def adjacent_overlap_rate(boxes: Sequence[np.ndarray]) -> float:
    """Mean adjacent intersection relative to the smaller character box."""

    if len(boxes) < 2:
        return 0.0
    overlaps = []
    for left, right in zip(boxes, boxes[1:]):
        left = np.asarray(left, dtype=np.float32).reshape(-1, 2)
        right = np.asarray(right, dtype=np.float32).reshape(-1, 2)
        left_area = _polygon_area(left)
        right_area = _polygon_area(right)
        if min(left_area, right_area) <= 0:
            overlaps.append(1.0)
            continue
        intersection, _polygon = cv2.intersectConvexConvex(left, right)
        overlaps.append(
            max(0.0, float(intersection)) / min(left_area, right_area)
        )
    return float(np.mean(overlaps)) if overlaps else 0.0


def _foreground_mask(gray: np.ndarray) -> np.ndarray:
    gray = np.asarray(gray)
    if gray.ndim == 3:
        gray = cv2.cvtColor(gray[:, :, :3], cv2.COLOR_RGB2GRAY)
    if gray.size == 0:
        return np.zeros_like(gray, dtype=np.uint8)
    height, width = gray.shape
    corner_size = max(1, min(6, min(height, width) // 6))
    corners = np.concatenate(
        [
            gray[:corner_size, :corner_size].reshape(-1),
            gray[:corner_size, -corner_size:].reshape(-1),
            gray[-corner_size:, :corner_size].reshape(-1),
            gray[-corner_size:, -corner_size:].reshape(-1),
        ]
    )
    threshold_type = (
        cv2.THRESH_BINARY_INV
        if float(np.median(corners)) >= 127.5
        else cv2.THRESH_BINARY
    )
    _threshold, foreground = cv2.threshold(
        gray.astype(np.uint8),
        0,
        1,
        threshold_type + cv2.THRESH_OTSU,
    )
    return foreground


def boundary_ink_density(
    gray: np.ndarray, boundaries: Sequence[float]
) -> float:
    """Mean foreground density on the requested vertical cut columns."""

    foreground = _foreground_mask(gray)
    if foreground.ndim != 2 or foreground.shape[1] == 0 or not boundaries:
        return 0.0
    columns = [
        int(np.clip(round(boundary), 0, foreground.shape[1] - 1))
        for boundary in boundaries
    ]
    return float(foreground[:, columns].mean())


def _box_sides(box: np.ndarray) -> Tuple[float, float]:
    points = np.asarray(box, dtype=np.float32).reshape(-1, 2)
    sides = [
        float(np.linalg.norm(points[index] - points[(index + 1) % 4]))
        for index in range(4)
    ]
    return min(sides), max(sides)


def _character_boxes(rows: Sequence[RowResult]) -> List[np.ndarray]:
    return [
        np.asarray(character["bbox"], dtype=np.float32).reshape(4, 2)
        for row in rows
        for character in row.characters
        if character.get("bbox") is not None
        and np.asarray(character["bbox"]).size == 8
    ]


def edge_touch_ink_density(
    image_rgb: np.ndarray, boxes: Sequence[np.ndarray]
) -> float:
    """Foreground density on saved character crops' left/right cut edges."""

    image_bgr = np.ascontiguousarray(
        np.asarray(image_rgb, dtype=np.uint8)[:, :, ::-1]
    )
    densities = []
    for box in boxes:
        try:
            crop = get_storage_crop(
                image_bgr, np.asarray(box, dtype=np.float32)
            )
        except (cv2.error, ValueError):
            continue
        if crop is None or crop.size == 0:
            continue
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        foreground = _foreground_mask(gray)
        if foreground.shape[1] == 1:
            densities.append(float(foreground[:, 0].mean()))
        else:
            densities.append(
                float(
                    np.concatenate(
                        [foreground[:, 0], foreground[:, -1]]
                    ).mean()
                )
            )
    return float(np.mean(densities)) if densities else 0.0


def character_boundary_ink_density(
    image_rgb: np.ndarray, rows: Sequence[RowResult]
) -> float:
    """Sample foreground on internal boundaries between adjacent boxes."""

    image_rgb = np.asarray(image_rgb, dtype=np.uint8)
    image_height, image_width = image_rgb.shape[:2]
    densities = []
    for row in rows:
        line_points = np.asarray(
            row.line_box, dtype=np.float32
        ).reshape(-1, 2)
        left = max(0, int(np.floor(line_points[:, 0].min())))
        top = max(0, int(np.floor(line_points[:, 1].min())))
        right = min(
            image_width,
            int(np.ceil(line_points[:, 0].max())) + 1,
        )
        bottom = min(
            image_height,
            int(np.ceil(line_points[:, 1].max())) + 1,
        )
        if right <= left or bottom <= top:
            continue
        foreground = _foreground_mask(
            image_rgb[top:bottom, left:right]
        )
        local_height, local_width = foreground.shape
        boxes = [
            np.asarray(character["bbox"], dtype=np.float32).reshape(4, 2)
            for character in row.characters
            if character.get("bbox") is not None
            and np.asarray(character["bbox"]).size == 8
        ]
        if len(boxes) < 2:
            continue
        centers = np.asarray([box.mean(axis=0) for box in boxes])
        vertical = (
            np.ptp(centers[:, 1]) > np.ptp(centers[:, 0])
            if len(centers) > 1
            else False
        )
        for current, following in zip(boxes, boxes[1:]):
            if vertical:
                start = (current[3] + following[0]) / 2.0
                end = (current[2] + following[1]) / 2.0
            else:
                start = (current[1] + following[0]) / 2.0
                end = (current[2] + following[3]) / 2.0
            sample_count = max(
                2, int(round(float(np.linalg.norm(end - start))))
            )
            sample_positions = (
                np.arange(sample_count, dtype=np.float32) + 0.5
            ) / sample_count
            xs = np.clip(
                np.rint(
                    start[0] + sample_positions * (end[0] - start[0])
                ).astype(int)
                - left,
                0,
                local_width - 1,
            )
            ys = np.clip(
                np.rint(
                    start[1] + sample_positions * (end[1] - start[1])
                ).astype(int)
                - top,
                0,
                local_height - 1,
            )
            densities.append(float(foreground[ys, xs].mean()))
    return float(np.mean(densities)) if densities else 0.0


def _draw_centered(
    draw: ImageDraw.ImageDraw,
    character: str,
    cell: Tuple[int, int, int, int],
    font: ImageFont.FreeTypeFont,
) -> None:
    left, top, right, bottom = cell
    text_box = draw.textbbox((0, 0), character, font=font)
    width = text_box[2] - text_box[0]
    height = text_box[3] - text_box[1]
    x = left + (right - left - width) / 2.0 - text_box[0]
    y = top + (bottom - top - height) / 2.0 - text_box[1]
    draw.text((x, y), character, fill=(0, 0, 0), font=font)


def _rect_box(left: int, top: int, right: int, bottom: int) -> np.ndarray:
    return np.asarray(
        [
            [left, top],
            [right, top],
            [right, bottom],
            [left, bottom],
        ],
        dtype=np.float32,
    )


def generate_synthetic_cases(font_path: str) -> Tuple[SyntheticCase, ...]:
    """Create deterministic horizontal/vertical Chinese ground-truth pages."""

    rng = random.Random(SYNTHETIC_SEED)
    characters = list("天地玄黄宇宙洪荒日月盈昃辰宿列张寒来暑往")
    rng.shuffle(characters)
    text = "".join(characters[:12])
    cell_size = 64
    margin = 24
    font = ImageFont.truetype(font_path, 50)

    horizontal = Image.new(
        "RGB",
        (margin * 2 + cell_size * len(text), margin * 2 + cell_size),
        "white",
    )
    horizontal_draw = ImageDraw.Draw(horizontal)
    horizontal_boxes = []
    for index, character in enumerate(text):
        cell = (
            margin + index * cell_size,
            margin,
            margin + (index + 1) * cell_size,
            margin + cell_size,
        )
        _draw_centered(horizontal_draw, character, cell, font)
        horizontal_boxes.append(_rect_box(*cell))

    vertical = Image.new(
        "RGB",
        (margin * 2 + cell_size, margin * 2 + cell_size * len(text)),
        "white",
    )
    vertical_draw = ImageDraw.Draw(vertical)
    vertical_boxes = []
    for index, character in enumerate(text):
        cell = (
            margin,
            margin + index * cell_size,
            margin + cell_size,
            margin + (index + 1) * cell_size,
        )
        _draw_centered(vertical_draw, character, cell, font)
        vertical_boxes.append(_rect_box(*cell))

    return (
        SyntheticCase(
            "synthetic-horizontal",
            np.asarray(horizontal),
            text,
            tuple(horizontal_boxes),
            "horizontal",
            cell_size,
        ),
        SyntheticCase(
            "synthetic-vertical",
            np.asarray(vertical),
            text,
            tuple(vertical_boxes),
            "vertical",
            cell_size,
        ),
    )


def normalized_text_similarity(expected: str, actual: str) -> float:
    """One minus normalized Levenshtein distance."""

    if not expected and not actual:
        return 1.0
    previous = list(range(len(actual) + 1))
    for expected_index, expected_char in enumerate(expected, start=1):
        current = [expected_index]
        for actual_index, actual_char in enumerate(actual, start=1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[actual_index] + 1,
                    previous[actual_index - 1]
                    + (expected_char != actual_char),
                )
            )
        previous = current
    distance = previous[-1]
    return max(0.0, 1.0 - distance / max(len(expected), len(actual), 1))


def _sort_characters(
    rows: Sequence[RowResult], orientation: str
) -> List[dict]:
    characters = [
        character
        for row in rows
        for character in row.characters
        if character.get("bbox") is not None
    ]
    axis = 0 if orientation == "horizontal" else 1
    return sorted(
        characters,
        key=lambda item: float(
            np.asarray(item["bbox"], dtype=np.float32)[:, axis].mean()
        ),
    )


def _synthetic_metrics(
    rows: Sequence[RowResult], case: SyntheticCase
) -> Dict[str, float]:
    characters = _sort_characters(rows, case.orientation)
    predicted_boxes = [
        np.asarray(item["bbox"], dtype=np.float32).reshape(4, 2)
        for item in characters
    ]
    predicted_text = "".join(item.get("char", "") for item in characters)
    expected_count = len(case.ground_truth_boxes)
    predicted_count = len(predicted_boxes)
    count_accuracy = (
        min(expected_count, predicted_count)
        / max(expected_count, predicted_count)
        if predicted_count
        else 0.0
    )

    pair_count = min(expected_count, predicted_count)
    matched_ious = [
        polygon_iou(predicted_boxes[index], case.ground_truth_boxes[index])
        for index in range(pair_count)
    ]
    axis = 0 if case.orientation == "horizontal" else 1
    boundary_errors = []
    for index in range(max(0, pair_count - 1)):
        predicted_left = predicted_boxes[index][:, axis].max()
        predicted_right = predicted_boxes[index + 1][:, axis].min()
        predicted_boundary = (predicted_left + predicted_right) / 2.0
        expected_boundary = case.ground_truth_boxes[index][:, axis].max()
        boundary_errors.append(abs(predicted_boundary - expected_boundary))

    return {
        "synthetic_count_accuracy": count_accuracy,
        "synthetic_text_similarity": normalized_text_similarity(
            case.text, predicted_text
        ),
        "synthetic_matched_iou": (
            float(np.mean(matched_ious)) if matched_ious else 0.0
        ),
        "synthetic_boundary_error_px": (
            float(np.mean(boundary_errors))
            if boundary_errors
            else float(case.cell_size)
        ),
        "synthetic_boundary_score": max(
            0.0,
            1.0
            - (
                float(np.mean(boundary_errors))
                if boundary_errors
                else float(case.cell_size)
            )
            / case.cell_size,
        ),
    }


def evaluate_result(
    image_rgb: np.ndarray,
    result: PipelineResult,
    synthetic_case: Optional[SyntheticCase] = None,
) -> Dict[str, float]:
    boxes = _character_boxes(result.rows)
    sides = [_box_sides(box) for box in boxes]
    aspect_ratios = [
        short_side / long_side
        for short_side, long_side in sides
        if long_side > 0
    ]
    overlap_rates = [
        adjacent_overlap_rate(
            [
                np.asarray(character["bbox"], dtype=np.float32)
                for character in row.characters
                if character.get("bbox") is not None
            ]
        )
        for row in result.rows
    ]
    space_char_count = float(
        sum(
            1
            for row in result.rows
            for character in row.characters
            # 空格框（v6 filler / legacy 字典空格）不是真实字符，
            # 总框数对比会把它误当成召回优势。
            if str(character.get("char", "")) == " "
        )
    )
    metrics = {
        "line_count": float(len(result.rows)),
        "char_count": float(len(boxes)),
        "space_char_count": space_char_count,
        "effective_char_count": float(len(boxes)) - space_char_count,
        "degenerate_count": float(
            sum(short_side <= 1.5 for short_side, _long_side in sides)
        ),
        "adjacent_overlap_rate": (
            float(np.mean(overlap_rates)) if overlap_rates else 0.0
        ),
        "median_square_ratio": (
            float(np.median(aspect_ratios)) if aspect_ratios else 0.0
        ),
        "aspect_outlier_rate": (
            float(np.mean(np.asarray(aspect_ratios) < 0.35))
            if aspect_ratios
            else 1.0
        ),
        "boundary_ink_density": character_boundary_ink_density(
            image_rgb, result.rows
        ),
        "edge_touch_ink_density": edge_touch_ink_density(
            image_rgb, boxes
        ),
        "det_ms": result.det_ms,
        "rec_ms": result.rec_ms,
        "total_ms": result.total_ms,
    }
    if synthetic_case is not None:
        metrics.update(_synthetic_metrics(result.rows, synthetic_case))
    return metrics


class PipelineRunner:
    """Load candidate backends once and run complete pipeline comparisons."""

    def __init__(self, model_root: Path):
        self.model_root = Path(model_root)
        self._detectors = {}
        self._recognizers = {}

    def detector(self, candidate: Candidate):
        if candidate.det_model not in self._detectors:
            if candidate.det_model == "legacy":
                handle = DBNET(str(self.model_root / "dbnet.onnx"))
            elif candidate.det_model.startswith("v6-"):
                tier = candidate.det_model.split("-", 1)[1]
                handle = PPOcrV6Detector(
                    self.model_root
                    / "ppocrv6"
                    / tier
                    / "inference.onnx",
                    limit_type="short",
                    max_side_limit=4000,
                )
            else:
                raise ValueError(
                    f"unsupported detector: {candidate.det_model}"
                )
            self._detectors[candidate.det_model] = handle
        return self._detectors[candidate.det_model]

    def recognizer(self, candidate: Candidate):
        key = (candidate.rec_model, candidate.char_box_strategy)
        if key not in self._recognizers:
            if candidate.rec_model == "crnn":
                handle = CRNNHandle(
                    str(self.model_root / "crnn_lite_lstm.onnx")
                )
            elif candidate.rec_model.startswith("v6-"):
                tier = candidate.rec_model.split("-", 1)[1]
                handle = PPOcrV6RecHandle(
                    str(
                        self.model_root
                        / "ppocrv6_rec"
                        / tier
                        / "inference.onnx"
                    ),
                    char_box_strategy=candidate.char_box_strategy,
                )
            else:
                raise ValueError(
                    f"unsupported recognizer: {candidate.rec_model}"
                )
            self._recognizers[key] = handle
        return self._recognizers[key]

    def run(
        self,
        candidate: Candidate,
        image: Image.Image,
        limit_side_len: int = 960,
    ) -> PipelineResult:
        rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
        detector = self.detector(candidate)
        recognizer = self.recognizer(candidate)

        started = time.perf_counter()
        if candidate.det_model == "legacy":
            detector_input = np.ascontiguousarray(rgb[:, :, ::-1])
        else:
            detector_input = rgb
        boxes, scores = detector.process(detector_input, limit_side_len)
        det_ms = (time.perf_counter() - started) * 1000.0

        bgr = np.ascontiguousarray(rgb[:, :, ::-1])
        recognition_items = []
        valid_lines = []
        for box, score in zip(boxes, scores):
            box = np.asarray(box, dtype=np.float32).reshape(4, 2)
            try:
                crop_bgr = get_recognition_crop(bgr, box)
            except (cv2.error, ValueError):
                continue
            if crop_bgr is None or crop_bgr.size == 0:
                continue
            crop_rgb = Image.fromarray(
                np.ascontiguousarray(crop_bgr[:, :, ::-1])
            )
            recognition_items.append((crop_rgb, copy.deepcopy(box)))
            valid_lines.append((box, float(score)))

        started = time.perf_counter()
        recognized_lines = []
        for recognition_item, line in zip(
            recognition_items, valid_lines
        ):
            try:
                recognition_result = recognizer.predict_rbg(
                    *recognition_item
                )
            except Exception as exc:
                logger.warning(
                    "candidate %s skipped one recognition row: %s",
                    candidate.name,
                    exc,
                )
                continue
            recognized_lines.append((line, recognition_result))
        rec_ms = (time.perf_counter() - started) * 1000.0
        rows = [
            RowResult(
                line_box=box,
                score=score,
                text=result.get("line_pred", ""),
                characters=result.get("char_list", []),
            )
            for (box, score), result in recognized_lines
        ]
        return PipelineResult(rows=rows, det_ms=det_ms, rec_ms=rec_ms)


def _median_metrics(repeated: Sequence[Dict[str, float]]) -> Dict[str, float]:
    keys = repeated[0].keys()
    return {
        key: float(statistics.median(item[key] for item in repeated))
        for key in keys
    }


def _aggregate_candidate(
    case_metrics: Sequence[Dict[str, float]],
) -> Dict[str, float]:
    keys = set().union(*(metrics.keys() for metrics in case_metrics))
    return {
        key: float(
            np.mean(
                [metrics[key] for metrics in case_metrics if key in metrics]
            )
        )
        for key in sorted(keys)
    }


def quality_score(metrics: Dict[str, float]) -> float:
    """Quality-first score; latency is a tie-breaker, not a quality proxy."""

    char_count = max(metrics.get("char_count", 0.0), 1.0)
    nondegenerate = 1.0 - min(
        1.0, metrics.get("degenerate_count", 0.0) / char_count
    )
    return (
        0.20 * metrics.get("synthetic_count_accuracy", 0.0)
        + 0.20 * metrics.get("synthetic_text_similarity", 0.0)
        + 0.20 * metrics.get("synthetic_matched_iou", 0.0)
        + 0.15 * metrics.get("synthetic_boundary_score", 0.0)
        + 0.10
        * (1.0 - min(1.0, metrics.get("adjacent_overlap_rate", 1.0)))
        + 0.07
        * (1.0 - min(1.0, metrics.get("boundary_ink_density", 1.0)))
        + 0.03
        * (1.0 - min(1.0, metrics.get("edge_touch_ink_density", 1.0)))
        + 0.05 * nondegenerate
    )


def select_winner(
    aggregate: Dict[str, Dict[str, float]]
) -> Tuple[str, Dict[str, float]]:
    ranked = []
    for name, metrics in aggregate.items():
        score = quality_score(metrics)
        ranked.append((score, -metrics["total_ms"], name, metrics))
    _score, _negative_latency, name, metrics = max(ranked)
    return name, {**metrics, "quality_score": _score}


def _render(
    image: Image.Image,
    result: PipelineResult,
    output_path: Path,
    title: str,
) -> None:
    canvas = image.convert("RGB").copy()
    draw = ImageDraw.Draw(canvas)
    for row in result.rows:
        line_points = [tuple(point) for point in row.line_box]
        draw.line(
            line_points + [line_points[0]], fill=(0, 80, 255), width=3
        )
        for character in row.characters:
            if character.get("bbox") is None:
                continue
            points = [
                tuple(point)
                for point in np.asarray(character["bbox"]).reshape(4, 2)
            ]
            draw.line(points + [points[0]], fill=(0, 180, 40), width=1)
    draw.rectangle((0, 0, canvas.width, 24), fill=(0, 0, 0))
    draw.text((4, 4), title, fill=(255, 255, 255))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)


def run_benchmark(
    model_root: Path,
    image_paths: Sequence[Path],
    font_path: Path,
    output_dir: Path,
    repeat: int,
) -> Tuple[dict, dict]:
    runner = PipelineRunner(model_root)
    real_cases = [
        (path.stem, Image.open(path).convert("RGB"), None)
        for path in image_paths
        if path.exists()
    ]
    synthetic_cases = [
        (
            case.name,
            Image.fromarray(case.image),
            case,
        )
        for case in generate_synthetic_cases(str(font_path))
    ]
    cases = real_cases + synthetic_cases
    details = {}
    aggregates = {}

    for candidate in CANDIDATES:
        candidate_details = {}
        candidate_metrics = []
        for case_name, image, synthetic_case in cases:
            runner.run(candidate, image)
            repeated_results = [
                runner.run(candidate, image) for _index in range(repeat)
            ]
            repeated_metrics = [
                evaluate_result(
                    np.asarray(image),
                    result,
                    synthetic_case=synthetic_case,
                )
                for result in repeated_results
            ]
            metrics = _median_metrics(repeated_metrics)
            candidate_details[case_name] = metrics
            candidate_metrics.append(metrics)
            _render(
                image,
                repeated_results[-1],
                output_dir / f"{case_name}__{candidate.name}.png",
                (
                    f"{candidate.name} | "
                    f"{int(metrics['line_count'])} lines "
                    f"{int(metrics['char_count'])} chars "
                    f"{metrics['total_ms']:.0f} ms"
                ),
            )
        details[candidate.name] = candidate_details
        aggregates[candidate.name] = _aggregate_candidate(candidate_metrics)

    winner_name, winner_metrics = select_winner(aggregates)
    winner = next(
        candidate for candidate in CANDIDATES if candidate.name == winner_name
    )
    recommendation = {
        **asdict(winner),
        "quality_score": winner_metrics["quality_score"],
        "aggregate_metrics": aggregates[winner_name],
    }
    summary = {
        "seed": SYNTHETIC_SEED,
        "repeat": repeat,
        "candidates": [asdict(candidate) for candidate in CANDIDATES],
        "details": details,
        "aggregate": aggregates,
        "recommendation": recommendation,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return summary, recommendation


def _print_summary(summary: dict) -> None:
    header = (
        f"{'candidate':24s} {'quality':>8s} {'chars':>8s} "
        f"{'overlap':>9s} {'cut_ink':>9s} {'syn_text':>9s} "
        f"{'syn_iou':>8s} {'total_ms':>10s}"
    )
    print(header)
    print("-" * len(header))
    for candidate in CANDIDATES:
        metrics = summary["aggregate"][candidate.name]
        print(
            f"{candidate.name:24s} "
            f"{quality_score(metrics):8.4f} "
            f"{metrics['char_count']:8.1f} "
            f"{metrics['adjacent_overlap_rate']:9.4f} "
            f"{metrics['boundary_ink_density']:9.4f} "
            f"{metrics.get('synthetic_text_similarity', 0.0):9.4f} "
            f"{metrics.get('synthetic_matched_iou', 0.0):8.4f} "
            f"{metrics['total_ms']:10.1f}"
        )
    print(
        "\nrecommended:",
        summary["recommendation"]["name"],
        f"(quality={summary['recommendation']['quality_score']:.4f})",
    )


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-root", type=Path, default=Path("models"))
    parser.add_argument(
        "--images",
        nargs="*",
        type=Path,
        default=[Path(path) for path in DEFAULT_IMAGES],
    )
    parser.add_argument(
        "--font", type=Path, default=Path("仿宋_GB2312.ttf")
    )
    parser.add_argument(
        "--output", type=Path, default=Path("bench/output_pipeline")
    )
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--recommendation-json", type=Path)
    args = parser.parse_args(argv)
    if args.repeat < 1:
        parser.error("--repeat must be at least 1")

    summary, recommendation = run_benchmark(
        model_root=args.model_root,
        image_paths=args.images,
        font_path=args.font,
        output_dir=args.output,
        repeat=args.repeat,
    )
    _print_summary(summary)
    if args.recommendation_json is not None:
        args.recommendation_json.parent.mkdir(parents=True, exist_ok=True)
        args.recommendation_json.write_text(
            json.dumps(recommendation, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
