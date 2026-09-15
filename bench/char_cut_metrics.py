"""Pure metrics and schema validation for visually annotated character cuts."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
from typing import Iterable, Sequence, Tuple

import cv2
import numpy as np


Polygon = np.ndarray


def safe_artifact_name(value: str, *, max_length: int = 80) -> str:
    """Return a portable, stable filename stem with a collision suffix."""

    if max_length < 16:
        raise ValueError("max_length must be at least 16")
    value = str(value)
    ascii_value = value.encode("ascii", errors="ignore").decode("ascii")
    slug = re.sub(r"[^A-Za-z0-9]+", "-", ascii_value).strip("-").lower()
    if not slug:
        slug = "case"
    digest = hashlib.sha256(value.encode("utf-8")).hexdigest()[:10]
    slug_limit = max_length - len(digest) - 1
    return f"{slug[:slug_limit].rstrip('-')}-{digest}"


def candidate_rank_key(metrics: dict) -> tuple:
    """Quality-first lexicographic key for boundary candidates."""

    severed_ink = float(metrics.get("severed_ink_rate", 1.0))
    boundary_error = float(
        metrics.get("mean_boundary_error_px", float("inf"))
    )
    return (
        int(metrics.get("matched_count", 0)),
        -severed_ink,
        -boundary_error,
        float(metrics.get("mean_iou", 0.0)),
        -float(metrics.get("runtime_ms", float("inf"))),
    )


def select_best_candidate(candidate_metrics: dict) -> tuple[str, dict]:
    """Return the deterministic winner and its metric mapping."""

    if not candidate_metrics:
        raise ValueError("candidate_metrics must not be empty")
    name = max(
        sorted(candidate_metrics),
        key=lambda candidate_name: candidate_rank_key(
            candidate_metrics[candidate_name]
        ),
    )
    return name, candidate_metrics[name]


@dataclass(frozen=True)
class GroundTruthCase:
    case_id: str
    image: str
    split: str
    polygons: Tuple[Polygon, ...]


@dataclass(frozen=True)
class GroundTruth:
    version: int
    cases: Tuple[GroundTruthCase, ...]


@dataclass(frozen=True)
class PolygonMatch:
    truth_index: int
    prediction_index: int
    iou: float
    boundary_error_px: float


@dataclass(frozen=True)
class PolygonMetrics:
    truth_count: int
    prediction_count: int
    matched_count: int
    missed_count: int
    extra_count: int
    mean_iou: float
    mean_boundary_error_px: float
    p95_boundary_error_px: float


def _as_polygon(points: Iterable[Iterable[float]]) -> Polygon:
    polygon = np.asarray(points, dtype=np.float32)
    if polygon.shape != (4, 2):
        raise ValueError("each polygon must contain exactly four points")
    if not np.isfinite(polygon).all():
        raise ValueError("polygon coordinates must be finite")
    if abs(float(cv2.contourArea(polygon))) <= 0:
        raise ValueError("polygon area must be positive")
    return polygon


def _order_quad(points: Iterable[Iterable[float]]) -> Polygon:
    polygon = _as_polygon(points)
    ordered = np.zeros((4, 2), dtype=np.float32)
    sums = polygon.sum(axis=1)
    differences = np.diff(polygon, axis=1).reshape(-1)
    ordered[0] = polygon[np.argmin(sums)]
    ordered[1] = polygon[np.argmin(differences)]
    ordered[2] = polygon[np.argmax(sums)]
    ordered[3] = polygon[np.argmax(differences)]
    return ordered


def _polygons_from_boundaries(item: dict) -> Tuple[Polygon, ...]:
    line_box = _order_quad(item.get("line_box"))
    orientation = str(item.get("orientation", "")).strip()
    if orientation not in {"horizontal", "vertical"}:
        raise ValueError(
            "boundary annotations require horizontal or vertical orientation"
        )
    boundaries = [float(value) for value in item.get("boundaries", ())]
    if len(boundaries) < 2 or any(
        right <= left for left, right in zip(boundaries, boundaries[1:])
    ):
        raise ValueError("boundaries must be strictly increasing")

    if orientation == "horizontal":
        edge_start = float(line_box[[0, 3], 0].mean())
        edge_end = float(line_box[[1, 2], 0].mean())
        first_edge = (line_box[0], line_box[1])
        second_edge = (line_box[3], line_box[2])
    else:
        edge_start = float(line_box[[0, 1], 1].mean())
        edge_end = float(line_box[[2, 3], 1].mean())
        first_edge = (line_box[0], line_box[3])
        second_edge = (line_box[1], line_box[2])
    if (
        edge_end <= edge_start
        or boundaries[0] < edge_start - 1e-3
        or boundaries[-1] > edge_end + 1e-3
    ):
        raise ValueError("boundaries must lie inside the line box")

    def edge_points(value: float):
        ratio = (value - edge_start) / (edge_end - edge_start)
        first = first_edge[0] + ratio * (first_edge[1] - first_edge[0])
        second = second_edge[0] + ratio * (
            second_edge[1] - second_edge[0]
        )
        return first, second

    cuts = [edge_points(value) for value in boundaries]
    polygons = []
    for left_or_top, right_or_bottom in zip(cuts, cuts[1:]):
        if orientation == "horizontal":
            top_left, bottom_left = left_or_top
            top_right, bottom_right = right_or_bottom
        else:
            top_left, top_right = left_or_top
            bottom_left, bottom_right = right_or_bottom
        polygons.append(
            np.asarray(
                [top_left, top_right, bottom_right, bottom_left],
                dtype=np.float32,
            )
        )
    return tuple(polygons)


def load_ground_truth(path: Path | str) -> GroundTruth:
    """Load and validate the versioned character-polygon annotation file."""

    with Path(path).open(encoding="utf-8") as stream:
        payload = json.load(stream)
    if not isinstance(payload, dict) or not isinstance(
        payload.get("cases"), list
    ):
        raise ValueError("ground truth must contain a cases list")

    cases = []
    seen_ids = set()
    for item in payload["cases"]:
        case_id = str(item.get("id", "")).strip()
        if not case_id:
            raise ValueError("case id must be non-empty")
        if case_id in seen_ids:
            raise ValueError(f"duplicate case id: {case_id}")
        seen_ids.add(case_id)
        split = str(item.get("split", "")).strip()
        if split not in {"train", "validation"}:
            raise ValueError(
                f"unsupported split for {case_id}: {split!r}"
            )
        if "polygons" in item:
            polygons = tuple(
                _as_polygon(points) for points in item.get("polygons", ())
            )
        elif "boundaries" in item:
            polygons = _polygons_from_boundaries(item)
        else:
            raise ValueError(
                f"case {case_id} must contain polygons or boundaries"
            )
        cases.append(
            GroundTruthCase(
                case_id=case_id,
                image=str(item.get("image", "")),
                split=split,
                polygons=polygons,
            )
        )
    return GroundTruth(
        version=int(payload.get("version", 0)),
        cases=tuple(cases),
    )


def polygon_iou(left: Polygon, right: Polygon) -> float:
    """Return convex quadrilateral intersection-over-union."""

    left = _as_polygon(left)
    right = _as_polygon(right)
    left_area = abs(float(cv2.contourArea(left)))
    right_area = abs(float(cv2.contourArea(right)))
    intersection, _polygon = cv2.intersectConvexConvex(left, right)
    union = left_area + right_area - float(intersection)
    return max(0.0, float(intersection)) / union if union > 0 else 0.0


def _boundary_error(left: Polygon, right: Polygon) -> float:
    """Mean corner distance after choosing the best cyclic alignment."""

    left = _as_polygon(left)
    right = _as_polygon(right)
    candidates = []
    for candidate in (right, right[::-1]):
        for shift in range(4):
            aligned = np.roll(candidate, shift, axis=0)
            candidates.append(
                float(np.linalg.norm(left - aligned, axis=1).mean())
            )
    return min(candidates)


def match_polygons(
    truth: Sequence[Polygon],
    predictions: Sequence[Polygon],
    *,
    min_iou: float = 0.25,
) -> Tuple[PolygonMatch, ...]:
    """Greedily match polygons one-to-one in descending IoU order."""

    if not 0.0 <= min_iou <= 1.0:
        raise ValueError("min_iou must be between 0 and 1")
    candidates = []
    for truth_index, truth_polygon in enumerate(truth):
        for prediction_index, prediction_polygon in enumerate(predictions):
            iou = polygon_iou(truth_polygon, prediction_polygon)
            if iou >= min_iou:
                candidates.append(
                    (iou, truth_index, prediction_index)
                )
    candidates.sort(key=lambda item: (-item[0], item[1], item[2]))

    matched_truth = set()
    matched_predictions = set()
    matches = []
    for iou, truth_index, prediction_index in candidates:
        if (
            truth_index in matched_truth
            or prediction_index in matched_predictions
        ):
            continue
        matched_truth.add(truth_index)
        matched_predictions.add(prediction_index)
        matches.append(
            PolygonMatch(
                truth_index=truth_index,
                prediction_index=prediction_index,
                iou=iou,
                boundary_error_px=_boundary_error(
                    truth[truth_index], predictions[prediction_index]
                ),
            )
        )
    matches.sort(key=lambda match: match.truth_index)
    return tuple(matches)


def evaluate_polygons(
    truth: Sequence[Polygon],
    predictions: Sequence[Polygon],
    *,
    min_iou: float = 0.25,
) -> PolygonMetrics:
    """Summarize one-to-one character-box agreement."""

    matches = match_polygons(truth, predictions, min_iou=min_iou)
    errors = [match.boundary_error_px for match in matches]
    return PolygonMetrics(
        truth_count=len(truth),
        prediction_count=len(predictions),
        matched_count=len(matches),
        missed_count=len(truth) - len(matches),
        extra_count=len(predictions) - len(matches),
        mean_iou=float(np.mean([match.iou for match in matches]))
        if matches
        else 0.0,
        mean_boundary_error_px=float(np.mean(errors)) if errors else 0.0,
        p95_boundary_error_px=float(np.percentile(errors, 95))
        if errors
        else 0.0,
    )


__all__ = [
    "candidate_rank_key",
    "GroundTruth",
    "GroundTruthCase",
    "PolygonMatch",
    "PolygonMetrics",
    "evaluate_polygons",
    "load_ground_truth",
    "match_polygons",
    "polygon_iou",
    "safe_artifact_name",
    "select_best_candidate",
]
