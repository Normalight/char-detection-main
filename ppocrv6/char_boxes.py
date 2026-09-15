"""Pure geometry helpers for turning CTC token positions into character boxes."""

from typing import Iterable, List, Sequence, Tuple

import cv2
import numpy as np


Interval = Tuple[float, float]


def midpoint_boundaries(
    centers: Iterable[float], crop_width: float
) -> List[float]:
    """Return crop edges plus midpoint boundaries between ordered centers."""

    crop_width = float(crop_width)
    if crop_width <= 0:
        raise ValueError("crop_width must be positive")
    values = [float(center) for center in centers]
    if any(right <= left for left, right in zip(values, values[1:])):
        raise ValueError("centers must be sorted and strictly increasing")
    if not values:
        return []
    if len(values) == 1:
        return [0.0, crop_width]

    boundaries = [0.0]
    boundaries.extend(
        min(crop_width, max(0.0, (left + right) / 2.0))
        for left, right in zip(values, values[1:])
    )
    boundaries.append(crop_width)
    return boundaries


def adaptive_midpoint_boundaries(
    centers: Sequence[float], crop_width: float
) -> List[float]:
    """Use midpoint cuts while trimming unsupported outer detector margin.

    PaddleOCR's official word-box path estimates Chinese character width from
    CTC-column spacing instead of assigning all detector padding to the first
    and last character.  Median spacing gives the same intent while retaining
    strictly shared, non-overlapping internal boundaries.
    """

    values = [float(center) for center in centers]
    boundaries = midpoint_boundaries(values, crop_width)
    if len(values) < 2:
        return boundaries
    half_spacing = float(np.median(np.diff(values))) / 2.0
    boundaries[0] = max(0.0, values[0] - half_spacing)
    boundaries[-1] = min(float(crop_width), values[-1] + half_spacing)
    return boundaries


def valid_interval_indices(
    centers: Sequence[float],
    crop_width: float,
    min_width: float = 1.5,
) -> List[Tuple[int, Interval]]:
    """Return source indices with non-degenerate midpoint intervals."""

    return valid_intervals_from_boundaries(
        midpoint_boundaries(centers, crop_width), min_width=min_width
    )


def valid_intervals_from_boundaries(
    boundaries: Sequence[float],
    min_width: float = 1.5,
) -> List[Tuple[int, Interval]]:
    """Return source indices with non-degenerate adjacent boundaries."""

    if min_width < 0:
        raise ValueError("min_width must be non-negative")
    values = [float(boundary) for boundary in boundaries]
    if any(right < left for left, right in zip(values, values[1:])):
        raise ValueError("boundaries must be sorted")
    return [
        (index, (left, right))
        for index, (left, right) in enumerate(
            zip(values, values[1:])
        )
        if right - left >= min_width
    ]


def midpoint_intervals(
    centers: Sequence[float],
    crop_width: float,
    min_width: float = 1.5,
) -> List[Interval]:
    """Partition a crop at CTC-center midpoints without adjacent overlap."""

    return [
        interval
        for _index, interval in valid_interval_indices(
            centers, crop_width, min_width=min_width
        )
    ]


def _as_grayscale(image: np.ndarray) -> np.ndarray:
    array = np.asarray(image)
    if array.ndim == 2:
        return array.astype(np.uint8, copy=False)
    if array.ndim == 3 and array.shape[2] >= 3:
        rgb = array[:, :, :3].astype(np.uint8, copy=False)
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    raise ValueError(f"expected gray or RGB image, got shape {array.shape}")


def _as_rgb(image: np.ndarray) -> np.ndarray:
    array = np.asarray(image)
    if array.ndim == 2:
        return np.repeat(array[:, :, None], 3, axis=2).astype(
            np.uint8, copy=False
        )
    if array.ndim == 3 and array.shape[2] >= 3:
        return array[:, :, :3].astype(np.uint8, copy=False)
    raise ValueError(f"expected gray or RGB image, got shape {array.shape}")


def _border_pixels(image_rgb: np.ndarray) -> np.ndarray:
    height, width = image_rgb.shape[:2]
    border_size = max(1, min(4, min(height, width) // 8))
    return np.concatenate(
        [
            image_rgb[:border_size].reshape(-1, 3),
            image_rgb[-border_size:].reshape(-1, 3),
            image_rgb[:, :border_size].reshape(-1, 3),
            image_rgb[:, -border_size:].reshape(-1, 3),
        ],
        axis=0,
    )


def _corner_pixels(image_rgb: np.ndarray) -> np.ndarray:
    height, width = image_rgb.shape[:2]
    corner_size = max(1, min(6, min(height, width) // 6))
    return np.concatenate(
        [
            image_rgb[:corner_size, :corner_size].reshape(-1, 3),
            image_rgb[:corner_size, -corner_size:].reshape(-1, 3),
            image_rgb[-corner_size:, :corner_size].reshape(-1, 3),
            image_rgb[-corner_size:, -corner_size:].reshape(-1, 3),
        ],
        axis=0,
    )


def estimate_border_background_luminance(image: np.ndarray) -> float:
    """Estimate row-background luminance from a thin outer border."""

    rgb = _as_rgb(image)
    background = np.median(
        _corner_pixels(rgb).astype(np.float32), axis=0
    )
    return float(
        0.299 * background[0]
        + 0.587 * background[1]
        + 0.114 * background[2]
    )


def contrast_ink_map(image: np.ndarray) -> np.ndarray:
    """Return polarity-independent glyph confidence in the range 0..1."""

    gray = _as_grayscale(image).astype(np.float32) / 255.0
    height, width = gray.shape
    border_size = max(1, min(4, min(height, width) // 8))
    border = np.concatenate(
        [
            gray[:border_size].reshape(-1),
            gray[-border_size:].reshape(-1),
            gray[:, :border_size].reshape(-1),
            gray[:, -border_size:].reshape(-1),
        ]
    )
    background = float(np.median(border))
    distance = np.abs(gray - background)
    border_distance = np.abs(border - background)
    noise_floor = float(np.percentile(border_distance, 85))
    signal_level = float(np.percentile(distance, 99.5))
    scale = max(signal_level - noise_floor, 1e-6)
    if scale < 0.06:
        return np.zeros(distance.shape, dtype=np.float32)
    confidence = np.clip((distance - noise_floor) / scale, 0.0, 1.0)
    return cv2.GaussianBlur(
        confidence.astype(np.float32), (3, 3), 0
    ).astype(np.float32, copy=False)


def refine_boundaries_by_projection(
    image: np.ndarray,
    centers: Sequence[float],
    boundaries: Sequence[float],
    *,
    min_width: float = 1.5,
    improvement_ratio: float = 0.10,
) -> List[float]:
    """Move internal cuts to nearby low-ink valleys when evidence is stronger.

    Each cut searches only the central 35%–65% portion of its two adjacent CTC
    centers.  This guard prevents a low-ink margin inside either glyph from
    pulling the cut far away from the CTC midpoint.
    """

    centers = [float(value) for value in centers]
    refined = [float(value) for value in boundaries]
    if not centers and not refined:
        return []
    if len(refined) != len(centers) + 1:
        raise ValueError("boundaries must contain one more value than centers")
    if any(right <= left for left, right in zip(centers, centers[1:])):
        raise ValueError("centers must be sorted and strictly increasing")
    if any(right <= left for left, right in zip(refined, refined[1:])):
        raise ValueError("boundaries must be sorted and strictly increasing")
    if min_width < 0:
        raise ValueError("min_width must be non-negative")
    if not 0.0 <= improvement_ratio < 1.0:
        raise ValueError("improvement_ratio must be between 0 and 1")
    if len(centers) < 2:
        return refined

    gray = _as_grayscale(image)
    _threshold, foreground = cv2.threshold(
        gray, 0, 1, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )
    projection = foreground.astype(np.float32).sum(axis=0)
    smoothed = np.convolve(
        projection, np.ones(3, dtype=np.float32) / 3.0, mode="same"
    )
    image_width = gray.shape[1]

    for boundary_index, (left_center, right_center) in enumerate(
        zip(centers, centers[1:]), start=1
    ):
        gap = right_center - left_center
        search_start = max(
            0,
            int(np.ceil(left_center + 0.35 * gap)),
            int(np.ceil(refined[boundary_index - 1] + min_width)),
        )
        search_end = min(
            image_width - 1,
            int(np.floor(left_center + 0.65 * gap)),
            int(np.floor(refined[boundary_index + 1] - min_width)),
        )
        if search_start > search_end:
            continue

        midpoint_index = int(
            np.clip(round(refined[boundary_index]), 0, image_width - 1)
        )
        candidates = np.arange(search_start, search_end + 1)
        candidate_values = smoothed[candidates]
        best_value = float(candidate_values.min())
        best_candidates = candidates[
            np.isclose(candidate_values, best_value, rtol=0.0, atol=1e-6)
        ]
        best_index = int(
            best_candidates[
                np.argmin(np.abs(best_candidates - midpoint_index))
            ]
        )
        midpoint_value = float(smoothed[midpoint_index])
        required_value = midpoint_value * (1.0 - improvement_ratio)
        if (
            best_value < midpoint_value
            and best_value <= required_value
        ):
            refined[boundary_index] = float(best_index)

    return refined


def refine_boundaries_by_contrast(
    image: np.ndarray,
    centers: Sequence[float],
    boundaries: Sequence[float],
    *,
    min_width: float = 1.5,
    search_low: float = 0.10,
    search_high: float = 0.90,
    max_move_ratio: float = 0.05,
    distance_weight: float = 0.05,
    min_gain: float = 0.03,
) -> List[float]:
    """Conservatively refine cuts using polarity-independent local contrast."""

    centers = [float(value) for value in centers]
    refined = [float(value) for value in boundaries]
    if not centers and not refined:
        return []
    if len(refined) != len(centers) + 1:
        raise ValueError("boundaries must contain one more value than centers")
    if any(right <= left for left, right in zip(centers, centers[1:])):
        raise ValueError("centers must be sorted and strictly increasing")
    if any(right <= left for left, right in zip(refined, refined[1:])):
        raise ValueError("boundaries must be sorted and strictly increasing")
    if min_width < 0:
        raise ValueError("min_width must be non-negative")
    if not 0.0 <= search_low < search_high <= 1.0:
        raise ValueError("search fractions must satisfy 0 <= low < high <= 1")
    if not 0.0 <= max_move_ratio <= 0.5:
        raise ValueError("max_move_ratio must be between 0 and 0.5")
    if distance_weight < 0 or min_gain < 0:
        raise ValueError("distance_weight and min_gain must be non-negative")
    if len(centers) < 2:
        return refined

    ink = contrast_ink_map(image)
    projection = (
        0.65 * ink.mean(axis=0)
        + 0.35 * (ink >= 0.45).mean(axis=0)
    ).astype(np.float32)
    image_width = ink.shape[1]

    for boundary_index, (left_center, right_center) in enumerate(
        zip(centers, centers[1:]), start=1
    ):
        gap = right_center - left_center
        midpoint = float(refined[boundary_index])
        max_move = max_move_ratio * gap
        search_start = max(
            0,
            int(np.ceil(left_center + search_low * gap)),
            int(np.ceil(refined[boundary_index - 1] + min_width)),
            int(np.ceil(midpoint - max_move)),
        )
        search_end = min(
            image_width - 1,
            int(np.floor(left_center + search_high * gap)),
            int(np.floor(refined[boundary_index + 1] - min_width)),
            int(np.floor(midpoint + max_move)),
        )
        if search_start > search_end:
            continue

        midpoint_index = int(
            np.clip(round(midpoint), 0, image_width - 1)
        )
        candidates = np.arange(search_start, search_end + 1)
        distance = np.abs(candidates - midpoint) / max(gap, 1.0)
        objective = projection[candidates] + distance_weight * distance
        best_value = float(objective.min())
        best_candidates = candidates[
            np.isclose(objective, best_value, rtol=0.0, atol=1e-8)
        ]
        best_index = int(
            best_candidates[
                np.argmin(np.abs(best_candidates - midpoint))
            ]
        )
        if (
            float(projection[midpoint_index])
            - float(projection[best_index])
            >= min_gain
        ):
            refined[boundary_index] = float(best_index)

    return refined


def refine_boundaries_by_polarity(
    image: np.ndarray,
    centers: Sequence[float],
    boundaries: Sequence[float],
    *,
    min_width: float = 1.5,
    background_threshold: float = 127.5,
) -> List[float]:
    """Refine light rows and preserve CTC cuts on noisy dark rows."""

    center_values = [float(value) for value in centers]
    values = [float(value) for value in boundaries]
    if not center_values and not values:
        return []
    if len(values) != len(center_values) + 1:
        raise ValueError(
            "boundaries must contain one more value than centers"
        )
    if any(
        right <= left
        for left, right in zip(center_values, center_values[1:])
    ):
        raise ValueError("centers must be sorted and strictly increasing")
    if any(right <= left for left, right in zip(values, values[1:])):
        raise ValueError(
            "boundaries must be sorted and strictly increasing"
        )
    if min_width < 0:
        raise ValueError("min_width must be non-negative")

    if estimate_border_background_luminance(image) < background_threshold:
        return values
    return refine_boundaries_by_projection(
        image,
        center_values,
        values,
        min_width=min_width,
    )


def independent_stroke_boxes(
    image: np.ndarray, centers: Sequence[float], characters=None,
    *, allow_outer_padding=False,
):
    """Estimate independent left/top/right/bottom crop coordinates.

    Assign disconnected strokes by centroid proximity to CTC anchors, keeping
    complete component extents even across anchor midpoints. Joined components
    spanning multiple anchors use bounded overlapping windows. Weak foreground
    evidence retains an independent geometric prior, not a shared partition.
    """
    gray = _as_grayscale(image)
    height, width = gray.shape
    values = np.asarray(centers, dtype=float)
    if not height or not width:
        raise ValueError("image must be non-empty")
    if not np.all(np.isfinite(values)) or np.any(np.diff(values) <= 0):
        raise ValueError("centers must be finite, sorted and strictly increasing")
    if np.any(values < 0) or np.any(values >= width):
        raise ValueError("centers must be inside the image")
    if not len(values):
        return []

    gaps = np.diff(values)
    spacing = float(np.median(gaps)) if len(gaps) else min(width, height)
    windows = []
    for i, center in enumerate(values):
        before = gaps[i - 1] if i else spacing
        after = gaps[i] if i < len(gaps) else spacing
        left = max(0.0, center - 0.60 * min(before, height))
        right = min(float(width), center + 0.60 * min(after, height))
        windows.append((left, 0.0, right, float(height)))

    ink = contrast_ink_map(image)
    # Use raw low-contrast pixels for extents: thresholding a blurred confidence
    # map at 0.45 erodes thin serifs and antialiased stroke tips.
    background = estimate_border_background_luminance(image)
    distance = np.abs(gray.astype(np.float32) - background)
    threshold = max(8.0, float(np.percentile(distance, 99.5)) * 0.08)
    mask = (distance >= threshold).astype(np.uint8)
    if not mask.any() and float(np.ptp(gray)) >= 16:
        # Ink touching a crop edge can contaminate the border noise estimate.
        # Otsu with corner polarity recovers it without inventing ink on flat
        # crops; the density guard below still rejects a background mask.
        threshold, bright = cv2.threshold(
            gray, 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
        background = estimate_border_background_luminance(image)
        mask = 1 - bright if background > threshold else bright
    # Dense texture is insufficient evidence to shrink a glyph to fragments.
    if not mask.any() or mask.mean() > 0.65:
        return windows
    count, labels, stats, centroids = cv2.connectedComponentsWithStats(
        mask, connectivity=8
    )
    evidence = [[] for _ in values]
    min_area = max(2, int(height * height * 0.001))
    padding = max(2.0, height * 0.10)
    for component in range(1, count):
        x, y, w, h, area = stats[component]
        if area < min_area:
            continue
        inside = np.flatnonzero((values >= x) & (values < x + w))
        if len(inside) > 1:
            # Connectivity cannot disambiguate joined glyphs. Keep bounded
            # overlapping portions, never force a shared cut.
            for i in inside:
                left, _, right, _ = windows[i]
                lo = max(x, int(np.floor(left)))
                hi = min(x + w, int(np.ceil(right)))
                ys, xs = np.nonzero(labels[y:y + h, lo:hi] == component)
                if len(xs):
                    evidence[i].append((lo + xs.min(), y + ys.min(),
                                        lo + xs.max() + 1, y + ys.max() + 1))
            continue
        i = int(np.argmin(np.abs(values - centroids[component, 0])))
        left, _, right, _ = windows[i]
        if x + w > left and x < right:
            evidence[i].append((x, y, x + w, y + h))

    boxes = []
    for index, (prior, parts) in enumerate(zip(windows, evidence)):
        if not parts:
            boxes.append(prior)
            continue
        parts = np.asarray(parts, dtype=float)
        left, top = parts[:, :2].min(axis=0)
        right, bottom = parts[:, 2:].max(axis=0)
        char = characters[index] if characters is not None else ""
        if any("\u3400" <= value <= "\u9fff" for value in char):
            # CJK radicals may be assigned to a neighbouring CTC anchor. Do
            # not shrink a full-size character to its strongest component.
            left, right = min(left, prior[0]), max(right, prior[2])
            middle = (top + bottom) / 2.0
            top = min(top, middle - height * 0.425)
            bottom = max(bottom, middle + height * 0.425)
        outer = padding if allow_outer_padding else 0.0
        boxes.append((max(-outer, left - padding),
                      max(-outer, top - padding),
                      min(float(width) + outer, right + padding),
                      min(float(height) + outer, bottom + padding)))
    return boxes


__all__ = [
    "independent_stroke_boxes",
    "Interval",
    "adaptive_midpoint_boundaries",
    "contrast_ink_map",
    "estimate_border_background_luminance",
    "midpoint_boundaries",
    "midpoint_intervals",
    "refine_boundaries_by_contrast",
    "refine_boundaries_by_polarity",
    "refine_boundaries_by_projection",
    "valid_interval_indices",
    "valid_intervals_from_boundaries",
]
