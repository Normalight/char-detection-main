import numpy as np
import pytest

from ocr.char_boxes import (
    adaptive_midpoint_boundaries,
    contrast_ink_map,
    estimate_border_background_luminance,
    midpoint_intervals,
    refine_boundaries_by_contrast,
    refine_boundaries_by_polarity,
    refine_boundaries_by_projection,
)


def test_midpoint_intervals_are_ordered_and_non_overlapping():
    intervals = midpoint_intervals([10.0, 30.0, 55.0], crop_width=70)

    assert intervals == [(0.0, 20.0), (20.0, 42.5), (42.5, 70.0)]
    assert all(left < right for left, right in intervals)


def test_midpoint_intervals_handle_single_character():
    assert midpoint_intervals([20.0], crop_width=50) == [(0.0, 50.0)]


def test_midpoint_intervals_reject_unsorted_centers():
    with pytest.raises(ValueError, match="sorted"):
        midpoint_intervals([20.0, 10.0], crop_width=50)


def test_adaptive_boundaries_trim_only_excess_outer_margin():
    assert adaptive_midpoint_boundaries(
        [10.0, 30.0, 55.0], crop_width=70
    ) == [0.0, 20.0, 42.5, 66.25]


def test_projection_refinement_moves_boundary_to_low_ink_valley():
    gray = np.full((20, 50), 255, dtype=np.uint8)
    gray[:, 3:27] = 0
    gray[:, 30:47] = 0

    refined = refine_boundaries_by_projection(
        gray,
        centers=[10.0, 40.0],
        boundaries=[0.0, 25.0, 50.0],
    )

    assert refined[1] != 25.0
    assert 27.0 <= refined[1] <= 29.0
    assert gray[:, int(refined[1])].mean() == 255


def test_projection_refinement_keeps_midpoint_without_improvement():
    gray = np.full((20, 50), 100, dtype=np.uint8)

    assert refine_boundaries_by_projection(
        gray,
        centers=[10.0, 40.0],
        boundaries=[0.0, 25.0, 50.0],
    ) == [0.0, 25.0, 50.0]


def test_background_luminance_detects_light_and_dark_polarity():
    light = np.full((24, 60, 3), 235, dtype=np.uint8)
    dark = np.full((24, 60, 3), 25, dtype=np.uint8)

    assert estimate_border_background_luminance(light) > 200
    assert estimate_border_background_luminance(dark) < 50


@pytest.mark.parametrize(
    "background, glyph",
    [
        (230, 75),
        (30, 205),
    ],
)
def test_contrast_ink_map_is_polarity_independent(background, glyph):
    image = np.full((30, 70, 3), background, dtype=np.uint8)
    image[4:26, 8:24] = glyph
    image[:, 50] = background + (12 if background < 128 else -12)

    ink = contrast_ink_map(image)

    assert ink.shape == image.shape[:2]
    assert ink.dtype == np.float32
    assert float(ink[7:23, 11:21].mean()) > 0.7
    assert float(ink[7:23, 34:45].mean()) < 0.1
    assert float(ink[:, 50].mean()) < 0.2


def test_dark_background_refinement_is_bounded_and_finds_safe_gap():
    image = np.full((30, 100, 3), 25, dtype=np.uint8)
    image[3:27, 4:39] = 215
    image[3:27, 47:94] = 215
    image[:, 44] = 42

    refined = refine_boundaries_by_contrast(
        image,
        centers=[20.0, 70.0],
        boundaries=[0.0, 45.0, 100.0],
    )

    assert 42.5 <= refined[1] <= 45.0
    assert abs(refined[1] - 45.0) <= 2.5
    assert image[:, int(round(refined[1]))].max() < 100


def test_dark_background_refinement_requires_real_contrast_gain():
    image = np.full((30, 100, 3), 30, dtype=np.uint8)
    image[:, 42:45] = 36

    assert refine_boundaries_by_contrast(
        image,
        centers=[20.0, 70.0],
        boundaries=[0.0, 45.0, 100.0],
    ) == [0.0, 45.0, 100.0]


def test_polarity_refinement_preserves_the_proven_light_background_path():
    image = np.full((20, 50, 3), 255, dtype=np.uint8)
    image[:, 3:27] = 0
    image[:, 30:47] = 0

    refined = refine_boundaries_by_polarity(
        image,
        centers=[10.0, 40.0],
        boundaries=[0.0, 25.0, 50.0],
    )

    assert 27.0 <= refined[1] <= 29.0


def test_polarity_refinement_keeps_ctc_boundary_on_dark_background():
    image = np.full((30, 100, 3), 25, dtype=np.uint8)
    image[3:27, 4:40] = 215
    image[3:27, 50:94] = 215
    image[:, 45] = 205
    boundaries = [0.0, 45.0, 100.0]

    refined = refine_boundaries_by_polarity(
        image,
        centers=[20.0, 70.0],
        boundaries=boundaries,
    )

    assert refined == boundaries


def test_polarity_refinement_keeps_order_and_minimum_width():
    image = np.full((24, 40, 3), 20, dtype=np.uint8)
    image[2:22, 1:12] = 220
    image[2:22, 14:26] = 220
    image[2:22, 28:39] = 220

    refined = refine_boundaries_by_polarity(
        image,
        centers=[6.0, 20.0, 34.0],
        boundaries=[0.0, 13.0, 27.0, 40.0],
        min_width=5.0,
    )

    assert all(
        right - left >= 5.0
        for left, right in zip(refined, refined[1:])
    )
    assert all(
        right > left for left, right in zip(refined, refined[1:])
    )
