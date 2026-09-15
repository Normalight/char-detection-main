import json
from pathlib import Path

import numpy as np
import pytest

from bench.char_cut_metrics import (
    candidate_rank_key,
    evaluate_polygons,
    load_ground_truth,
    match_polygons,
    safe_artifact_name,
    select_best_candidate,
)


def _box(left, top, right, bottom):
    return [
        [left, top],
        [left, bottom],
        [right, bottom],
        [right, top],
    ]


def test_load_ground_truth_normalizes_polygons_and_split(tmp_path):
    path = tmp_path / "truth.json"
    path.write_text(
        json.dumps(
            {
                "version": 1,
                "cases": [
                    {
                        "id": "page-0-row-1",
                        "image": "images/0.png",
                        "split": "validation",
                        "polygons": [_box(1, 2, 8, 10)],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    truth = load_ground_truth(path)

    assert truth.version == 1
    assert truth.cases[0].case_id == "page-0-row-1"
    assert truth.cases[0].split == "validation"
    assert truth.cases[0].polygons[0].dtype == np.float32
    np.testing.assert_array_equal(
        truth.cases[0].polygons[0],
        np.asarray(_box(1, 2, 8, 10), dtype=np.float32),
    )


def test_load_ground_truth_expands_auditable_line_boundaries(tmp_path):
    path = tmp_path / "truth.json"
    path.write_text(
        json.dumps(
            {
                "version": 1,
                "cases": [
                    {
                        "id": "horizontal-row",
                        "image": "images/0.png",
                        "split": "train",
                        "orientation": "horizontal",
                        "line_box": _box(0, 0, 30, 10),
                        "boundaries": [2, 12, 22, 28],
                    },
                    {
                        "id": "vertical-row",
                        "image": "images/1.png",
                        "split": "validation",
                        "orientation": "vertical",
                        "line_box": _box(0, 0, 10, 30),
                        "boundaries": [3, 13, 27],
                    },
                ],
            }
        ),
        encoding="utf-8",
    )

    truth = load_ground_truth(path)

    np.testing.assert_allclose(
        truth.cases[0].polygons,
        [
            [[2, 0], [12, 0], [12, 10], [2, 10]],
            [[12, 0], [22, 0], [22, 10], [12, 10]],
            [[22, 0], [28, 0], [28, 10], [22, 10]],
        ],
    )
    np.testing.assert_allclose(
        truth.cases[1].polygons,
        [
            [[0, 3], [10, 3], [10, 13], [0, 13]],
            [[0, 13], [10, 13], [10, 27], [0, 27]],
        ],
    )


@pytest.mark.parametrize(
    "mutator, message",
    [
        (
            lambda payload: payload["cases"].append(payload["cases"][0]),
            "duplicate case id",
        ),
        (
            lambda payload: payload["cases"][0].update(
                {"polygons": [[[0, 0], [1, 0], [1, 1]]]}
            ),
            "four points",
        ),
    ],
)
def test_load_ground_truth_rejects_ambiguous_annotations(
    tmp_path, mutator, message
):
    payload = {
        "version": 1,
        "cases": [
            {
                "id": "case-a",
                "image": "images/0.png",
                "split": "train",
                "polygons": [_box(0, 0, 10, 10)],
            }
        ],
    }
    mutator(payload)
    path = tmp_path / "truth.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        load_ground_truth(path)


def test_match_polygons_is_one_to_one_and_reports_unmatched_items():
    truth = [
        np.asarray(_box(0, 0, 10, 10), dtype=np.float32),
        np.asarray(_box(12, 0, 22, 10), dtype=np.float32),
    ]
    predictions = [
        np.asarray(_box(0, 0, 10, 10), dtype=np.float32),
        np.asarray(_box(1, 0, 11, 10), dtype=np.float32),
        np.asarray(_box(30, 0, 40, 10), dtype=np.float32),
    ]

    matches = match_polygons(truth, predictions, min_iou=0.5)

    assert [(item.truth_index, item.prediction_index) for item in matches] == [
        (0, 0)
    ]


def test_evaluate_polygons_reports_iou_boundary_error_and_counts():
    truth = [
        np.asarray(_box(0, 0, 10, 10), dtype=np.float32),
        np.asarray(_box(12, 0, 22, 10), dtype=np.float32),
    ]
    predictions = [
        np.asarray(_box(1, 0, 11, 10), dtype=np.float32),
        np.asarray(_box(12, 0, 22, 10), dtype=np.float32),
        np.asarray(_box(30, 0, 40, 10), dtype=np.float32),
    ]

    metrics = evaluate_polygons(truth, predictions, min_iou=0.5)

    assert metrics.matched_count == 2
    assert metrics.missed_count == 0
    assert metrics.extra_count == 1
    assert metrics.mean_iou == pytest.approx((9 / 11 + 1.0) / 2)
    assert metrics.mean_boundary_error_px == pytest.approx(0.5)
    assert metrics.p95_boundary_error_px == pytest.approx(0.95)


def test_safe_artifact_name_is_stable_portable_and_collision_resistant():
    source = "images/0.png#row:01 中文"

    first = safe_artifact_name(source)

    assert first == safe_artifact_name(source)
    assert first != safe_artifact_name("images-0.png#row:01 中文")
    assert first.isascii()
    assert all(character.isalnum() or character == "-" for character in first)
    assert len(first) <= 80


def test_committed_ground_truth_has_real_train_and_validation_cases():
    path = Path(__file__).parents[1] / "bench" / "char_cut_ground_truth.json"

    truth = load_ground_truth(path)

    assert truth.version == 1
    assert {case.split for case in truth.cases} == {"train", "validation"}
    assert len(truth.cases) >= 6
    assert sum(len(case.polygons) for case in truth.cases) >= 60
    assert all(case.image.startswith("images/") for case in truth.cases)
    train_images = {
        case.image for case in truth.cases if case.split == "train"
    }
    validation_images = {
        case.image for case in truth.cases if case.split == "validation"
    }
    assert train_images.isdisjoint(validation_images)


def test_candidate_rank_prioritizes_integrity_before_speed():
    safer = {
        "matched_count": 20,
        "severed_ink_rate": 0.01,
        "mean_boundary_error_px": 2.0,
        "mean_iou": 0.9,
        "runtime_ms": 8.0,
    }
    faster_but_clipped = {
        **safer,
        "severed_ink_rate": 0.08,
        "runtime_ms": 1.0,
    }

    assert candidate_rank_key(safer) > candidate_rank_key(
        faster_but_clipped
    )
    assert (
        select_best_candidate(
            {
                "faster": faster_but_clipped,
                "safer": safer,
            }
        )[0]
        == "safer"
    )


def test_candidate_rank_prioritizes_recall_before_other_metrics():
    complete = {
        "matched_count": 20,
        "severed_ink_rate": 0.02,
        "mean_boundary_error_px": 3.0,
        "mean_iou": 0.8,
        "runtime_ms": 9.0,
    }
    incomplete = {
        "matched_count": 19,
        "severed_ink_rate": 0.0,
        "mean_boundary_error_px": 0.0,
        "mean_iou": 1.0,
        "runtime_ms": 1.0,
    }

    assert candidate_rank_key(complete) > candidate_rank_key(incomplete)


def test_candidate_rank_prioritizes_severed_ink_before_boundary_error():
    closer_to_truth = {
        "matched_count": 20,
        "severed_ink_rate": 0.02,
        "mean_boundary_error_px": 2.0,
        "mean_iou": 0.9,
        "runtime_ms": 8.0,
    }
    blank_but_displaced = {
        **closer_to_truth,
        "severed_ink_rate": 0.001,
        "mean_boundary_error_px": 9.0,
    }

    assert candidate_rank_key(blank_but_displaced) > candidate_rank_key(
        closer_to_truth
    )
