import numpy as np

from bench.compare_pipeline import (
    CANDIDATES,
    PipelineRunner,
    RowResult,
    adjacent_overlap_rate,
    boundary_ink_density,
    character_boundary_ink_density,
    generate_synthetic_cases,
)
from PIL import Image


def box(left, top, right, bottom):
    return np.asarray(
        [
            [left, top],
            [right, top],
            [right, bottom],
            [left, bottom],
        ],
        dtype=np.float32,
    )


def test_overlap_rate_is_zero_for_touching_boxes():
    assert adjacent_overlap_rate(
        [box(0, 0, 10, 10), box(10, 0, 20, 10)]
    ) == 0.0


def test_overlap_rate_detects_intersection():
    assert adjacent_overlap_rate(
        [box(0, 0, 12, 10), box(10, 0, 20, 10)]
    ) > 0.0


def test_boundary_ink_density_prefers_white_cut():
    gray = np.full((10, 30), 255, np.uint8)
    gray[:, 5:12] = 0
    gray[:, 18:25] = 0

    assert boundary_ink_density(gray, [15]) == 0.0


def test_boundary_ink_density_handles_white_text_on_dark_background():
    gray = np.full((10, 30), 25, np.uint8)
    gray[:, 5:12] = 225
    gray[:, 18:25] = 225

    assert boundary_ink_density(gray, [15]) == 0.0
    assert boundary_ink_density(gray, [8]) == 1.0


def test_character_boundary_density_uses_each_rows_local_polarity():
    image = np.full((40, 80, 3), 245, np.uint8)
    image[5:35, 10:70] = 25
    image[9:31, 14:34] = 225
    image[9:31, 46:66] = 225
    row = RowResult(
        line_box=box(10, 5, 70, 35),
        score=1.0,
        text="甲乙",
        characters=[
            {"char": "甲", "bbox": box(10, 5, 40, 35)},
            {"char": "乙", "bbox": box(40, 5, 70, 35)},
        ],
    )

    assert character_boundary_ink_density(image, [row]) == 0.0


def test_candidate_matrix_covers_legacy_hybrid_and_full_v6():
    names = [candidate.name for candidate in CANDIDATES]
    assert names == [
        "legacy",
        "hybrid-v6-rec",
        "full-v6-midpoint",
        "full-v6-projection",
        "full-v6-polarity",
    ]


def test_synthetic_cases_are_deterministic_and_have_ground_truth():
    first = generate_synthetic_cases("仿宋_GB2312.ttf")
    second = generate_synthetic_cases("仿宋_GB2312.ttf")

    assert [case.name for case in first] == ["synthetic-horizontal", "synthetic-vertical"]
    assert [case.text for case in first] == [case.text for case in second]
    assert all(len(case.ground_truth_boxes) == len(case.text) for case in first)
    assert all(
        np.array_equal(left.image, right.image)
        for left, right in zip(first, second)
    )


def test_runner_isolates_a_single_recognition_failure(monkeypatch, tmp_path):
    class FakeDetector:
        def process(self, _image, _limit):
            return [box(0, 0, 20, 20)], [0.9]

    class FailingRecognizer:
        def predict_rbg(self, _crop, _bbox):
            raise ValueError("degenerate geometry")

    runner = PipelineRunner(tmp_path)
    monkeypatch.setattr(runner, "detector", lambda _candidate: FakeDetector())
    monkeypatch.setattr(
        runner, "recognizer", lambda _candidate: FailingRecognizer()
    )

    result = runner.run(
        CANDIDATES[0], Image.new("RGB", (30, 30), "white")
    )

    assert result.rows == []
