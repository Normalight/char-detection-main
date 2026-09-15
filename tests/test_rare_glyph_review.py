# -*- coding: utf-8 -*-
"""轻量生僻字开放集复核的纯逻辑测试。"""

import hashlib
import json
import math
import os

import numpy as np
from PIL import Image
import pytest

import rare_glyph_review
from rare_glyph_review import (
    LazyOnnxGlyphReviewer,
    OnnxGlyphReviewer,
    REVIEW_COLOR_HIGH,
    REVIEW_COLOR_LOW,
    REVIEW_COLOR_MEDIUM,
    choose_glyph_suggestion,
    normalize_candidates,
    review_color_level,
)


def test_normalize_candidates_filters_invalid_deduplicates_and_ranks():
    candidates = normalize_candidates(
        [
            {"char": " 浇 ", "score": 0.71},
            {"char": "澆", "score": 0.82},
            {"char": "浇", "score": 0.77},
            {"char": "两个", "score": 0.99},
            {"char": "?", "score": 0.95},
            {"char": "\x00", "score": 0.94},
            {"char": "灌", "score": float("nan")},
            {"char": "", "score": 0.90},
        ],
        limit=5,
    )

    assert [candidate.char for candidate in candidates] == ["澆", "浇"]
    assert [candidate.score for candidate in candidates] == [0.82, 0.77]
    assert [candidate.rank for candidate in candidates] == [1, 2]


def test_normalize_candidates_clamps_scores_to_similarity_range():
    candidates = normalize_candidates(
        [
            {"char": "甲", "score": 2.0},
            {"char": "乙", "score": -0.5},
        ]
    )

    assert candidates[0].score == 1.0
    assert candidates[1].score == 0.0


def test_open_set_suggestion_uses_top_candidate_and_margin():
    suggestion = choose_glyph_suggestion(
        status="low_confidence",
        primary_text="浇",
        primary_score=0.43,
        review_candidates=[
            {"char": "澆", "score": 0.81},
            {"char": "浇", "score": 0.69},
            {"char": "滰", "score": 0.51},
        ],
    )

    assert suggestion is not None
    assert suggestion.text == "澆"
    assert suggestion.score == 0.81
    assert math.isclose(suggestion.margin, 0.12)
    assert suggestion.source == "open_set"
    assert suggestion.review_required is True


def test_primary_single_character_is_safe_fallback_when_model_has_no_candidate():
    suggestion = choose_glyph_suggestion(
        status="missing_confidence",
        primary_text="𠮷",
        primary_score=None,
        review_candidates=[],
    )

    assert suggestion is not None
    assert suggestion.text == "𠮷"
    assert suggestion.score is None
    assert suggestion.margin is None
    assert suggestion.source == "primary"
    assert suggestion.review_required is True


def test_no_valid_candidate_does_not_force_a_group():
    suggestion = choose_glyph_suggestion(
        status="review",
        primary_text="两个",
        primary_score=0.9,
        review_candidates=[{"char": "?", "score": 0.99}],
    )

    assert suggestion is None


def test_non_han_candidates_do_not_become_rare_chinese_group_suggestions():
    suggestion = choose_glyph_suggestion(
        status="review",
        primary_text="a",
        primary_score=0.91,
        review_candidates=[
            {"char": "a", "score": 0.99},
            {"char": "汎", "score": 0.80},
        ],
    )

    assert suggestion is not None
    assert suggestion.text == "汎"
    assert suggestion.source == "open_set"

    assert (
        choose_glyph_suggestion(
            status="review",
            primary_text="a",
            primary_score=0.91,
            review_candidates=[],
        )
        is None
    )


def test_recognized_record_is_not_re_reviewed():
    suggestion = choose_glyph_suggestion(
        status="recognized",
        primary_text="字",
        primary_score=0.99,
        review_candidates=[{"char": "宇", "score": 0.95}],
    )

    assert suggestion is None


def test_review_color_levels_use_similarity_margin_and_fallback():
    assert (
        review_color_level(score=0.80, margin=0.10, source="open_set")
        == REVIEW_COLOR_HIGH
    )
    assert (
        review_color_level(score=0.80, margin=0.03, source="open_set")
        == REVIEW_COLOR_MEDIUM
    )
    assert (
        review_color_level(score=0.60, margin=0.20, source="open_set")
        == REVIEW_COLOR_MEDIUM
    )
    assert (
        review_color_level(score=0.54, margin=0.20, source="open_set")
        == REVIEW_COLOR_LOW
    )
    assert (
        review_color_level(score=0.99, margin=0.50, source="primary")
        == REVIEW_COLOR_LOW
    )


class _FakeNode:
    def __init__(self, name):
        self.name = name


class _FakeSession:
    def __init__(self, output):
        self.output = np.asarray(output, dtype=np.float32)
        self.inputs = []

    def get_inputs(self):
        return [_FakeNode("image")]

    def get_outputs(self):
        return [_FakeNode("embedding")]

    def run(self, output_names, feeds):
        self.inputs.append((output_names, feeds))
        return [self.output]


class _FakeBatchSession(_FakeSession):
    @property
    def run_calls(self):
        return len(self.inputs)


def _write_reviewer_assets(
    folder,
    *,
    characters=("甲", "乙", "丙", "丁"),
    prototypes=None,
    manifest_updates=None,
):
    model_path = os.path.join(folder, "glyph_encoder.onnx")
    with open(model_path, "wb") as model_file:
        model_file.write(b"fake onnx")
    model_sha256 = hashlib.sha256(b"fake onnx").hexdigest()

    with open(
        os.path.join(folder, "characters.txt"),
        "w",
        encoding="utf-8",
    ) as character_file:
        character_file.write("\n".join(characters) + "\n")
    character_path = os.path.join(folder, "characters.txt")
    with open(character_path, "rb") as character_file:
        characters_sha256 = hashlib.sha256(character_file.read()).hexdigest()

    if prototypes is None:
        prototypes = np.asarray(
            [
                [1.0, 0.0],
                [0.8, 0.6],
                [-1.0, 0.0],
                [0.0, 1.0],
            ],
            dtype=np.float32,
        )
    prototype_path = os.path.join(folder, "prototypes.npy")
    np.save(prototype_path, prototypes)
    with open(prototype_path, "rb") as prototype_file:
        prototypes_sha256 = hashlib.sha256(prototype_file.read()).hexdigest()

    manifest = {
        "version": 1,
        "model_id": "unit-test-reviewer",
        "prototype_version": "unit-test-v1",
        "source": "https://example.test/model",
        "license": "test-only",
        "model": "glyph_encoder.onnx",
        "model_sha256": model_sha256,
        "characters": "characters.txt",
        "characters_sha256": characters_sha256,
        "prototypes": "prototypes.npy",
        "prototypes_sha256": prototypes_sha256,
        "input": {
            "height": 8,
            "width": 8,
            "color": "RGB",
            "layout": "NCHW",
            "mean": [0.5, 0.5, 0.5],
            "std": [0.5, 0.5, 0.5],
        },
    }
    manifest.update(manifest_updates or {})
    with open(
        os.path.join(folder, "manifest.json"),
        "w",
        encoding="utf-8",
    ) as manifest_file:
        json.dump(manifest, manifest_file)


def test_onnx_reviewer_is_lazy_and_returns_chunked_cosine_top_k(tmp_path):
    _write_reviewer_assets(tmp_path)
    fake_session = _FakeSession([[1.0, 0.0]])
    factory_calls = []

    def session_factory(model_path):
        factory_calls.append(model_path)
        return fake_session

    reviewer = OnnxGlyphReviewer(
        str(tmp_path),
        session_factory=session_factory,
        prototype_block_size=2,
    )
    assert factory_calls == []

    candidates = reviewer.review(
        Image.new("RGB", (5, 7), "white"),
        top_k=3,
    )

    assert len(factory_calls) == 1
    assert [item["char"] for item in candidates] == ["甲", "乙", "丁"]
    assert [item["rank"] for item in candidates] == [1, 2, 3]
    assert candidates[0]["score"] == pytest.approx(1.0)
    assert candidates[1]["score"] == pytest.approx(0.9)
    assert candidates[2]["score"] == pytest.approx(0.5)

    output_names, feeds = fake_session.inputs[0]
    assert output_names == ["embedding"]
    assert list(feeds) == ["image"]
    tensor = feeds["image"]
    assert tensor.shape == (1, 3, 8, 8)
    assert tensor.dtype == np.float32
    assert np.isfinite(tensor).all()


def test_runtime_preprocess_uses_the_same_root_padding_helper_as_training(
    tmp_path,
):
    import glyph_preprocess
    from training.megahan_lite import preprocess as training_preprocess

    _write_reviewer_assets(tmp_path)
    reviewer = OnnxGlyphReviewer(str(tmp_path))
    crop = Image.new("RGB", (9, 3), (11, 22, 33))
    crop.putpixel((4, 1), (220, 210, 200))

    runtime_tensor = reviewer._preprocess(crop)
    padded = glyph_preprocess.pad_glyph_image(crop, (8, 8))
    expected = np.asarray(padded, dtype=np.float32) / 255.0
    expected = (expected - 0.5) / 0.5
    expected = np.ascontiguousarray(
        expected.transpose(2, 0, 1)[None]
    )

    assert (
        rare_glyph_review.pad_glyph_image
        is glyph_preprocess.pad_glyph_image
    )
    assert (
        training_preprocess.pad_glyph_image
        is glyph_preprocess.pad_glyph_image
    )
    assert np.array_equal(runtime_tensor, expected)


def test_onnx_reviewer_exposes_auditable_model_metadata(tmp_path):
    _write_reviewer_assets(tmp_path)
    reviewer = OnnxGlyphReviewer(
        str(tmp_path),
        session_factory=lambda _path: _FakeSession([[1.0, 0.0]]),
    )

    assert reviewer.metadata == {
        "model_id": "unit-test-reviewer",
        "prototype_version": "unit-test-v1",
        "source": "https://example.test/model",
        "license": "test-only",
        "character_count": 4,
        "embedding_size": 2,
    }


def test_onnx_reviewer_runs_one_batch_for_multiple_crops(
    tmp_path,
):
    _write_reviewer_assets(tmp_path)
    fake_session = _FakeBatchSession(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [-1.0, 0.0],
        ]
    )
    reviewer = OnnxGlyphReviewer(
        str(tmp_path),
        session_factory=lambda _path: fake_session,
        prototype_block_size=2,
    )

    results = reviewer.review_batch(
        [
            Image.new("RGB", (7, 7), "white"),
            Image.new("RGB", (9, 6), "black"),
            Image.new("RGB", (4, 10), "gray"),
        ],
        top_k=1,
    )

    assert fake_session.run_calls == 1
    _, feeds = fake_session.inputs[0]
    assert feeds["image"].shape == (3, 3, 8, 8)
    assert [
        candidates[0]["char"] for candidates in results
    ] == ["甲", "丁", "丙"]


def test_onnx_reviewer_does_not_load_session_for_empty_or_disabled_batch(
    tmp_path,
):
    _write_reviewer_assets(tmp_path)
    factory_calls = []

    def session_factory(_path):
        factory_calls.append(True)
        return _FakeBatchSession([[1.0, 0.0]])

    reviewer = OnnxGlyphReviewer(
        str(tmp_path),
        session_factory=session_factory,
    )

    assert reviewer.review_batch([], top_k=3) == []
    assert reviewer.review_batch(
        [Image.new("RGB", (8, 8), "white")],
        top_k=0,
    ) == [[]]
    assert factory_calls == []


@pytest.mark.parametrize(
    "output, message",
    [
        (
            [[1.0, 0.0], [0.0, 1.0]],
            "batch",
        ),
        (
            [[1.0, 0.0, 0.0]] * 3,
            "维度",
        ),
        (
            [[1.0, 0.0], [0.0, 0.0], [0.0, 1.0]],
            "范数为 0",
        ),
        (
            [[1.0, 0.0], [np.nan, 1.0], [0.0, 1.0]],
            "非有限值",
        ),
    ],
)
def test_onnx_reviewer_rejects_invalid_batch_embeddings(
    tmp_path,
    output,
    message,
):
    _write_reviewer_assets(tmp_path)
    reviewer = OnnxGlyphReviewer(
        str(tmp_path),
        session_factory=lambda _path: _FakeBatchSession(output),
    )
    crops = [Image.new("RGB", (8, 8), "white") for _ in range(3)]

    with pytest.raises(ValueError, match=message):
        reviewer.review_batch(crops, top_k=1)


def test_onnx_reviewer_rejects_missing_or_modified_model_asset(tmp_path):
    _write_reviewer_assets(tmp_path)
    os.remove(tmp_path / "glyph_encoder.onnx")
    with pytest.raises(FileNotFoundError, match="glyph_encoder.onnx"):
        OnnxGlyphReviewer(str(tmp_path))

    _write_reviewer_assets(tmp_path)
    with open(tmp_path / "glyph_encoder.onnx", "ab") as model_file:
        model_file.write(b"modified")
    with pytest.raises(ValueError, match="SHA-256"):
        OnnxGlyphReviewer(str(tmp_path))


def test_onnx_reviewer_rejects_character_prototype_dimension_mismatch(
    tmp_path,
):
    _write_reviewer_assets(
        tmp_path,
        prototypes=np.ones((3, 2), dtype=np.float32),
    )

    with pytest.raises(ValueError, match="字符数"):
        OnnxGlyphReviewer(str(tmp_path))


def test_onnx_reviewer_rejects_modified_character_or_prototype_assets(
    tmp_path,
):
    _write_reviewer_assets(tmp_path)
    with open(tmp_path / "characters.txt", "a", encoding="utf-8") as file:
        file.write("戊\n")
    with pytest.raises(ValueError, match="characters SHA-256"):
        OnnxGlyphReviewer(str(tmp_path))

    _write_reviewer_assets(tmp_path)
    with open(tmp_path / "prototypes.npy", "ab") as file:
        file.write(b"modified")
    with pytest.raises(ValueError, match="prototypes SHA-256"):
        OnnxGlyphReviewer(str(tmp_path))


@pytest.mark.parametrize(
    "prototypes, message",
    [
        (np.zeros((4, 2), dtype=np.float32), "范数为 0"),
        (
            np.asarray(
                [[1, 0], [0, 1], [np.nan, 0], [-1, 0]],
                dtype=np.float32,
            ),
            "非有限值",
        ),
    ],
)
def test_onnx_reviewer_rejects_invalid_prototype_rows(
    tmp_path,
    prototypes,
    message,
):
    _write_reviewer_assets(tmp_path, prototypes=prototypes)

    with pytest.raises(ValueError, match=message):
        OnnxGlyphReviewer(str(tmp_path))


def test_onnx_reviewer_requires_source_and_license_metadata(tmp_path):
    _write_reviewer_assets(tmp_path, manifest_updates={"license": ""})

    with pytest.raises(ValueError, match="license"):
        OnnxGlyphReviewer(str(tmp_path))


def test_session_load_failure_is_cached_and_reported_by_lazy_reviewer(
    tmp_path,
):
    _write_reviewer_assets(tmp_path)
    session_calls = []

    def reviewer_factory(asset_dir):
        def failing_session(model_path):
            session_calls.append(model_path)
            raise RuntimeError("unsupported ONNX")

        return OnnxGlyphReviewer(
            asset_dir,
            session_factory=failing_session,
        )

    reviewer = LazyOnnxGlyphReviewer(
        str(tmp_path),
        reviewer_factory=reviewer_factory,
    )
    for _attempt in range(2):
        with pytest.raises(RuntimeError, match="unsupported ONNX"):
            reviewer.review(Image.new("RGB", (8, 8)), top_k=3)

    assert len(session_calls) == 1
    assert "unsupported ONNX" in reviewer.load_error


def test_lazy_reviewer_does_not_touch_assets_until_first_review(tmp_path):
    calls = []

    class FakeReviewer:
        metadata = {"model_id": "lazy-test"}

        def review(self, crop_rgb, top_k=5):
            return [{"char": "字", "score": 0.9, "rank": 1}]

    def reviewer_factory(asset_dir):
        calls.append(asset_dir)
        return FakeReviewer()

    reviewer = LazyOnnxGlyphReviewer(
        str(tmp_path),
        reviewer_factory=reviewer_factory,
    )
    assert calls == []
    assert reviewer.metadata == {}

    result = reviewer.review(Image.new("RGB", (8, 8)), top_k=3)

    assert calls == [str(tmp_path)]
    assert result[0]["char"] == "字"
    assert reviewer.metadata == {"model_id": "lazy-test"}
    assert reviewer.load_error == ""
