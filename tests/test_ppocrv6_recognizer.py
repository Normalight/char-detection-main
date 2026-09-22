import builtins
import importlib

import numpy as np
from PIL import Image
import pytest


def test_recognizer_is_importable_from_production_package():
    module = importlib.import_module("ocr.recognizer")
    assert hasattr(module, "PPOcrV6RecHandle")


def test_recognizer_uses_shared_midpoint_boundaries(monkeypatch):
    module = importlib.import_module("ocr.recognizer")
    handle = module.PPOcrV6RecHandle.__new__(module.PPOcrV6RecHandle)
    handle.character = ["<blank>", "甲", "乙", "丙"]
    handle.decode_limit = len(handle.character)
    handle.char_box_strategy = "ctc-midpoint"

    predictions = np.zeros((1, 8, len(handle.character)), dtype=np.float32)
    predictions[:, :, 0] = 1.0
    for timestep, token in ((1, 1), (3, 2), (6, 3)):
        predictions[0, timestep, 0] = 0.0
        predictions[0, timestep, token] = 2.0

    monkeypatch.setattr(
        module,
        "reverse_rotate_crop_image",
        lambda _image, _crop, _bbox, points, _origin: np.asarray(
            points, dtype=np.float32
        ),
    )
    result = handle._postprocess(
        predictions,
        scale=1.0,
        resized_w=64,
        crop_im=Image.new("RGB", (64, 48), "white"),
        bbox=np.asarray([[0, 0], [64, 0], [64, 48], [0, 48]], np.float32),
    )

    intervals = [
        (
            float(np.min(np.asarray(item["bbox"])[:, 0])),
            float(np.max(np.asarray(item["bbox"])[:, 0])),
        )
        for item in result["char_list"]
    ]
    # Official PaddleOCR word-box coordinates use (CTC column + 0.5)
    # multiplied by crop_width / time_steps.  Internal cuts are midpoints.
    assert intervals == [(0.0, 20.0), (20.0, 40.0), (40.0, 64.0)]


def test_recognizer_exposes_character_and_line_confidence(monkeypatch):
    module = importlib.import_module("ocr.recognizer")
    handle = module.PPOcrV6RecHandle.__new__(module.PPOcrV6RecHandle)
    handle.character = ["<blank>", "甲", "乙"]
    handle.decode_limit = len(handle.character)
    handle.char_box_strategy = "ctc-midpoint"

    predictions = np.zeros((1, 5, len(handle.character)), dtype=np.float32)
    predictions[:, :, 0] = 1.0
    predictions[0, 1] = [0.05, 0.80, 0.15]
    predictions[0, 3] = [0.05, 0.25, 0.70]
    monkeypatch.setattr(
        module,
        "reverse_rotate_crop_image",
        lambda _image, _crop, _bbox, points, _origin: np.asarray(
            points, dtype=np.float32
        ),
    )

    result = handle._postprocess(
        predictions,
        scale=1.0,
        resized_w=50,
        crop_im=Image.new("RGB", (50, 24), "white"),
        bbox=np.asarray([[0, 0], [50, 0], [50, 24], [0, 24]], np.float32),
    )

    assert [item["score"] for item in result["char_list"]] == pytest.approx(
        [0.80, 0.70]
    )
    assert result["score"] == pytest.approx(0.75)


def test_projection_strategy_refines_the_actual_crop(monkeypatch):
    module = importlib.import_module("ocr.recognizer")
    handle = module.PPOcrV6RecHandle.__new__(module.PPOcrV6RecHandle)
    handle.character = ["<blank>", "甲", "乙"]
    handle.decode_limit = len(handle.character)
    handle.char_box_strategy = "projection-refined"

    predictions = np.zeros((1, 6, len(handle.character)), dtype=np.float32)
    predictions[:, :, 0] = 1.0
    predictions[0, 1] = [0.0, 2.0, 0.0]
    predictions[0, 4] = [0.0, 0.0, 2.0]
    crop = Image.new("RGB", (48, 24), (231, 232, 233))
    calls = []

    def fake_refiner(image, centers, boundaries):
        calls.append((np.asarray(image), centers, boundaries))
        return boundaries

    monkeypatch.setattr(module, "refine_boundaries_by_projection", fake_refiner)
    monkeypatch.setattr(
        module,
        "reverse_rotate_crop_image",
        lambda _image, _crop, _bbox, points, _origin: np.asarray(
            points, dtype=np.float32
        ),
    )

    handle._postprocess(
        predictions,
        scale=1.0,
        resized_w=48,
        crop_im=crop,
        bbox=np.asarray([[0, 0], [48, 0], [48, 24], [0, 24]], np.float32),
    )

    assert len(calls) == 1
    assert calls[0][0].shape == (24, 48, 3)
    assert np.all(calls[0][0] == (231, 232, 233))


def test_unsupported_character_box_strategy_is_rejected():
    module = importlib.import_module("ocr.recognizer")

    with pytest.raises(ValueError, match="char_box_strategy"):
        module.PPOcrV6RecHandle(
            "unused.onnx", char_box_strategy="guess-the-glyph"
        )


def test_out_of_dict_token_becomes_space_with_box(monkeypatch):
    # v6 字典无空格：超字典 token 应还原为空格并生成字框（对齐 legacy
    # 词间空格行为），而不是输出 '?' 噪声字符。
    module = importlib.import_module("ocr.recognizer")
    handle = module.PPOcrV6RecHandle.__new__(module.PPOcrV6RecHandle)
    handle.character = ["<blank>", "A", "B"]
    # vocab 维度 5，但字典只有 3：idx 3/4 是超字典 filler。
    handle.decode_limit = len(handle.character)
    handle.char_box_strategy = "ctc-midpoint"

    predictions = np.zeros((1, 6, 5), dtype=np.float32)
    predictions[:, :, 0] = 1.0
    predictions[0, 1, 1] = 2.0          # A
    predictions[0, 3, 3] = 2.0          # filler (unused slot)
    predictions[0, 5, 2] = 2.0          # B
    monkeypatch.setattr(
        module,
        "reverse_rotate_crop_image",
        lambda _image, _crop, _bbox, points, _origin: np.asarray(
            points, dtype=np.float32
        ),
    )

    result = handle._postprocess(
        predictions,
        scale=1.0,
        resized_w=60,
        crop_im=Image.new("RGB", (60, 24), "white"),
        bbox=np.asarray([[0, 0], [60, 0], [60, 24], [0, 24]], np.float32),
    )

    assert result["line_pred"] == "A B"
    assert [item["char"] for item in result["char_list"]] == ["A", " ", "B"]
    assert [item.get("filler") for item in result["char_list"]] == [
        False,
        True,
        False,
    ]
    assert all(
        item.get("bbox") is not None for item in result["char_list"]
    )


def test_consecutive_filler_tokens_collapse_into_one_space():
    module = importlib.import_module("ocr.recognizer")
    handle = module.PPOcrV6RecHandle.__new__(module.PPOcrV6RecHandle)
    handle.character = ["<blank>", "A", "B"]
    handle.decode_limit = len(handle.character)
    handle.char_box_strategy = "ctc-midpoint"

    predictions = np.zeros((1, 6, 5), dtype=np.float32)
    predictions[:, :, 0] = 1.0
    predictions[0, 1, 3] = 2.0          # filler slot 3
    predictions[0, 3, 4] = 2.0          # filler slot 4（不同 idx，仍应折叠）
    predictions[0, 5, 1] = 2.0          # A

    line_pred, char_list = handle._decode_ctc(predictions)

    assert line_pred == " A"
    assert [item["char"] for item in char_list] == [" ", "A"]
    assert [item["filler"] for item in char_list] == [True, False]


def test_inference_config_is_read_as_utf8_on_windows(monkeypatch, tmp_path):
    module = importlib.import_module("ocr.recognizer")
    config_path = tmp_path / "inference.yml"
    config_path.write_text(
        "PostProcess:\n  character_dict: ['甲', '乙']\n",
        encoding="utf-8",
    )

    class FakeValue:
        name = "input"
        shape = [1, 1, 3]

    class FakeSession:
        def get_inputs(self):
            return [FakeValue()]

        def get_outputs(self):
            return [FakeValue()]

    real_open = builtins.open

    def encoding_checked_open(path, *args, **kwargs):
        assert kwargs.get("encoding") == "utf-8"
        return real_open(path, *args, **kwargs)

    monkeypatch.setattr(module, "_build_session", lambda _path: FakeSession())
    monkeypatch.setattr(module, "open", encoding_checked_open, raising=False)

    handle = module.PPOcrV6RecHandle(
        "unused.onnx", yml_path=str(config_path)
    )

    assert handle.character == ["<blank>", "甲", "乙"]


def test_projection_strategy_accepts_empty_ctc_output(monkeypatch):
    module = importlib.import_module("ocr.recognizer")
    handle = module.PPOcrV6RecHandle.__new__(module.PPOcrV6RecHandle)
    handle.character = ["<blank>", "甲"]
    handle.decode_limit = len(handle.character)
    handle.char_box_strategy = "projection-refined"
    predictions = np.zeros((1, 4, 2), dtype=np.float32)
    predictions[:, :, 0] = 1.0
    monkeypatch.setattr(
        module,
        "reverse_rotate_crop_image",
        lambda _image, _crop, _bbox, points, _origin: np.asarray(
            points, dtype=np.float32
        ),
    )

    result = handle._postprocess(
        predictions,
        scale=1.0,
        resized_w=32,
        crop_im=Image.new("RGB", (32, 24), "white"),
        bbox=np.asarray([[0, 0], [32, 0], [32, 24], [0, 24]], np.float32),
    )

    assert result["line_pred"] == ""
    assert result["char_list"] == []


@pytest.mark.parametrize("background", ["white", (24, 25, 26)])
def test_polarity_strategy_accepts_empty_ctc_output(
    monkeypatch, background
):
    module = importlib.import_module("ocr.recognizer")
    handle = module.PPOcrV6RecHandle.__new__(module.PPOcrV6RecHandle)
    handle.character = ["<blank>", "甲"]
    handle.decode_limit = len(handle.character)
    handle.char_box_strategy = "polarity-aware"
    predictions = np.zeros((1, 4, 2), dtype=np.float32)
    predictions[:, :, 0] = 1.0
    monkeypatch.setattr(
        module,
        "reverse_rotate_crop_image",
        lambda _image, _crop, _bbox, points, _origin: np.asarray(
            points, dtype=np.float32
        ),
    )

    result = handle._postprocess(
        predictions,
        scale=1.0,
        resized_w=32,
        crop_im=Image.new("RGB", (32, 24), background),
        bbox=np.asarray([[0, 0], [32, 0], [32, 24], [0, 24]], np.float32),
    )

    assert result["line_pred"] == ""
    assert result["char_list"] == []


def test_polarity_strategy_refines_actual_crop_without_changing_text(
    monkeypatch,
):
    module = importlib.import_module("ocr.recognizer")
    handle = module.PPOcrV6RecHandle.__new__(module.PPOcrV6RecHandle)
    handle.character = ["<blank>", "甲", "乙"]
    handle.decode_limit = len(handle.character)
    handle.char_box_strategy = "polarity-aware"

    predictions = np.zeros((1, 6, len(handle.character)), dtype=np.float32)
    predictions[:, :, 0] = 1.0
    predictions[0, 1] = [0.0, 2.0, 0.0]
    predictions[0, 4] = [0.0, 0.0, 2.0]
    crop = Image.new("RGB", (48, 24), (24, 25, 26))
    calls = []

    def fake_refiner(image, centers, boundaries):
        calls.append((np.asarray(image), centers, list(boundaries)))
        refined = list(boundaries)
        refined[1] = 23.0
        return refined

    monkeypatch.setattr(
        module, "refine_boundaries_by_polarity", fake_refiner
    )
    monkeypatch.setattr(
        module,
        "reverse_rotate_crop_image",
        lambda _image, _crop, _bbox, points, _origin: np.asarray(
            points, dtype=np.float32
        ),
    )

    result = handle._postprocess(
        predictions,
        scale=1.0,
        resized_w=48,
        crop_im=crop,
        bbox=np.asarray([[0, 0], [48, 0], [48, 24], [0, 24]], np.float32),
    )

    assert result["line_pred"] == "甲乙"
    assert len(calls) == 1
    assert np.all(calls[0][0] == (24, 25, 26))
    intervals = [
        (
            float(np.min(np.asarray(item["bbox"])[:, 0])),
            float(np.max(np.asarray(item["bbox"])[:, 0])),
        )
        for item in result["char_list"]
    ]
    assert intervals == [(0.0, 23.0), (23.0, 48.0)]


def test_polarity_strategy_preserves_outer_crop_edges(monkeypatch):
    module = importlib.import_module("ocr.recognizer")
    handle = module.PPOcrV6RecHandle.__new__(module.PPOcrV6RecHandle)
    handle.character = ["<blank>", "甲", "乙"]
    handle.decode_limit = len(handle.character)
    handle.char_box_strategy = "polarity-aware"

    predictions = np.zeros((1, 8, len(handle.character)), dtype=np.float32)
    predictions[:, :, 0] = 1.0
    predictions[0, 2] = [0.0, 2.0, 0.0]
    predictions[0, 5] = [0.0, 0.0, 2.0]
    monkeypatch.setattr(
        module,
        "refine_boundaries_by_polarity",
        lambda _image, _centers, boundaries: boundaries,
    )
    monkeypatch.setattr(
        module,
        "reverse_rotate_crop_image",
        lambda _image, _crop, _bbox, points, _origin: np.asarray(
            points, dtype=np.float32
        ),
    )

    result = handle._postprocess(
        predictions,
        scale=1.0,
        resized_w=80,
        crop_im=Image.new("RGB", (80, 32), "white"),
        bbox=np.asarray([[0, 0], [80, 0], [80, 32], [0, 32]], np.float32),
    )

    intervals = [
        (
            float(np.min(np.asarray(item["bbox"])[:, 0])),
            float(np.max(np.asarray(item["bbox"])[:, 0])),
        )
        for item in result["char_list"]
    ]
    assert intervals == [(0.0, 40.0), (40.0, 80.0)]
