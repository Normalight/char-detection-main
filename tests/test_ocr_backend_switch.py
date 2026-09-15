import json
import os
from pathlib import Path
from unittest.mock import patch

import config
import cutbatch
import numpy as np
from PIL import Image


def test_legacy_detector_dispatch():
    with patch.object(cutbatch, "det_model", "legacy"):
        with patch("cutbatch.DBNET", return_value="LEGACY") as constructor:
            handle, name = cutbatch._build_det_handle()

    constructor.assert_called_once_with(cutbatch.model_path)
    assert handle == "LEGACY"
    assert name == "legacy-dbnet"


def test_v6_small_detector_dispatch():
    model_path = os.path.join("/tmp", "v6-small", "inference.onnx")

    class FakeDetector:
        def __init__(self, path, **kwargs):
            self.path = path
            self.kwargs = kwargs

    with patch.object(cutbatch, "det_model", "v6-small"):
        with patch.object(
            cutbatch,
            "ppocrv6_det_model_path",
            {"small": model_path},
            create=True,
        ):
            with patch("os.path.exists", return_value=True):
                with patch.dict(
                    cutbatch.__dict__,
                    {"PPOcrV6Detector": FakeDetector},
                ):
                    handle, name = cutbatch._build_det_handle()

    assert name == "v6-small"
    assert isinstance(handle, FakeDetector)
    assert handle.path == model_path
    assert handle.kwargs["limit_type"] == "short"
    assert handle.kwargs["max_side_limit"] == 4000
    assert handle.kwargs["box_thresh"] is None


def test_v6_detector_dispatch_forwards_box_thresh_override():
    model_path = os.path.join("/tmp", "v6-small", "inference.onnx")

    class FakeDetector:
        def __init__(self, path, **kwargs):
            self.path = path
            self.kwargs = kwargs

    with patch.object(cutbatch, "det_model", "v6-small"):
        with patch.object(
            cutbatch,
            "ppocrv6_det_model_path",
            {"small": model_path},
            create=True,
        ):
            with patch("os.path.exists", return_value=True):
                with patch.dict(
                    cutbatch.__dict__,
                    {"PPOcrV6Detector": FakeDetector},
                ):
                    with patch.object(
                        cutbatch, "ppocrv6_det_box_thresh", 0.45
                    ):
                        handle, name = cutbatch._build_det_handle()

    assert name == "v6-small"
    assert handle.kwargs["box_thresh"] == 0.45


def test_unknown_detector_raises_clear_error():
    with patch.object(cutbatch, "det_model", "v7-imaginary"):
        try:
            cutbatch._build_det_handle()
        except RuntimeError as exc:
            message = str(exc)
        else:
            raise AssertionError("unknown detector did not raise")

    assert "未知 det_model" in message
    assert "legacy" in message
    assert "v6-small" in message


def test_missing_v6_detector_file_raises_download_hint():
    missing = "/nonexistent/ppocrv6/small/inference.onnx"
    with patch.object(cutbatch, "det_model", "v6-small"):
        with patch.object(
            cutbatch,
            "ppocrv6_det_model_path",
            {"small": missing},
            create=True,
        ):
            with patch("os.path.exists", return_value=False):
                try:
                    cutbatch._build_det_handle()
                except RuntimeError as exc:
                    message = str(exc)
                else:
                    raise AssertionError("missing detector model did not raise")

    assert "v6 det 模型文件不存在" in message
    assert "bench/README.md" in message


def test_config_exposes_full_v6_candidate():
    assert config.det_model == "v6-small"
    assert config.rec_model == "v6-small"
    assert config.char_box_strategy == "stroke-ownership"


def test_config_detector_resize_policy_restores_legacy_semantics():
    # max 策略永不上采样，小图会以原始低分辨率进检测导致漏检；
    # 默认必须是 short（短边目标，可上采样），与 legacy DBNet 一致。
    assert config.ppocrv6_det_limit_type == "short"
    # None 表示沿用官方 inference.yml 的 box_thresh。
    assert config.ppocrv6_det_box_thresh is None


def test_ocr_handle_does_not_expose_grouping_cache_id():
    class FakeRecognizer:
        pass

    recognizer = FakeRecognizer()
    with (
        patch("cutbatch._build_det_handle", return_value=(object(), "det")),
        patch(
            "cutbatch._build_rec_handle",
            return_value=(recognizer, "v6-small"),
        ),
    ):
        handle = cutbatch.OcrHandle()

    assert not hasattr(handle, "rec_cache_id")
    assert not hasattr(recognizer, "rec_cache_id")


def test_model_defaults_match_historical_benchmark_winner():
    winner = json.loads(
        Path("bench/recommended_backend.json").read_text(encoding="utf-8")
    )
    assert config.det_model == winner["det_model"]
    assert config.rec_model == winner["rec_model"]
    # Historical benchmark evaluated shared cuts, not independent glyph extents.
    assert winner["char_box_strategy"] == "polarity-aware"


def test_pyinstaller_collects_production_ppocrv6_without_bench_package():
    source = Path("app.spec").read_text(encoding="utf-8")

    assert "collect_submodules('ppocrv6')" in source
    assert "collect_submodules('bench')" not in source
    assert "ppocrv6_det_model_path" in source
    assert "ppocrv6_rec_model_path" in source
    assert "rare_glyph_review_dir" not in source


def test_pyinstaller_builds_onedir_distribution():
    source = Path("app.spec").read_text(encoding="utf-8")

    assert "exclude_binaries=True" in source
    assert "coll = COLLECT(" in source
    assert "a.binaries" in source
    assert "a.datas" in source
    assert "_is_incompatible_root_icu" in source
    assert "filename == 'icuuc.dll'" in source
    assert "filename.startswith('icudt')" in source


def test_windows_build_script_checks_onedir_executable():
    source = Path("build_windows.bat").read_text(encoding="utf-8")

    assert "dist\\interactive_annotator\\interactive_annotator.exe" in source
    assert 'if not exist "dist\\interactive_annotator.exe"' not in source
    assert 'del /q "dist\\interactive_annotator.exe"' in source
    assert 'if exist "dist\\interactive_annotator.exe"' in source
    assert "still in use" in source


def _capture_detector_input(detector_name):
    captured = {}

    class FakeDetector:
        def process(self, image, limit_side_len):
            captured["image"] = np.asarray(image).copy()
            captured["limit"] = limit_side_len
            return [], []

    handle = cutbatch.OcrHandle.__new__(cutbatch.OcrHandle)
    handle.det_name = detector_name
    handle.text_handle = FakeDetector()
    handle.crnnRecWithBox = lambda _image, _boxes, _scores: []
    image = Image.fromarray(
        np.asarray([[[11, 22, 33]]], dtype=np.uint8), mode="RGB"
    )
    handle.text_predict(image, 960)
    return captured


def test_legacy_detector_receives_explicit_bgr_input():
    captured = _capture_detector_input("legacy-dbnet")
    assert captured["image"].tolist() == [[[33, 22, 11]]]
    assert captured["limit"] == 960


def test_v6_detector_receives_rgb_input():
    captured = _capture_detector_input("v6-small")
    assert captured["image"].tolist() == [[[11, 22, 33]]]
