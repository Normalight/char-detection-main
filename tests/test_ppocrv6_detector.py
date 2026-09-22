from unittest.mock import patch

import pytest

from ocr.detector import (
    PPOcrV6Detector,
    calculate_resized_shape,
    load_detector_config,
)


def test_max_limit_does_not_upscale_small_image():
    assert calculate_resized_shape(432, 827, 960, "max", 4000) == (448, 832)


def test_max_limit_downscales_large_image():
    assert calculate_resized_shape(5857, 7665, 960, "max", 4000) == (736, 960)


def test_min_limit_upscales_short_side_but_honors_max_side():
    height, width = calculate_resized_shape(100, 1000, 960, "min", 4000)
    assert max(height, width) <= 4000
    assert height % 32 == 0
    assert width % 32 == 0


def test_short_limit_upscales_small_image_like_legacy():
    # 短边 383 拉到 960：与 legacy DBNet 同语义，小图恢复检测分辨率。
    assert calculate_resized_shape(572, 383, 960, "short", 4000) == (1440, 960)


def test_short_limit_downscales_large_image():
    height, width = calculate_resized_shape(5857, 7665, 960, "short", 4000)
    assert min(height, width) == 960
    assert height % 32 == 0
    assert width % 32 == 0


def test_short_limit_clamps_extreme_panorama_by_max_side():
    height, width = calculate_resized_shape(300, 20000, 960, "short", 4000)
    assert max(height, width) <= 4000
    assert height % 32 == 0
    assert width % 32 == 0


def test_invalid_limit_type_is_rejected():
    with pytest.raises(ValueError, match="limit_type"):
        calculate_resized_shape(100, 200, 960, "sideways", 4000)


@pytest.fixture
def detector_yml(tmp_path):
    path = tmp_path / "inference.yml"
    path.write_text(
        """
PostProcess:
  box_thresh: 0.45
  max_candidates: 3000
  thresh: 0.2
  unclip_ratio: 1.4
PreProcess:
  transform_ops:
    - DecodeImage:
        img_mode: BGR
    - NormalizeImage:
        mean: [0.485, 0.456, 0.406]
        scale: 1./255.
        std: [0.229, 0.224, 0.225]
""",
        encoding="utf-8",
    )
    return path


def test_load_detector_config_uses_inference_yml(detector_yml):
    config = load_detector_config(detector_yml)

    assert config.thresh == 0.2
    assert config.box_thresh == 0.45
    assert config.unclip_ratio == 1.4
    assert config.max_candidates == 3000
    assert config.mean == (0.485, 0.456, 0.406)
    assert config.std == (0.229, 0.224, 0.225)


def test_constructor_overrides_win_over_inference_yml(detector_yml):
    class FakeInput:
        name = "x"

    class FakeSession:
        def get_inputs(self):
            return [FakeInput()]

    with patch("ocr.detector._build_session", return_value=FakeSession()):
        detector = PPOcrV6Detector(
            "unused.onnx",
            yml_path=detector_yml,
            thresh=0.3,
            box_thresh=0.6,
            unclip_ratio=1.5,
            max_candidates=1000,
        )

    assert detector.config.thresh == 0.3
    assert detector.config.box_thresh == 0.6
    assert detector.config.unclip_ratio == 1.5
    assert detector.config.max_candidates == 1000
