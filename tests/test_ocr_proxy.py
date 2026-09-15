from unittest.mock import Mock
import numpy as np
from PIL import Image
import config
from cutbatch import OcrHandle
from annotator_core import AnnotatorCore
from worker import _load_item_pil
from storage_utils import save_item_boxes_to_disk


def test_proxy_maps_all_geometry_with_actual_rounded_axes(monkeypatch):
    monkeypatch.setattr(config, "ocr_max_side", 256)
    image = Image.new("RGB", (1001, 777), "white")
    original = image.tobytes()
    handle = OcrHandle.__new__(OcrHandle)
    handle.det_name = "v6-small"
    handle.text_handle = Mock()
    handle.text_handle.process.return_value = ([], [])
    quad = np.array([[10, 20], [30, 20], [30, 40], [10, 40]])
    line = np.array([[20, 20], [20, 40]])
    handle.crnnRecWithBox = Mock(return_value=[{"bbox": quad.copy(), "result": {
        "char_list": [{"bbox": quad.copy(), "line": line.copy()}],
        "words": [{"bbox": quad.copy()}]}}])
    rows = handle.text_predict(image, 960)
    assert handle.text_handle.process.call_args.args[0].shape == (199, 256, 3)
    factors = [1001/256, 777/199]
    row = rows[0]
    np.testing.assert_allclose(row["bbox"], quad*factors)
    np.testing.assert_allclose(row["result"]["char_list"][0]["bbox"], quad*factors)
    np.testing.assert_allclose(row["result"]["char_list"][0]["line"], line*factors)
    np.testing.assert_allclose(row["result"]["words"][0]["bbox"], quad*factors)
    assert image.size == (1001, 777) and image.tobytes() == original


def test_original_pixels_survive_load_and_export_with_small_ocr_scale(tmp_path):
    pixels = np.random.default_rng(3).integers(0, 256, (200, 300, 3), dtype=np.uint8)
    path = tmp_path / "original.png"
    Image.fromarray(pixels).save(path)
    item = {"type": "image", "source_path": str(path), "display_name": "original.png",
            "boxes": [np.array([[30, 40], [130, 40], [130, 120], [30, 120]], np.float32)],
            "visited": True}
    core = AnnotatorCore()
    core.image_scale = .1
    for loaded in (core._load_base_pil_for_item(item), _load_item_pil(item, 200, None, .1)):
        np.testing.assert_array_equal(np.asarray(loaded), pixels)
    out = tmp_path / "export"
    assert save_item_boxes_to_disk(str(out), item, loaded) == 1
    crops = list(out.rglob("*.png"))
    assert len(crops) == 1
    assert Image.open(crops[0]).size == (100, 80)


def test_cancel_after_detection_skips_recognition():
    handle = OcrHandle.__new__(OcrHandle)
    handle.det_name = "v6-small"
    handle.text_handle = Mock()
    handle.text_handle.process.return_value = ([], [])
    handle.crnnRecWithBox = Mock()
    cancelled = Mock(side_effect=[False, True])
    assert handle.text_predict(Image.new("RGB", (10, 10)), 960, cancel_check=cancelled) == []
    handle.crnnRecWithBox.assert_not_called()
