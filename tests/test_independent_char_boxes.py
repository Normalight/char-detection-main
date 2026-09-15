import numpy as np
import pytest
from PIL import Image

from ppocrv6.char_boxes import independent_stroke_boxes


def glyph_pair(background=240, foreground=20):
    image = np.full((48, 80, 3), background, dtype=np.uint8)
    # Separate connected glyphs, with interleaved overhangs in x.
    image[5:33, 12:24] = foreground
    image[5:10, 22:45] = foreground
    image[17:43, 52:64] = foreground
    image[38:43, 35:54] = foreground
    return image


@pytest.mark.parametrize("background,foreground", [(240, 20), (20, 240)])
def test_preserves_interleaved_strokes_and_independent_vertical_extents(background, foreground):
    boxes = independent_stroke_boxes(glyph_pair(background, foreground), [20, 58])
    a, b = boxes
    assert a[0] <= 12 and a[2] >= 45
    assert b[0] <= 35 and b[2] >= 64
    assert a[2] > b[0]  # real overlap, not just touching
    assert 0 < a[1] <= 5 and 33 <= a[3] < 38
    assert 10 < b[1] <= 17 and 43 <= b[3] < 48


def test_disconnected_radicals_are_combined_without_neighbour():
    image = np.full((40, 80), 255, np.uint8)
    image[5:30, 10:17] = 0
    image[8:33, 23:30] = 0
    image[12:28, 55:65] = 0
    a, b = independent_stroke_boxes(image, [20, 60])
    assert a[0] <= 10 and a[2] >= 30 and a[2] < b[0]
    assert b[0] <= 55 and b[2] >= 65


def test_touching_glyphs_have_bounded_overlapping_boxes():
    image = np.full((40, 100), 255, np.uint8)
    image[7:33, 15:30] = 0
    image[7:33, 65:80] = 0
    image[20:23, 25:70] = 0
    a, b = independent_stroke_boxes(image, [25, 70])
    assert a[2] > b[0]
    assert a[2] < 65 and b[0] > 30


def test_single_character_trims_large_margins_and_keeps_edge_strokes():
    image = np.full((40, 100), 255, np.uint8)
    image[0:30, 0:22] = 0
    box, = independent_stroke_boxes(image, [12])
    assert box[0] == box[1] == 0
    assert 22 <= box[2] < 30 and 30 <= box[3] < 40


def test_blank_image_preserves_count_without_partitioning_whole_row():
    boxes = independent_stroke_boxes(np.full((40, 200), 128, np.uint8), [30, 65])
    assert len(boxes) == 2
    assert boxes[0][2] > boxes[1][0]
    assert boxes[-1][2] < 100
    assert independent_stroke_boxes(np.zeros((20, 30), np.uint8), []) == []


@pytest.mark.parametrize("centers", [[20, 10], [10, 10], [float('nan')], [-1], [80]])
def test_rejects_invalid_anchors(centers):
    with pytest.raises(ValueError):
        independent_stroke_boxes(glyph_pair(), centers)


@pytest.mark.parametrize("strategy", ["independent-strokes", "stroke-ownership"])
@pytest.mark.parametrize("vertical", [False, True])
def test_recognizer_maps_stroke_boxes_to_original_coordinates(vertical, strategy):
    from ppocrv6.recognizer import PPOcrV6RecHandle
    from storage_utils import extract_boxes_from_ocr_results
    handle = PPOcrV6RecHandle.__new__(PPOcrV6RecHandle)
    handle.char_box_strategy = strategy
    handle._decode_ctc = lambda _preds: ("甲乙", [
        {"char": "甲", "idx": 19, "score": .9},
        {"char": "乙", "idx": 57, "score": .9},
    ])
    bbox = np.array([[100, 200], [148, 200], [148, 280], [100, 280]]
                    if vertical else [[100, 200], [180, 200], [180, 248], [100, 248]],
                    dtype=np.float32)
    original = bbox.copy()
    result = handle._postprocess(np.zeros((1, 80, 3)), 1, 80,
                                 Image.fromarray(glyph_pair()), bbox)
    assert result["line_pred"] == "甲乙"
    assert result["char_box_strategy"] == strategy
    boxes = extract_boxes_from_ocr_results([{"bbox": bbox, "result": result}])
    assert len(boxes) == 2
    for box in boxes:
        assert box.shape == (4, 2) and np.isfinite(box).all()
        assert (box.min(axis=0) >= bbox.min(axis=0) - 6).all()
        assert (box.max(axis=0) <= bbox.max(axis=0) + 6).all()
    lo = np.maximum(boxes[0].min(axis=0), boxes[1].min(axis=0))
    hi = np.minimum(boxes[0].max(axis=0), boxes[1].max(axis=0))
    assert (hi > lo).all()
    np.testing.assert_array_equal(bbox, original)


def test_independent_boxes_are_not_replaced_by_geometry_fallback():
    from storage_utils import extract_boxes_from_ocr_results
    box = np.array([[10, 5], [22, 5], [22, 35], [10, 35]], np.float32)
    result = {"line_pred": "甲", "score": .1, "char_list": [{"bbox": box}],
              "char_box_strategy": "independent-strokes"}
    boxes = extract_boxes_from_ocr_results([{
        "bbox": [[0, 0], [200, 0], [200, 40], [0, 40]], "result": result}])
    assert len(boxes) == 1
    np.testing.assert_array_equal(boxes[0], box)
    assert "char_box_fallback" not in result


def test_faint_connected_serif_is_retained_with_whitespace():
    image = np.full((50, 70), 255, np.uint8)
    image[10:40, 25:40] = 0
    image[12:14, 12:27] = 210
    box, = independent_stroke_boxes(image, [32])
    assert box[0] <= 7 and box[2] >= 45
    assert box[1] <= 5 and box[3] >= 45


def test_cjk_does_not_shrink_to_one_strong_radical():
    image = np.full((50, 80), 255, np.uint8)
    image[16:35, 35:43] = 0
    box, = independent_stroke_boxes(image, [40], ["中"])
    assert box[2] - box[0] >= 50
    assert box[3] - box[1] >= 42


@pytest.mark.parametrize("strategy", ["independent-strokes", "stroke-ownership"])
def test_padding_is_clipped_to_source_image_after_mapping(strategy):
    from cutbatch import OcrHandle
    from unittest.mock import Mock
    handle = OcrHandle.__new__(OcrHandle)
    polygon = np.array([[-4, -4], [44, -4], [44, 44], [-4, 44]], np.float32)
    result = {"char_box_strategy": strategy,
              "char_list": [{"bbox": polygon.copy()}], "words": [{"bbox": polygon.copy()}]}
    handle.crnn_handle = Mock()
    handle.crnn_handle.predict_rbg_batch.return_value = [result]
    handle.crnn_handle.predict_rbg.return_value = result
    row = np.array([[0, 0], [39, 0], [39, 39], [0, 39]], np.float32)
    output = handle.crnnRecWithBox(np.zeros((40, 40, 3), np.uint8), [row], [1])
    for obj in output[0]["result"]["char_list"] + output[0]["result"]["words"]:
        assert obj["bbox"].min() >= 0 and obj["bbox"].max() <= 40
