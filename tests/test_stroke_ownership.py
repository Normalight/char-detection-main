import numpy as np
import pytest
from ocr.stroke_ownership import stroke_owned_boxes, refine_stroke_row


def test_faint_tip_grows_from_strong_stroke_without_proportional_padding():
    image = np.full((50, 80), 255, np.uint8)
    image[10:40, 25:40] = 0
    image[12:14, 12:27] = 210
    boxes, masks = stroke_owned_boxes(image, [32], ["中"], return_masks=True)
    assert boxes == [(11, 9, 41, 41)]
    assert masks[0][12:14, 12:27].all()
    assert masks[0].sum() == np.count_nonzero(image != 255)


def test_global_grouping_recovers_radical_past_ctc_midpoint():
    image = np.full((40, 90), 255, np.uint8)
    image[5:33, 15:22] = 0
    image[7:32, 44:49] = 0
    image[6:34, 58:68] = 0
    boxes, masks = stroke_owned_boxes(image, [30, 60], ["你", "好"], return_masks=True)
    assert boxes[0][0] <= 15 and boxes[0][2] >= 49
    assert masks[0][7:32, 44:49].all()
    assert not masks[1][7:32, 44:49].any()
    assert boxes[0][2] < boxes[1][0]


@pytest.mark.parametrize("background,ink", [(255, 0), (20, 235)])
def test_interleaved_glyph_masks_are_disjoint_but_boxes_overlap(background, ink):
    image = np.full((48, 80), background, np.uint8)
    image[5:33, 12:24] = ink
    image[5:10, 22:45] = ink
    image[17:43, 52:64] = ink
    image[38:43, 35:54] = ink
    boxes, masks = stroke_owned_boxes(image, [20, 58], return_masks=True)
    assert boxes[0][2] > boxes[1][0]
    assert not (masks[0] & masks[1]).any()
    assert (masks[0] | masks[1]).sum() == np.count_nonzero(image != background)
    assert boxes[0][3] < boxes[1][3] and boxes[0][1] < boxes[1][1]


def test_joined_component_is_fully_owned_without_background_growth():
    image = np.full((40, 100), 255, np.uint8)
    image[7:33, 15:30] = 0
    image[7:33, 65:80] = 0
    image[20:23, 25:70] = 0
    boxes, masks = stroke_owned_boxes(image, [25, 70], return_masks=True)
    assert masks[0] is not None and masks[1] is not None
    assert np.array_equal(masks[0] | masks[1], image == 0)
    assert not (masks[0] & masks[1]).any()
    assert boxes[0][2] < 65 and boxes[1][0] > 30


def test_small_glyph_is_not_forced_to_cjk_square():
    image = np.full((50, 80), 255, np.uint8)
    image[16:35, 35:43] = 0
    assert stroke_owned_boxes(image, [40], ["中"]) == [(34, 15, 44, 36)]


def test_disconnected_noise_does_not_expand_glyph():
    image = np.full((50, 80), 255, np.uint8)
    image[10:40, 25:40] = 0
    image[8, 15] = 0
    image[15:18, 10:13] = 235
    assert stroke_owned_boxes(image, [32]) == [(24, 9, 41, 41)]


@pytest.mark.parametrize("centers", [[20, 10], [10, 10], [float("nan")], [-1], [80]])
def test_invalid_anchors(centers):
    with pytest.raises(ValueError):
        stroke_owned_boxes(np.zeros((40, 80), np.uint8), centers)


def test_blank_and_empty_output_are_supported():
    assert stroke_owned_boxes(np.zeros((40, 80), np.uint8), []) == []
    boxes, masks = stroke_owned_boxes(np.zeros((40, 80), np.uint8), [20, 60], return_masks=True)
    assert len(boxes) == 2 and masks == [None, None]


@pytest.mark.parametrize("vertical", [False, True])
def test_original_context_recovers_strokes_beyond_detector_row(vertical):
    image = np.full((100, 100, 3), 255, np.uint8)
    image[25:57, 25:47] = 0
    if vertical:
        row = np.float32([[28, 20], [44, 20], [44, 65], [28, 65]])
        line = [[28, 40], [44, 40]]
    else:
        row = np.float32([[20, 28], [65, 28], [65, 54], [20, 54]])
        line = [[36, 28], [36, 54]]
    result = {"char_list": [{"char": "甲", "line": line}], "words": []}
    refine_stroke_row(image, row, result)
    box = result["char_list"][0]["bbox"]
    assert box[:, 0].min() <= 25 and box[:, 0].max() >= 47
    assert box[:, 1].min() <= 25 and box[:, 1].max() >= 57
    assert box[:, 0].max() - box[:, 0].min() < 28
    assert box[:, 1].max() - box[:, 1].min() < 38


def test_large_joined_component_retains_full_foreground_after_bounded_search():
    image = np.full((250, 600), 255, np.uint8)
    image[25:220, 40:190] = 0
    image[25:220, 400:550] = 0
    image[120:124, 180:410] = 0
    _, masks = stroke_owned_boxes(image, [110, 470], return_masks=True)
    assert masks[0] is not None and masks[1] is not None
    assert np.array_equal(masks[0] | masks[1], image == 0)
    assert not (masks[0] & masks[1]).any()


def test_spaces_do_not_steal_neighbour_ink():
    image = np.full((40, 100), 255, np.uint8)
    image[8:32, 10:26] = 0
    image[8:32, 72:88] = 0
    boxes, masks = stroke_owned_boxes(image, [18, 50, 80], ["A", " ", "B"], return_masks=True)
    assert masks[1] is None
    assert masks[0].sum() == masks[2].sum() == 24*16
    assert boxes[0][2] < boxes[2][0]
