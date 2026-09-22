import copy
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image

from ocr.pipeline import _deduplicate_text_rows
from ocr.stroke_ownership import _separated_column_boxes, refine_stroke_row


def test_pale_margin_cannot_invert_dark_text_core():
    rng = np.random.default_rng(42)
    image = np.full((60, 220), 220, np.uint8)
    image[10:50] = rng.integers(25, 55, (40, 220), dtype=np.uint8)
    for x in (20, 70, 120, 170):
        image[18:43, x:x+20] = 230
    boxes = _separated_column_boxes(image, [30, 80, 130, 180], (10, 50))
    assert boxes is not None
    for box, x in zip(boxes, (20, 70, 120, 170)):
        assert x-2 <= box[0] <= x and x+20 <= box[2] <= x+22
        assert 16 <= box[1] <= 18 and 43 <= box[3] <= 45


def test_c1_has_one_box_per_visual_position_without_expensive_pixel_growth():
    root = Path(__file__).resolve().parents[1]
    fixture = json.loads((root/'tests/fixtures/c1_anchors.json').read_text(encoding='utf-8'))
    image = Image.open(root/fixture['image']).convert('RGB')
    source_w, source_h = image.size
    image.thumbnail((2560, 2560), Image.Resampling.LANCZOS)
    factors = np.array([image.width/source_w, image.height/source_h])
    bgr = np.asarray(image)[:, :, ::-1].copy()
    visual_y = np.array([104,164,231,296,358,425,487,553,615,680,739])*source_w/1100*factors[1]
    count = 0
    with patch('ocr.stroke_ownership._split_joined', side_effect=AssertionError('unexpected slow fallback')):
        for column in fixture['columns']:
            characters = copy.deepcopy(column['characters'])
            for item in characters:
                item['line'] = np.asarray(item['line'])*factors
            result = {'char_list': characters, 'words': []}
            refine_stroke_row(bgr, np.asarray(column['row_box'])*factors, result)
            assert len(characters) == 11
            for i, item in enumerate(characters):
                box = np.asarray(item['bbox'])
                top, bottom = box[:,1].min(), box[:,1].max()
                assert top <= visual_y[i] <= bottom
                if i: assert top > visual_y[i-1]
                if i < 10: assert bottom < visual_y[i+1]
                count += 1
    assert count == 110


def test_duplicate_proposals_removed_but_partial_overlap_is_allowed():
    box = np.float32([[0,0],[40,0],[40,100],[0,100]])
    almost_same = box + [1,1]
    neighbouring = box + [25,0]
    enclosing = box * 2
    boxes, scores = _deduplicate_text_rows([box,almost_same,neighbouring,enclosing],[.7,.95,.9,.8])
    assert scores == [.95,.9,.8]
    np.testing.assert_array_equal(boxes[0], almost_same)
