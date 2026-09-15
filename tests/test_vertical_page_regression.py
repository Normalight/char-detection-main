import copy
import json
from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pytest
from PIL import Image

from ppocrv6.stroke_ownership import _separated_column_boxes, refine_stroke_row, stroke_owned_boxes


def test_shaded_paper_and_page_rules_do_not_become_character_ink():
    image = np.repeat(np.linspace(180, 235, 220, dtype=np.uint8)[None], 60, axis=0)
    image[4:6, :] = 10
    image[55:57, :] = 10
    for x in (20, 70, 120, 170):
        image[18:43, x:x+20] = 0
    boxes = _separated_column_boxes(image, [30, 80, 130, 180], (10, 50))
    assert boxes is not None and len(boxes) == 4
    for box, x in zip(boxes, (20, 70, 120, 170)):
        assert x-2 <= box[0] <= x and x+20 <= box[2] <= x+22
        assert 15 <= box[1] <= 18 and 43 <= box[3] <= 46


def test_unrelated_horizontal_rule_does_not_expand_single_glyph():
    image=np.full((50,80),255,np.uint8)
    image[10:40,25:40]=0
    clean=stroke_owned_boxes(image,[32],['中'])
    image[3:5,5:70]=0
    assert stroke_owned_boxes(image,[32],['中']) == clean


def test_reported_page_keeps_46_glyphs_clear_of_frame_and_adjacent_centres():
    root=Path(__file__).resolve().parents[1]
    fixture=json.loads((root/'tests/fixtures/vertical_page_case.json').read_text(encoding='utf-8'))
    image=np.asarray(Image.open(root/fixture['image']).convert('RGB'))[:,:,::-1].copy()
    count=0
    for column in fixture['columns']:
        result={'char_list':copy.deepcopy(column['characters']), 'words':[]}
        refine_stroke_row(image, column['row_box'], result)
        centers=column['glyph_center_y']
        for i,item in enumerate(result['char_list']):
            box=np.asarray(item['bbox'])
            assert np.isfinite(box).all() and box.shape == (4,2)
            top,bottom=box[:,1].min(),box[:,1].max()
            assert top <= centers[i] <= bottom
            if i:
                assert top > centers[i-1]+9
            if i+1<len(centers):
                assert bottom < centers[i+1]-9
            count+=1
        assert np.asarray(result['char_list'][0]['bbox'])[:,1].min()>71
        assert np.asarray(result['char_list'][-1]['bbox'])[:,1].max()<560
    assert count == 46


@pytest.mark.parametrize('det,rec,suppressed',[(.55,.306,True),(.95,.306,False),(.55,.9,False),(.55,None,False)])
def test_only_jointly_weak_detection_and_recognition_are_suppressed(det,rec,suppressed):
    from cutbatch import OcrHandle
    handle=OcrHandle.__new__(OcrHandle)
    box=np.float32([[1,1],[18,1],[18,18],[1,18]])
    candidate={'char_box_strategy':'stroke-ownership','line_pred':'鱼',
               'char_list':[{'char':'鱼','bbox':box.copy()}], 'words':[]}
    if rec is not None: candidate['score']=rec
    handle.crnn_handle=Mock()
    handle.crnn_handle.predict_rbg_batch.return_value=[candidate]
    output=handle.crnnRecWithBox(np.full((20,20,3),255,np.uint8),[box],[det])[0]['result']
    assert len(output['char_list']) == (0 if suppressed else 1)
    assert ('suppressed_candidate' in output) is suppressed
    if suppressed:
        assert output['suppressed_candidate']['text']=='鱼'
