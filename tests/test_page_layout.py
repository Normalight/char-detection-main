import copy
import json
from pathlib import Path
from unittest.mock import patch
import cv2
import numpy as np
import pytest
from PIL import Image
from ppocrv6.page_layout import refine_regular_rubbing_page


def regular_page():
    image=np.full((650,600,3),35,np.uint8)
    rows=[]; truth=[]
    for col in range(8):
        x=50+col*65; chars=[]
        for j in range(11):
            y=50+j*50
            half_w=24 if (col,j)==(0,2) else 15
            half_h=3 if (col,j)==(1,4) else 18
            image[y-half_h:y+half_h,x-half_w:x+half_w]=230
            image[y-2:y+1,x+28:x+31]=210  # detached peripheral noise
            truth.append((x,y,half_w,half_h))
            if (col,j)!=(0,5):
                chars.append({'char':'甲','score':.9,'line':[[x-27,y],[x+27,y]]})
        rows.append({'bbox':np.float32([[x-27,28],[x+27,28],[x+27,572],[x-27,572]]),
                     'score':.9,'result':{'char_box_strategy':'stroke-ownership','char_list':chars,'words':[]}})
    return image,rows,truth


def test_page_prior_preserves_long_strokes_thin_glyphs_and_missing_token():
    image,rows,truth=regular_page()
    handled=refine_regular_rubbing_page(image,rows)
    assert len(handled)==8
    chars=[c for r in rows for c in r['result']['char_list']]
    assert len(chars)==88 and sum(c.get('unrecognized',False) for c in chars)==1
    for c,(x,y,hw,hh) in zip(chars,truth):
        box=np.asarray(c['bbox'])
        lo,hi=box.min(axis=0),box.max(axis=0)
        assert (lo<=[x-hw,y-hh]).all() and (hi>=[x+hw,y+hh]).all()
        assert hi[0]-lo[0] <= 2*hw+4  # no detached side speck
        assert hi[1]-lo[1] <= 2*hh+4


def test_overlapping_detector_fragments_do_not_duplicate_slots():
    image,rows,truth=regular_page()
    fragment=copy.deepcopy(rows[1])
    fragment['bbox'][2:,1]=185
    fragment['result']['char_list']=fragment['result']['char_list'][:3]
    rows.append(fragment)
    refine_regular_rubbing_page(image,rows)
    chars=[c for r in rows for c in r['result']['char_list']]
    assert len(chars)==88


def test_light_pages_do_not_get_forced_into_dark_page_grid():
    image,rows,_=regular_page()
    before=copy.deepcopy(rows)
    assert refine_regular_rubbing_page(255-image,rows)==set()
    for a,b in zip(rows,before):
        assert 'geometry_source' not in a['result']
        assert len(a['result']['char_list'])==len(b['result']['char_list'])


@pytest.mark.parametrize('number',[1,2,3,4])
def test_new_images_do_not_cover_adjacent_body_centres(number):
    root=Path(__file__).resolve().parents[1]
    path=root/f'newImages/{number}.jpg'
    if not path.exists(): pytest.skip('user image set is not present')
    data=json.loads((root/'tests/fixtures/new_images_anchors.json').read_text(encoding='utf-8'))
    rows=data[str(number)]
    im=Image.open(path).convert('RGB'); w,h=im.size
    im=im.resize((2560,round(h*2560/w)),Image.Resampling.LANCZOS)
    scale=np.array([im.width/w,im.height/h])
    for row in rows:
        row['bbox']=np.asarray(row['bbox'])*scale
        for c in row['result']['char_list']: c['line']=np.asarray(c['line'])*scale
    handled=refine_regular_rubbing_page(np.asarray(im)[:,:,::-1].copy(),rows)
    assert len(handled)>=18
    boxes=[np.asarray(c['bbox']) for i in handled for c in rows[i]['result']['char_list']]
    assert 270<=len(boxes)<=288
    centers=np.array([b.mean(axis=0) for b in boxes])
    for i,box in enumerate(boxes):
        lo,hi=box.min(axis=0),box.max(axis=0)
        inside=np.all((centers>lo+3)&(centers<hi-3),axis=1)
        assert inside.sum()==1
    if number==3:
        # Manually inspected stroke tips: 史's long stroke and 曹's lower 日.
        for point,anchor_y in (([5823,763],622),([5698,1017],887),([5697,1537],1417)):
            probe=np.asarray(point)*scale
            characters=[c for i in handled for c in rows[i]['result']['char_list']]
            target=np.asarray([5700,anchor_y])*scale
            character=min(characters,key=lambda c:np.linalg.norm(np.asarray(c['line']).mean(axis=0)-target))
            assert cv2.pointPolygonTest(np.asarray(character['bbox'],np.float32),tuple(probe),False)>=0


def test_pipeline_can_defer_only_the_redundant_local_refinement():
    from ppocrv6.recognizer import PPOcrV6RecHandle
    handle=PPOcrV6RecHandle.__new__(PPOcrV6RecHandle)
    handle.char_box_strategy='stroke-ownership'
    handle._decode_ctc=lambda _:('甲乙',[{'char':'甲','idx':10,'score':.9},{'char':'乙','idx':30,'score':.9}])
    with patch('ppocrv6.recognizer.stroke_owned_boxes',side_effect=AssertionError('redundant refinement')):
        result=handle._postprocess(np.zeros((1,40,3)),1,80,Image.new('RGB',(80,48)),
                                   np.float32([[0,0],[80,0],[80,48],[0,48]]),defer_stroke_refinement=True)
    assert result['line_pred']=='甲乙'
    assert all('line' in c and 'bbox' in c for c in result['char_list'])


def test_pale_page_edge_cannot_fill_an_empty_unrecognized_slot():
    image,rows,_=regular_page()
    image[280:320,20:82]=35  # remove the token already absent from OCR
    image[28:572,18:23]=230  # page edge beside the dark body column
    refine_regular_rubbing_page(image,rows)
    chars=[c for row in rows for c in row['result']['char_list']]
    assert len(chars)==87
    assert not any(c.get('unrecognized') for c in chars)
