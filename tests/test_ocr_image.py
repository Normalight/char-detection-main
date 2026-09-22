from unittest.mock import Mock
import numpy as np
from PIL import Image
import config
from ocr.pipeline import OcrHandle
from ocr.image import load_ocr_image, prepare_ocr_image, target_size
from annotation.workers import OcrPrefetchWorker, OcrBatchPrefetchWorker
from utils.storage import save_item_boxes_to_disk


def fake_handle():
    h=OcrHandle.__new__(OcrHandle);h.det_name='v6-small';h.text_handle=Mock()
    h.text_handle.process.return_value=([],[])
    box=np.float32([[10,10],[30,10],[30,30],[10,30]])
    h.recognize_rows=lambda *args,**kwargs:[{'bbox':box.copy(),'score':.99,'result':{
        'char_box_strategy':'stroke-ownership','char_list':[{'char':'甲','bbox':box.copy(),'line':[[20,10],[20,30]]}],
        'words':[{'value':'甲','bbox':box.copy()}]}}]
    return h


def test_large_jpeg_uses_scaled_decode_and_exact_original_dimensions(tmp_path,monkeypatch):
    path=tmp_path/'large.jpg'
    image=Image.new('RGB',(8001,5003),'white');image.save(path);image.close()
    converted=[];original=Image.Image.convert
    def tracked(image,*args,**kwargs):
        converted.append(image.size);return original(image,*args,**kwargs)
    monkeypatch.setattr(Image.Image,'convert',tracked)
    prepared=load_ocr_image(path)
    assert prepared.source_size==(8001,5003)
    assert prepared.target_size==target_size((8001,5003))
    assert all(w<8001 for w,h in converted)
    np.testing.assert_allclose(prepared.scale_back,[8001/2560,5003/prepared.image.height])


def test_disabled_legacy_limit_cannot_bypass_large_image_bound():
    assert max(target_size((40000,30000),max_side=0))==2560


def test_current_prefetch_batch_prefetch_and_save_fallback_share_cap_and_mapping(tmp_path,monkeypatch):
    monkeypatch.setattr(config,'ocr_max_side',160)
    path=tmp_path/'page.jpg';Image.new('RGB',(1600,1200),'white').save(path)
    image=Image.open(path).convert('RGB');handle=fake_handle();payloads=[]
    item={'type':'image','source_path':str(path),'display_name':'page.jpg','boxes':[],'context':{}}
    current=OcrPrefetchWorker(image,6000,'char',handle)
    current.ocr_done.connect(payloads.append);current.run()
    batch=OcrBatchPrefetchWorker([item],handle,6000,'char',200,None,1)
    batch.page_ocr_done.connect(payloads.append);batch.run()
    assert len(payloads)==2
    for payload in payloads:
        np.testing.assert_allclose(payload['boxes'][0],[[100,100],[300,100],[300,300],[100,300]])
        assert payload['ocr_geometry']=={'source_size':[1600,1200],'ocr_size':[160,120]}
    assert save_item_boxes_to_disk(str(tmp_path/'out'),item,image,handle,6000,'char')==1
    for call in handle.text_handle.process.call_args_list:
        pixels,short_size=call.args
        assert pixels.shape[:2]==(120,160) and short_size==120
    crop=next((tmp_path/'out').rglob('*.png'))
    assert Image.open(crop).size==(200,200)
