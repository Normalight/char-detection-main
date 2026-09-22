import os
os.environ.setdefault('QT_QPA_PLATFORM','offscreen')
import time
from unittest.mock import Mock
import numpy as np
import pytest
from PIL import Image
from PySide6.QtCore import QThread, QTimer
from PySide6.QtWidgets import QApplication
from annotation.controller import AnnotatorCore
import annotation.workers as worker
@pytest.fixture(scope='module')
def qt():
    app=QApplication.instance() or QApplication([])
    yield app


def spin(qt, predicate, seconds=3):
    end=time.monotonic()+seconds
    while not predicate() and time.monotonic()<end:
        qt.processEvents()
        time.sleep(.003)
    qt.processEvents()
    assert predicate()


def core_pages(tmp_path):
    core=AnnotatorCore()
    core.output_dir=str(tmp_path/'out')
    core.running=True
    core.items=[]
    for i in range(2):
        path=tmp_path/f'{i}.png';Image.new('RGB',(64,48),'white').save(path)
        core.items.append(dict(type='image',source_path=str(path),display_name=path.name,
                               page_index=None,visited=i==0,boxes=[],dirty=i==0,ever_saved=False,
                               ocr_pending=False,ocr_completed=False,_edit_revision=0))
    core.total_items=2
    core.base_image_bgr=np.full((48,64,3),255,np.uint8)
    core.base_size=(64,48)
    core.boxes=[np.float32([[10,10],[30,10],[30,30],[10,30]])]
    core._maybe_start_background_prefetch=Mock()
    core._start_ocr_prefetch=Mock()
    core._apply_view_for_current_item=Mock()
    core._refresh_canvas=Mock()
    core._msg_save_discard_cancel=Mock(return_value='save')
    core._msg_critical=Mock()
    core.set_status=Mock()
    return core


def test_save_and_turn_runs_io_off_main_thread_and_keeps_event_loop_alive(qt,tmp_path,monkeypatch):
    core=core_pages(tmp_path);ticks=[];threads=[]
    timer=QTimer();timer.setInterval(10);timer.timeout.connect(lambda:ticks.append(time.monotonic()));timer.start()
    actual_load=worker._load_item_pil
    def save(*args,**kwargs):
        threads.append(QThread.currentThread()==qt.thread());time.sleep(.2);return 1
    def excel(*args,**kwargs):
        threads.append(QThread.currentThread()==qt.thread());time.sleep(.1)
    def load(*args):
        threads.append(QThread.currentThread()==qt.thread());time.sleep(.15);return actual_load(*args)
    monkeypatch.setattr(worker,'save_current_slices_and_json',save)
    monkeypatch.setattr(worker,'build_excel_for_single_item',excel)
    monkeypatch.setattr(worker,'_load_item_pil',load)
    started=time.monotonic();assert core.next_item();assert time.monotonic()-started<.1
    assert core.current_item_index==0 and core._save_worker is not None
    core.next_item()  # duplicate navigation cannot launch another save
    spin(qt,lambda:core.current_item_index==1 and not core.any_worker_running())
    timer.stop()
    assert len(ticks)>=15 and threads==[False,False,False]
    assert core.items[0]['ever_saved'] and not core.items[0]['dirty']
    assert core.base_size==(64,48)


def test_excel_failure_does_not_turn_or_clear_dirty(qt,tmp_path,monkeypatch):
    core=core_pages(tmp_path)
    monkeypatch.setattr(worker,'save_current_slices_and_json',lambda *a,**k:1)
    monkeypatch.setattr(worker,'build_excel_for_single_item',Mock(side_effect=OSError('locked excel')))
    core.next_item();spin(qt,lambda:not core.any_worker_running())
    assert core.current_item_index==0 and core.items[0]['dirty']
    assert core._navigation_target is None
    core._msg_critical.assert_called_once()


def test_edit_during_snapshot_save_is_preserved_and_stops_navigation(qt,tmp_path,monkeypatch):
    core=core_pages(tmp_path)
    def save(*args,**kwargs): time.sleep(.15);return 1
    monkeypatch.setattr(worker,'save_current_slices_and_json',save)
    monkeypatch.setattr(worker,'build_excel_for_single_item',lambda *args,**kwargs:None)
    core.next_item();core.boxes[0]+=2;core._mark_dirty()
    spin(qt,lambda:not core.any_worker_running())
    assert core.current_item_index==0 and core.items[0]['dirty']
    assert core._navigation_target is None
    np.testing.assert_array_equal(core.boxes[0][0],[12,12])


def test_load_failure_preserves_current_image_and_index(qt,tmp_path):
    core=core_pages(tmp_path);original=core.base_image_bgr
    core.items[1]['source_path']=str(tmp_path/'missing.png')
    assert core._load_item(1)
    spin(qt,lambda:not core.any_worker_running())
    assert core.current_item_index==0 and core.base_image_bgr is original
    core._msg_critical.assert_called_once()


def test_stale_page_load_cannot_replace_a_new_session(tmp_path):
    core=core_pages(tmp_path);old=Mock();core._load_worker=Mock();core._session_generation=4
    original=core.base_image_bgr
    core._on_page_loaded(old,{'context':{'generation':3}})
    assert core.base_image_bgr is original and core.current_item_index==0


def test_real_save_and_next_preserves_pixels_and_embeds_thumbnails_without_temp_files(qt,tmp_path,monkeypatch):
    import tempfile
    import zipfile
    import cv2
    from PySide6.QtCore import QEventLoop
    from utils.storage import get_rotate_crop_image
    core=core_pages(tmp_path)
    core.base_image_bgr=np.arange(48*64*3,dtype=np.uint8).reshape(48,64,3)
    expected=get_rotate_crop_image(core.base_image_bgr,core.boxes[0])
    # XLSX must package its bounded thumbnails directly, avoiding a temporary
    # file open/write/read/delete cycle for each character on Windows.
    monkeypatch.setattr(tempfile,'mkstemp',Mock(side_effect=AssertionError('temporary thumbnail file')))
    assert core.next_item()
    loop=QEventLoop();poll=QTimer();poll.setInterval(5)
    # Worker completion queues GUI callbacks; wait for navigation to be applied,
    # since there can be an idle gap between saving and loading the next page.
    poll.timeout.connect(lambda:loop.quit() if core.current_item_index==1 and not core.any_worker_running() else None)
    deadline=QTimer();deadline.setSingleShot(True);deadline.timeout.connect(loop.quit)
    poll.start();deadline.start(3000);loop.exec();poll.stop();deadline.stop()
    assert not core.any_worker_running()
    assert core.current_item_index==1
    core._msg_critical.assert_not_called()
    crop=next((tmp_path/'out').rglob('*.png'))
    actual=cv2.imdecode(np.fromfile(crop,np.uint8),cv2.IMREAD_COLOR)
    np.testing.assert_array_equal(actual,expected)
    with zipfile.ZipFile(next((tmp_path/'out').rglob('*.xlsx'))) as archive:
        assert archive.testzip() is None
        assert len([name for name in archive.namelist() if name.startswith('xl/media/')])==1


def test_edit_while_next_original_loads_stays_on_current_page(qt,tmp_path,monkeypatch):
    core=core_pages(tmp_path);actual_load=worker._load_item_pil
    def slow_load(*args):
        time.sleep(.1)
        return actual_load(*args)
    monkeypatch.setattr(worker,'_load_item_pil',slow_load)
    assert core._load_item(1)
    core.boxes[0]+=3;core._mark_dirty()
    spin(qt,lambda:not core.any_worker_running())
    assert core.current_item_index==0 and core.items[0]['dirty']
    np.testing.assert_array_equal(core.boxes[0][0],[13,13])
