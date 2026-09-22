import numpy as np
import pytest
from unittest.mock import Mock
from utils import storage, excel


def page():
    image = np.arange(80*100*3, dtype=np.uint8).reshape(80,100,3)
    boxes = [np.array([[x,0],[x+20,0],[x+20,30],[x,30]], dtype=np.float32) for x in (0,40)]
    return image, boxes, {'type':'image','source_path':'/tmp/test.png','boxes':boxes}


def test_repeat_and_single_edit_reuse_outputs(tmp_path, monkeypatch):
    image, boxes, item = page()
    storage.save_current_slices_and_json(str(tmp_path), item, image, boxes)
    excel.build_excel_for_single_item(str(tmp_path), item)
    cropper = Mock(wraps=storage.get_rotate_crop_image)
    monkeypatch.setattr(storage, 'get_rotate_crop_image', cropper)
    thumbnail = Mock(wraps=excel._make_thumbnail_bytes)
    monkeypatch.setattr(excel, '_make_thumbnail_bytes', thumbnail)
    workbook = next(tmp_path.rglob('*.xlsx')); before = workbook.stat().st_mtime_ns
    storage.save_current_slices_and_json(str(tmp_path), item, image, boxes)
    excel.build_excel_for_single_item(str(tmp_path), item)
    assert cropper.call_count == thumbnail.call_count == 0
    assert workbook.stat().st_mtime_ns == before
    boxes[0] += 1
    storage.save_current_slices_and_json(str(tmp_path), item, image, boxes)
    excel.build_excel_for_single_item(str(tmp_path), item)
    assert cropper.call_count == thumbnail.call_count == 1
    assert workbook.stat().st_mtime_ns != before


def test_changed_pixels_missing_file_and_delete_are_not_stale(tmp_path, monkeypatch):
    image, boxes, item = page()
    storage.save_current_slices_and_json(str(tmp_path), item, image, boxes)
    cropper = Mock(wraps=storage.get_rotate_crop_image)
    monkeypatch.setattr(storage, 'get_rotate_crop_image', cropper)
    image[0,0] = 0
    storage.save_current_slices_and_json(str(tmp_path), item, image, boxes)
    assert cropper.call_count == 2
    next(tmp_path.rglob('*.png')).unlink()
    cropper.reset_mock()
    storage.save_current_slices_and_json(str(tmp_path), item, image, boxes)
    assert cropper.call_count == 1
    storage.save_current_slices_and_json(str(tmp_path), item, image, boxes[:1])
    assert len(list(tmp_path.rglob('*.png'))) == 1


def test_excel_failure_preserves_previous_workbook(tmp_path, monkeypatch):
    image, boxes, item = page()
    storage.save_current_slices_and_json(str(tmp_path), item, image, boxes)
    excel.build_excel_for_single_item(str(tmp_path), item)
    workbook = next(tmp_path.rglob('*.xlsx')); before = workbook.read_bytes()
    def fail(*args, **kwargs):
        raise OSError('disk full')
    monkeypatch.setattr(excel.xlsxwriter, 'Workbook', fail)
    with pytest.raises(OSError):
        excel.build_excel_for_single_item(str(tmp_path), item, cell_size_px=90)
    assert workbook.read_bytes() == before


def test_save_waits_for_ocr_and_is_cancellable():
    from annotation.workers import SaveCurrentWorker
    worker = SaveCurrentWorker('/tmp/unused', {}, None)
    dependency = Mock()
    dependency.wait.side_effect = [False, True]
    worker.save_dependencies = [dependency]
    assert worker._wait_for_save_dependencies()
    assert dependency.wait.call_count == 2
    worker.cancel()
    assert not worker._wait_for_save_dependencies()
