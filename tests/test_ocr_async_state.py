# -*- coding: utf-8 -*-
"""OCR 预取的页面归属、请求版本和编辑保护回归。"""

import unittest
from unittest import mock

import numpy as np

from annotator_core import AnnotatorCore


def _box(x0, y0, x1, y1):
    return np.asarray(
        [[x0, y0], [x1, y0], [x1, y1], [x0, y1]],
        dtype=np.float32,
    )


def _item(name, request_id, revision=0):
    return {
        "type": "image",
        "source_path": f"/books/{name}.jpg",
        "display_name": f"{name}.jpg",
        "page_index": None,
        "boxes": [],
        "visited": True,
        "dirty": False,
        "ever_saved": False,
        "ocr_pending": True,
        "ocr_completed": False,
        "ocr_request_id": request_id,
        "_edit_revision": revision,
    }


def _context(core, item, request_id, revision=0, target_index=0):
    return {
        "item_identity": core._item_identity(item),
        "request_id": request_id,
        "edit_revision": revision,
        "target_index": target_index,
    }


class OcrPageBindingTests(unittest.TestCase):
    def test_result_updates_bound_page_after_user_switches_page(self):
        core = AnnotatorCore()
        first = _item("first", request_id=11)
        second = _item("second", request_id=12)
        core.items = [first, second]
        core.current_item_index = 1
        core.boxes = [_box(50, 50, 60, 60)]
        core._refresh_canvas = mock.MagicMock()

        core._on_ocr_done(
            mock.MagicMock(),
            {
                "boxes": [_box(0, 0, 10, 10)],
                "context": _context(core, first, 11),
            },
        )

        self.assertEqual(len(first["boxes"]), 1)
        self.assertEqual(second["boxes"], [])
        self.assertTrue(np.allclose(core.boxes[0], _box(50, 50, 60, 60)))
        core._refresh_canvas.assert_not_called()

    def test_user_edit_prevents_late_ocr_overwrite(self):
        core = AnnotatorCore()
        item = _item("edited", request_id=21, revision=2)
        item["dirty"] = True
        item["boxes"] = [_box(30, 30, 40, 40)]
        item["_edit_revision"] = 3
        core.items = [item]
        core.current_item_index = 0
        core.boxes = [item["boxes"][0].copy()]
        core.set_status = mock.MagicMock()

        core._on_ocr_done(
            mock.MagicMock(),
            {
                "boxes": [_box(0, 0, 10, 10)],
                "context": _context(core, item, 21, revision=2),
            },
        )

        self.assertTrue(np.allclose(item["boxes"][0], _box(30, 30, 40, 40)))
        self.assertFalse(item["ocr_pending"])
        self.assertIsNone(item["ocr_request_id"])

    def test_stale_request_cannot_replace_newer_request(self):
        core = AnnotatorCore()
        item = _item("retry", request_id=32)
        core.items = [item]
        core.current_item_index = 0
        core.boxes = []

        core._on_ocr_done(
            mock.MagicMock(),
            {
                "boxes": [_box(0, 0, 10, 10)],
                "context": _context(core, item, 31),
            },
        )

        self.assertEqual(item["boxes"], [])
        self.assertTrue(item["ocr_pending"])
        self.assertEqual(item["ocr_request_id"], 32)


class OcrPendingStateTests(unittest.TestCase):
    def test_unedited_pending_page_does_not_prompt_before_navigation(self):
        core = AnnotatorCore()
        item = _item("pending", request_id=41)
        core.items = [item]
        core.current_item_index = 0
        core.boxes = []
        core.view_scale = 1.0
        core._msg_save_discard_cancel = mock.MagicMock()

        self.assertTrue(core._prompt_save_current_if_needed())

        core._msg_save_discard_cancel.assert_not_called()

    def test_unedited_pending_page_is_not_unsaved_content(self):
        core = AnnotatorCore()
        core.items = [_item("pending", request_id=42)]

        self.assertFalse(core._has_unsaved_items())

    def test_save_all_result_invalidates_prefetch_request(self):
        core = AnnotatorCore()
        item = _item("pending", request_id=43)
        core.items = [item]
        saved = dict(item)
        saved["boxes"] = [_box(0, 0, 10, 10)]
        saved["ever_saved"] = True

        core._on_save_page_saved(saved)

        self.assertFalse(item["ocr_pending"])
        self.assertTrue(item["ocr_completed"])
        self.assertIsNone(item["ocr_request_id"])
        self.assertTrue(item["ever_saved"])
        self.assertFalse(item["dirty"])


class OcrErrorBindingTests(unittest.TestCase):
    def test_stale_request_error_does_not_interrupt_current_request(self):
        core = AnnotatorCore()
        item = _item("retry", request_id=52)
        core.items = [item]
        core._on_worker_error = mock.MagicMock()
        worker = mock.MagicMock()
        worker._context = _context(core, item, request_id=51)

        core._on_ocr_error(worker, "旧请求失败")

        self.assertTrue(item["ocr_pending"])
        self.assertEqual(item["ocr_request_id"], 52)
        core._on_worker_error.assert_not_called()

    def test_current_request_error_clears_pending_and_is_reported(self):
        core = AnnotatorCore()
        item = _item("current", request_id=61)
        core.items = [item]
        core._on_worker_error = mock.MagicMock()
        worker = mock.MagicMock()
        worker._context = _context(core, item, request_id=61)

        core._on_ocr_error(worker, "当前请求失败")

        self.assertFalse(item["ocr_pending"])
        self.assertIsNone(item["ocr_request_id"])
        core._on_worker_error.assert_called_once_with("当前请求失败")


if __name__ == "__main__":
    unittest.main()
