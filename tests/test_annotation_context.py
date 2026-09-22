# -*- coding: utf-8 -*-
"""JSON 识别上下文和配置不匹配处理回归。"""

import json
import tempfile
import unittest
from unittest import mock

import numpy as np
from PIL import Image

from annotation.controller import AnnotatorCore
from utils.storage import (
    ANNOTATION_SCHEMA_VERSION,
    annotation_context_matches,
    get_json_path_for_item,
    load_item_annotation_record_if_exists,
    save_item_json,
)


def _box():
    return np.asarray(
        [[0, 0], [9, 0], [9, 9], [0, 9]],
        dtype=np.float32,
    )


def _context(level="char", rec_model="v6-small"):
    return {
        "level": level,
        "det_model": "v6-small",
        "rec_model": rec_model,
        "char_box_strategy": "polarity-aware",
    }


class AnnotationJsonContextTests(unittest.TestCase):
    def setUp(self):
        self.output_dir = tempfile.mkdtemp(prefix="chardet_context_")
        self.item = {
            "type": "image",
            "source_path": "/books/page.jpg",
            "page_index": None,
            "annotation_context": _context(),
            "box_context": _context(),
        }

    def test_new_json_records_schema_and_ocr_context(self):
        save_item_json(
            self.output_dir,
            self.item,
            [_box()],
            image_size_wh=(10, 10),
        )

        with open(
            get_json_path_for_item(self.output_dir, self.item),
            "r",
            encoding="utf-8",
        ) as json_file:
            data = json.load(json_file)

        self.assertEqual(data["schema_version"], ANNOTATION_SCHEMA_VERSION)
        self.assertEqual(data["annotation_context"], _context())

    def test_legacy_json_loads_with_unknown_context(self):
        save_item_json(
            self.output_dir,
            self.item,
            [_box()],
            image_size_wh=(10, 10),
        )
        path = get_json_path_for_item(self.output_dir, self.item)
        with open(path, "r", encoding="utf-8") as json_file:
            data = json.load(json_file)
        data.pop("schema_version")
        data.pop("annotation_context")
        with open(path, "w", encoding="utf-8") as json_file:
            json.dump(data, json_file)

        record = load_item_annotation_record_if_exists(
            self.output_dir,
            self.item,
            target_size_wh=(10, 10),
        )

        self.assertEqual(record["schema_version"], 1)
        self.assertEqual(record["annotation_context"], {})
        self.assertFalse(
            annotation_context_matches(
                record["annotation_context"],
                _context(),
            )
        )

    def test_level_or_backend_difference_is_not_compatible(self):
        self.assertTrue(annotation_context_matches(_context(), _context()))
        self.assertFalse(
            annotation_context_matches(_context("line"), _context("char"))
        )
        self.assertFalse(
            annotation_context_matches(
                _context(rec_model="crnn"),
                _context(rec_model="v6-small"),
            )
        )


class AnnotationContextChoiceTests(unittest.TestCase):
    def _core(self):
        core = AnnotatorCore()
        core.items = [
            {
                "type": "image",
                "source_path": "/books/page.jpg",
                "display_name": "page.jpg",
                "page_index": None,
                "visited": False,
                "boxes": [],
                "dirty": False,
                "ever_saved": False,
                "ocr_pending": False,
                "ocr_completed": False,
                "ocr_request_id": None,
                "_edit_revision": 0,
                "annotation_context": _context(),
                "box_context": None,
                "view_scale": None,
                "nav_scale": 1.0,
            }
        ]
        core.total_items = 1
        core.current_item_index = 0
        core._load_base_pil_for_item = mock.MagicMock(
            return_value=Image.new("RGB", (10, 10), "white")
        )
        core._apply_view_for_current_item = mock.MagicMock()
        core._refresh_canvas = mock.MagicMock()
        core.status_messages = []
        core.set_status = core.status_messages.append
        return core

    def _legacy_record(self):
        return {
            "boxes": [_box()],
            "schema_version": 1,
            "annotation_context": {},
            "json_path": "/output/page/page.json",
        }

    def test_user_can_keep_mismatched_old_boxes(self):
        core = self._core()
        core._msg_save_discard_cancel = mock.MagicMock(return_value="save")
        with mock.patch(
            "annotation.controller.load_item_annotation_record_if_exists",
            return_value=self._legacy_record(),
        ):
            self.assertTrue(_apply_in_memory(core))

        self.assertEqual(len(core.boxes), 1)
        self.assertEqual(core.items[0]["box_source"], "json")
        self.assertEqual(core.items[0]["box_context"], {})
        self.assertIn("Level 未记录", core.status_messages[-1])

    def test_matching_json_status_shows_actual_level_and_backend(self):
        core = self._core()
        record = self._legacy_record()
        record["schema_version"] = ANNOTATION_SCHEMA_VERSION
        record["annotation_context"] = _context(level="char")
        with mock.patch(
            "annotation.controller.load_item_annotation_record_if_exists",
            return_value=record,
        ):
            self.assertTrue(_apply_in_memory(core))

        self.assertIn("Level=char", core.status_messages[-1])
        self.assertIn("rec=v6-small", core.status_messages[-1])

    def test_user_can_reocr_mismatched_old_boxes(self):
        core = self._core()
        core._msg_save_discard_cancel = mock.MagicMock(
            return_value="discard"
        )
        core._start_ocr_prefetch = mock.MagicMock()
        with mock.patch(
            "annotation.controller.load_item_annotation_record_if_exists",
            return_value=self._legacy_record(),
        ):
            self.assertTrue(_apply_in_memory(core))

        self.assertEqual(core.boxes, [])
        core._start_ocr_prefetch.assert_called_once()

    def test_cancel_does_not_mark_page_visited(self):
        core = self._core()
        core._msg_save_discard_cancel = mock.MagicMock(
            return_value="cancel"
        )
        with mock.patch(
            "annotation.controller.load_item_annotation_record_if_exists",
            return_value=self._legacy_record(),
        ):
            self.assertFalse(_apply_in_memory(core))

        self.assertFalse(core.items[0]["visited"])


if __name__ == "__main__":
    unittest.main()


def _apply_in_memory(core):
    import annotation.controller as annotator_core
    pil = core._load_base_pil_for_item(core.items[core.current_item_index])
    record = annotator_core.load_item_annotation_record_if_exists(core.output_dir, core.items[core.current_item_index], target_size_wh=pil.size)
    return core._apply_loaded_item(pil, np.zeros((pil.height,pil.width,3), np.uint8), record)
