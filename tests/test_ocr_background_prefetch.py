# -*- coding: utf-8 -*-
"""空闲后台预取（批量预标注未访问页）的调度与状态回归。"""

import unittest
from unittest import mock

import numpy as np
from PIL import Image

import annotation.controller as annotator_core
from annotation.controller import AnnotatorCore
from annotation.workers import OcrBatchPrefetchWorker


def _box(x0, y0, x1, y1):
    return np.asarray(
        [[x0, y0], [x1, y0], [x1, y1], [x0, y1]],
        dtype=np.float32,
    )


def _item(name, **overrides):
    item = {
        "type": "image",
        "source_path": f"/books/{name}.jpg",
        "display_name": f"{name}.jpg",
        "page_index": None,
        "boxes": [],
        "visited": False,
        "dirty": False,
        "ever_saved": False,
        "ocr_pending": False,
        "ocr_completed": False,
        "ocr_request_id": None,
        "_edit_revision": 0,
    }
    item.update(overrides)
    return item


def _core(items, current=0, handler=object()):
    core = AnnotatorCore()
    core.items = items
    core.current_item_index = current
    core.ocr_handler = handler
    core.set_status = mock.MagicMock()
    return core


class CandidateFilterTests(unittest.TestCase):
    def test_only_untouched_pages_without_json_are_candidates(self):
        core = _core([_item("clean")])
        with mock.patch.object(
            annotator_core,
            "load_item_annotation_record_if_exists",
            return_value=None,
        ):
            self.assertTrue(core._item_needs_background_ocr(core.items[0]))

    def test_visited_dirty_saved_pending_or_failed_pages_are_skipped(self):
        cases = [
            _item("visited", visited=True),
            _item("boxes", boxes=[_box(0, 0, 5, 5)]),
            _item("dirty", dirty=True),
            _item("saved", ever_saved=True),
            _item("pending", ocr_pending=True),
            _item("completed", ocr_completed=True),
            _item("failed", ocr_failed=True),
        ]
        core = _core(cases)
        for item in cases:
            with mock.patch.object(
                annotator_core,
                "load_item_annotation_record_if_exists",
                return_value=None,
            ):
                self.assertFalse(core._item_needs_background_ocr(item))

    def test_page_with_saved_json_is_skipped(self):
        core = _core([_item("json")])
        with mock.patch.object(
            annotator_core,
            "load_item_annotation_record_if_exists",
            return_value={"boxes": []},
        ):
            self.assertFalse(core._item_needs_background_ocr(core.items[0]))


class PrefetchSchedulingTests(unittest.TestCase):
    def _fake_worker_class(self):
        created = []

        class SignalStub:
            def connect(self, *_args, **_kwargs):
                pass

        class FakeWorker:
            page_ocr_done = SignalStub()
            page_ocr_failed = SignalStub()
            error = SignalStub()
            finished = SignalStub()

            def __init__(self, jobs, *args, **kwargs):
                self.jobs = list(jobs)
                self.started = False
                self.cancelled = False
                created.append(self)

            def cancel(self):
                self.cancelled = True

            def start(self):
                self.started = True

        return FakeWorker, created

    def test_prefetch_enqueues_only_eligible_pages(self):
        core = _core(
            [
                _item("current", visited=True, boxes=[_box(0, 0, 5, 5)]),
                _item("next"),
                _item("withjson"),
            ],
            current=0,
        )

        def fake_record(_root, item, **_kwargs):
            return {"boxes": []} if item["display_name"] == "withjson.jpg" else None

        fake_cls, created = self._fake_worker_class()
        with (
            mock.patch.object(
                annotator_core,
                "load_item_annotation_record_if_exists",
                side_effect=fake_record,
            ),
            mock.patch("annotation.workers.OcrBatchPrefetchWorker", fake_cls),
        ):
            core._maybe_start_background_prefetch()

        self.assertEqual(len(created), 1)
        worker = created[0]
        self.assertTrue(worker.started)
        self.assertIs(core._prefetch_worker, worker)
        self.assertEqual(len(worker.jobs), 1)
        self.assertEqual(worker.jobs[0]["display_name"], "next.jpg")
        enqueued = core.items[1]
        self.assertTrue(enqueued["ocr_pending"])
        self.assertEqual(
            enqueued["ocr_request_id"],
            worker.jobs[0]["context"]["request_id"],
        )
        self.assertEqual(worker.prefetch_request_ids,
                         frozenset([enqueued["ocr_request_id"]]))

    def test_prefetch_waits_when_busy_or_priority_conflicts(self):
        fake_cls, created = self._fake_worker_class()
        with mock.patch("annotation.workers.OcrBatchPrefetchWorker", fake_cls):
            # 模型未就绪
            core = _core([_item("a")], handler=None)
            core._maybe_start_background_prefetch()
            # 当前页 OCR 进行中
            core = _core([_item("cur", ocr_pending=True), _item("b")], current=0)
            with mock.patch.object(
                annotator_core,
                "load_item_annotation_record_if_exists",
                return_value=None,
            ):
                core._maybe_start_background_prefetch()
            # 批量保存运行中
            core = _core([_item("c")])
            save_worker = mock.MagicMock()
            save_worker.isRunning.return_value = True
            core._save_worker = save_worker
            with mock.patch.object(
                annotator_core,
                "load_item_annotation_record_if_exists",
                return_value=None,
            ):
                core._maybe_start_background_prefetch()
            # 已有预取在跑
            core = _core([_item("d")])
            core._prefetch_worker = object()
            with mock.patch.object(
                annotator_core,
                "load_item_annotation_record_if_exists",
                return_value=None,
            ):
                core._maybe_start_background_prefetch()

        self.assertEqual(created, [])

    def test_cancel_resets_only_batch_pending_flags(self):
        core = _core(
            [
                _item("in-batch", ocr_pending=True, ocr_request_id=5),
                _item("other", ocr_pending=True, ocr_request_id=6),
            ]
        )
        worker = mock.MagicMock()
        worker.prefetch_request_ids = frozenset({5})
        core._prefetch_worker = worker

        core._cancel_background_prefetch()

        worker.cancel.assert_called_once()
        self.assertIsNone(core._prefetch_worker)
        self.assertFalse(core.items[0]["ocr_pending"])
        self.assertIsNone(core.items[0]["ocr_request_id"])
        self.assertTrue(core.items[1]["ocr_pending"])
        self.assertEqual(core.items[1]["ocr_request_id"], 6)

    def test_thread_finish_resets_and_continues(self):
        core = _core([_item("left", ocr_pending=True, ocr_request_id=9)])
        worker = mock.MagicMock()
        worker.prefetch_request_ids = frozenset({9})
        core._prefetch_worker = worker
        core._maybe_start_background_prefetch = mock.MagicMock()

        core._on_prefetch_thread_finished(worker)

        self.assertIsNone(core._prefetch_worker)
        self.assertFalse(core.items[0]["ocr_pending"])
        core._maybe_start_background_prefetch.assert_called_once_with()

    def test_failed_page_is_marked_and_not_refetched(self):
        core = _core([_item("bad", ocr_pending=True, ocr_request_id=7)])
        context = {
            "item_identity": core._item_identity(core.items[0]),
            "request_id": 7,
            "target_index": 0,
        }

        core._on_prefetch_page_failed({"context": context})

        item = core.items[0]
        self.assertFalse(item["ocr_pending"])
        self.assertIsNone(item["ocr_request_id"])
        self.assertTrue(item["ocr_failed"])
        with mock.patch.object(
            annotator_core,
            "load_item_annotation_record_if_exists",
            return_value=None,
        ):
            self.assertFalse(core._item_needs_background_ocr(item))

    def test_current_page_ocr_cancels_background_prefetch(self):
        core = _core([_item("cur")])
        core._cancel_background_prefetch = mock.MagicMock()
        core.ocr_handler = object()

        class SignalStub:
            def connect(self, *_args, **_kwargs):
                pass

        class FakeSingleWorker:
            progress = SignalStub()
            ocr_done = SignalStub()
            error = SignalStub()
            finished = SignalStub()

            def __init__(self, *args, **kwargs):
                pass

            def start(self):
                pass

        with mock.patch("annotation.workers.OcrPrefetchWorker", FakeSingleWorker):
            core._track_worker = mock.MagicMock()
            core._start_ocr_prefetch(Image.new("RGB", (4, 4)))

        core._cancel_background_prefetch.assert_called_once_with()

    def test_save_all_cancels_background_prefetch(self):
        core = _core([_item("a")])
        core._cancel_background_prefetch = mock.MagicMock()
        core._sync_current_boxes_to_item = mock.MagicMock()

        class FakeSaveWorker:
            def __init__(self, *args, **kwargs):
                pass

            def start(self):
                pass

        with mock.patch("annotation.workers.SaveAllWorker", FakeSaveWorker):
            core._track_worker = mock.MagicMock()
            core.save_all()

        core._cancel_background_prefetch.assert_called_once_with()


class LoadItemPrefetchDisplayTests(unittest.TestCase):
    def _core_ready_to_load(self, item):
        core = _core([item])
        core.base_size = None
        core._load_base_pil_for_item = mock.MagicMock(
            return_value=Image.new("RGB", (32, 32))
        )
        core._apply_view_for_current_item = mock.MagicMock()
        core._refresh_canvas = mock.MagicMock()
        core._maybe_start_background_prefetch = mock.MagicMock()
        core._start_ocr_prefetch = mock.MagicMock()
        return core

    def test_prefetched_page_displays_without_new_ocr(self):
        item = _item(
            "ready",
            ocr_completed=True,
            boxes=[_box(1, 1, 6, 6)],
            box_context={"level": "char", "det_model": "v6-small",
                         "rec_model": "v6-small", "char_box_strategy": "polarity-aware"},
        )
        core = self._core_ready_to_load(item)

        with mock.patch.object(
            annotator_core,
            "load_item_annotation_record_if_exists",
            return_value=None,
        ):
            self.assertTrue(core._apply_loaded_item(Image.new('RGB',(32,32)), np.zeros((32,32,3),np.uint8), None))

        self.assertEqual(len(core.boxes), 1)
        core._start_ocr_prefetch.assert_not_called()
        status = core.set_status.call_args[0][0]
        self.assertIn("后台预标注", status)
        core._maybe_start_background_prefetch.assert_called_once_with()

    def test_pending_prefetched_page_preempts_batch(self):
        item = _item("inflight", ocr_pending=True, ocr_request_id=3)
        core = self._core_ready_to_load(item)
        core._cancel_background_prefetch = mock.MagicMock()

        with mock.patch.object(
            annotator_core,
            "load_item_annotation_record_if_exists",
            return_value=None,
        ):
            self.assertTrue(core._apply_loaded_item(Image.new('RGB',(32,32)), np.zeros((32,32,3),np.uint8), None))

        self.assertEqual(core.boxes, [])
        core._cancel_background_prefetch.assert_called_once_with()
        core._start_ocr_prefetch.assert_called_once()


class PrefetchWorkerBehaviorTests(unittest.TestCase):
    def _collect(self, worker):
        done, failed, ok = [], [], []
        worker.page_ocr_done.connect(done.append)
        worker.page_ocr_failed.connect(failed.append)
        worker.finished_ok.connect(lambda: ok.append(True))
        return done, failed, ok

    def test_worker_sequentially_emits_and_skips_failures(self):
        import tempfile, os

        with tempfile.TemporaryDirectory() as tmp:
            good = os.path.join(tmp, "good.png")
            Image.new("RGB", (24, 24), "white").save(good)
            jobs = [
                {
                    "type": "image",
                    "source_path": good,
                    "page_index": None,
                    "display_name": "good.png",
                    "context": {"request_id": 1, "item_identity": ("good",)},
                },
                {
                    "type": "image",
                    "source_path": os.path.join(tmp, "missing.png"),
                    "page_index": None,
                    "display_name": "missing.png",
                    "context": {"request_id": 2, "item_identity": ("missing",)},
                },
            ]
            boxes = [_box(0, 0, 4, 4)]

            class FakeOcr:
                def text_predict(self, pil, size):
                    return [{"bbox": None, "result": {}, "score": 1.0}]

            worker = OcrBatchPrefetchWorker(
                jobs, FakeOcr(), 960, "char", 200, None, 1.0
            )
            with mock.patch(
                "annotation.workers.extract_boxes_from_ocr_results", return_value=boxes
            ):
                done, failed, ok = self._collect(worker)
                worker.run()

        self.assertEqual(len(done), 1)
        self.assertEqual(done[0]["context"]["request_id"], 1)
        self.assertTrue(
            np.allclose(np.asarray(done[0]["boxes"][0]), boxes[0])
        )
        self.assertEqual(len(failed), 1)
        self.assertEqual(failed[0]["context"]["request_id"], 2)
        self.assertEqual(ok, [True])

    def test_cancelled_worker_stops_between_pages(self):
        jobs = [
            {"type": "image", "source_path": "a.png", "page_index": None,
             "display_name": "a", "context": {"request_id": 1}},
            {"type": "image", "source_path": "b.png", "page_index": None,
             "display_name": "b", "context": {"request_id": 2}},
        ]

        class FakeOcr:
            def text_predict(self, pil, size):
                return []

        worker = OcrBatchPrefetchWorker(
            jobs, FakeOcr(), 960, "char", 200, None, 1.0
        )
        done, failed, ok = self._collect(worker)
        worker.cancel()
        with mock.patch(
            "annotation.workers.extract_boxes_from_ocr_results", return_value=[]
        ) as extract:
            worker.run()

        self.assertEqual(done, [])
        self.assertEqual(failed, [])
        self.assertEqual(ok, [])
        extract.assert_not_called()


class PrefetchUnsavedSemanticsTests(unittest.TestCase):
    def test_prefetched_pages_still_count_as_unsaved_on_quit(self):
        core = _core(
            [
                _item(
                    "prefetched",
                    ocr_completed=True,
                    boxes=[_box(0, 0, 5, 5)],
                )
            ]
        )
        self.assertTrue(core._has_unsaved_items())

    def test_unsaved_prefetched_page_prompts_save_on_navigation(self):
        """未落盘的预取页与旧版纯 OCR 页一致：翻页前询问保存。"""
        core = _core(
            [
                _item(
                    "prefetched",
                    visited=True,
                    ocr_completed=True,
                    boxes=[_box(0, 0, 5, 5)],
                )
            ]
        )
        core.view_scale = 1.0
        core.boxes = [core.items[0]["boxes"][0].copy()]
        core._msg_save_discard_cancel = mock.MagicMock(
            return_value="discard"
        )

        self.assertTrue(core._prompt_save_current_if_needed())
        core._msg_save_discard_cancel.assert_called_once()

    def test_pending_page_without_boxes_does_not_prompt_on_navigation(self):
        """OCR 尚未返回、画布为空的页没有可保存内容，翻页不打扰。"""
        core = _core(
            [
                _item(
                    "pending",
                    visited=True,
                    ocr_pending=True,
                    ocr_request_id=7,
                )
            ]
        )
        core.view_scale = 1.0
        core.boxes = []
        core._msg_save_discard_cancel = mock.MagicMock()

        self.assertTrue(core._prompt_save_current_if_needed())
        core._msg_save_discard_cancel.assert_not_called()

    def test_edited_page_still_prompts_save_on_navigation(self):
        core = _core([_item("edited", visited=True, dirty=True)])
        core.view_scale = 1.0
        core.boxes = [_box(0, 0, 5, 5)]
        core._msg_save_discard_cancel = mock.MagicMock(return_value="cancel")

        self.assertFalse(core._prompt_save_current_if_needed())
        core._msg_save_discard_cancel.assert_called_once()


if __name__ == "__main__":
    unittest.main()
