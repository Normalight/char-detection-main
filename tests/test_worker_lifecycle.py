# -*- coding: utf-8 -*-
"""全部 worker 跟踪和重复会话保护回归。"""

from pathlib import Path
import unittest
from unittest import mock

from annotation.controller import AnnotatorCore


class WorkerRegistryTests(unittest.TestCase):
    def test_registry_keeps_older_worker_visible(self):
        core = AnnotatorCore()
        older = mock.MagicMock()
        newer = mock.MagicMock()
        older.isRunning.return_value = True
        newer.isRunning.return_value = True
        core._track_worker(older)
        core._track_worker(newer)
        core._ocr_worker = newer

        active = core.active_workers()

        self.assertTrue(any(worker is older for worker in active))
        self.assertTrue(any(worker is newer for worker in active))

    def test_cancel_all_workers_reaches_every_tracked_worker(self):
        core = AnnotatorCore()
        workers = [mock.MagicMock(), mock.MagicMock()]
        for worker in workers:
            worker.isRunning.return_value = True
            core._track_worker(worker)

        core.cancel_all_workers()

        for worker in workers:
            worker.cancel.assert_called_once_with()

    def test_finished_worker_is_removed_from_registry(self):
        core = AnnotatorCore()
        worker = mock.MagicMock()
        core._track_worker(worker)

        core._untrack_worker(worker)

        self.assertFalse(any(existing is worker for existing in core._workers))


class SessionGenerationTests(unittest.TestCase):
    def test_old_model_result_cannot_reset_new_session(self):
        core = AnnotatorCore()
        old_worker = mock.MagicMock()
        current_worker = mock.MagicMock()
        core._model_worker = current_worker
        core._session_generation = 2
        core.ocr_handler = "current"
        core._load_item = mock.MagicMock()

        core._on_model_ready(old_worker, 1, "stale")

        self.assertEqual(core.ocr_handler, "current")
        core._load_item.assert_not_called()

    def test_prepare_items_rejects_running_session(self):
        core = AnnotatorCore()
        core.running = True
        core._msg_warning = mock.MagicMock()

        self.assertFalse(core.prepare_items())

        core._msg_warning.assert_called_once()

    def test_cancelled_first_page_load_restores_retryable_state(self):
        core = AnnotatorCore()
        worker = mock.MagicMock()
        core._model_worker = worker
        core._session_generation = 1
        core.running = True
        core._load_item = mock.MagicMock(return_value=False)
        core._refresh_canvas = mock.MagicMock()
        core.set_hotkeys_enabled = mock.MagicMock()
        core.set_status = mock.MagicMock()
        core.on_session_start_aborted = mock.MagicMock()

        core._on_model_ready(worker, 1, mock.MagicMock())

        self.assertFalse(core.running)
        self.assertIsNone(core.ocr_handler)
        core.on_session_start_aborted.assert_called_once_with()
        core._refresh_canvas.assert_not_called()
        self.assertEqual(
            core.set_hotkeys_enabled.call_args_list,
            [mock.call(True), mock.call(False)],
        )

    # Repeated Start is covered by the real Qt interaction tests in
    # test_restart_session.py; the old always-disabled requirement was removed.


if __name__ == "__main__":
    unittest.main()
