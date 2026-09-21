# -*- coding: utf-8 -*-
"""统一退出状态机的纯逻辑回归测试。"""

from pathlib import Path
import unittest
from unittest import mock

import numpy as np

from annotator_core import AnnotatorCore


def _core_with_unsaved_item():
    core = AnnotatorCore()
    core.items = [
        {
            "type": "image",
            "source_path": "/books/page.jpg",
            "display_name": "page.jpg",
            "visited": True,
            "dirty": True,
            "ever_saved": False,
        }
    ]
    core.current_item_index = 0
    core.boxes = [
        np.asarray(
            [[0, 0], [9, 0], [9, 9], [0, 9]],
            dtype=np.float32,
        )
    ]
    core.view_scale = 1.0
    core.set_status = mock.MagicMock()
    return core


class ExitChoiceTests(unittest.TestCase):
    def test_update_exit_saves_without_discard_prompt(self):
        core = _core_with_unsaved_item()
        core._msg_save_discard_cancel = mock.MagicMock(return_value="discard")
        core.save_all = mock.MagicMock(return_value=True)
        core._begin_worker_shutdown = mock.MagicMock()
        self.assertTrue(core.quit_app(save_before_exit=True))
        core._msg_save_discard_cancel.assert_not_called()
        core.save_all.assert_called_once_with()
        core._begin_worker_shutdown.assert_not_called()
        self.assertEqual(core._quit_state, core.QUIT_SAVING)


    def test_cancel_keeps_window_open(self):
        core = _core_with_unsaved_item()
        core._msg_save_discard_cancel = mock.MagicMock(
            return_value="cancel"
        )
        core._begin_worker_shutdown = mock.MagicMock()

        self.assertFalse(core.quit_app())

        self.assertEqual(core._quit_state, core.QUIT_IDLE)
        core._begin_worker_shutdown.assert_not_called()

    def test_discard_starts_safe_worker_shutdown(self):
        core = _core_with_unsaved_item()
        core._msg_save_discard_cancel = mock.MagicMock(
            return_value="discard"
        )
        core._begin_worker_shutdown = mock.MagicMock()
        core.on_quit_interaction_lock_changed = mock.MagicMock()

        self.assertTrue(core.quit_app())

        core._begin_worker_shutdown.assert_called_once_with()
        self.assertTrue(core._quit_interaction_locked)
        core.on_quit_interaction_lock_changed.assert_called_once_with(True)

    def test_save_start_failure_restores_idle_state(self):
        core = _core_with_unsaved_item()
        core._msg_save_discard_cancel = mock.MagicMock(
            return_value="save"
        )
        core.save_all = mock.MagicMock(return_value=False)
        core.on_quit_interaction_lock_changed = mock.MagicMock()

        self.assertFalse(core.quit_app())

        self.assertEqual(core._quit_state, core.QUIT_IDLE)
        self.assertFalse(core._pending_quit)
        self.assertFalse(core._quit_interaction_locked)
        self.assertEqual(
            core.on_quit_interaction_lock_changed.call_args_list,
            [mock.call(True), mock.call(False)],
        )

    def test_clean_session_closes_without_save_prompt(self):
        core = _core_with_unsaved_item()
        core.items[0]["dirty"] = False
        core.items[0]["ever_saved"] = True
        core._msg_save_discard_cancel = mock.MagicMock()
        core._begin_worker_shutdown = mock.MagicMock()

        self.assertTrue(core.quit_app())

        core._msg_save_discard_cancel.assert_not_called()
        core._begin_worker_shutdown.assert_called_once_with()
        self.assertTrue(core._quit_interaction_locked)

    def test_page_navigation_cancel_does_not_continue(self):
        core = _core_with_unsaved_item()
        core._msg_save_discard_cancel = mock.MagicMock(
            return_value="cancel"
        )

        self.assertFalse(core._prompt_save_current_if_needed())

    def test_page_navigation_save_failure_does_not_continue(self):
        core = _core_with_unsaved_item()
        core._msg_save_discard_cancel = mock.MagicMock(
            return_value="save"
        )
        core.save_current = mock.MagicMock(return_value=False)

        self.assertFalse(core._prompt_save_current_if_needed())


class ExitWorkerLifecycleTests(unittest.TestCase):
    def test_shutdown_waits_for_running_worker_without_blocking_wait(self):
        core = AnnotatorCore()
        core.set_status = mock.MagicMock()
        worker = mock.MagicMock()
        worker.isRunning.return_value = True
        worker._quit_watch_connected = False
        core._ocr_worker = worker
        core._really_quit = mock.MagicMock()

        core._begin_worker_shutdown()

        worker.cancel.assert_called_once_with()
        worker.finished.connect.assert_called_once()
        worker.wait.assert_not_called()
        core._really_quit.assert_not_called()

    def test_shutdown_rechecks_worker_after_connect_and_cancel(self):
        core = AnnotatorCore()
        core.set_status = mock.MagicMock()
        worker = mock.MagicMock()
        worker.isRunning.side_effect = [True, False]
        worker._quit_watch_connected = False
        core._ocr_worker = worker
        core._really_quit = mock.MagicMock()

        core._begin_worker_shutdown()

        worker.finished.connect.assert_called_once_with(
            core._maybe_finish_quit
        )
        worker.cancel.assert_called_once_with()
        core._really_quit.assert_called_once_with()

    def test_worker_is_watched_before_it_can_start(self):
        core = AnnotatorCore()
        worker = mock.MagicMock()
        worker._quit_watch_connected = False

        core._track_worker(worker)

        self.assertTrue(worker._quit_watch_connected)
        self.assertIn(
            mock.call(core._maybe_finish_quit),
            worker.finished.connect.call_args_list,
        )

    def test_stale_save_signal_does_not_release_current_worker(self):
        core = AnnotatorCore()
        core.set_status = mock.MagicMock()
        current = mock.MagicMock()
        stale = mock.MagicMock()
        core._save_worker = current
        core._quit_state = core.QUIT_SAVING

        core._on_save_all_stopped(stale, "旧任务已停止")

        self.assertIs(core._save_worker, current)
        self.assertEqual(core._quit_state, core.QUIT_SAVING)
        current.deleteLater.assert_not_called()

    def test_save_worker_is_released_only_after_qthread_finished(self):
        core = AnnotatorCore()
        core.set_status = mock.MagicMock()
        worker = mock.MagicMock()
        core._save_worker = worker

        core._on_save_all_finished(worker)
        self.assertIs(core._save_worker, worker)

        core._on_save_worker_thread_finished(worker)
        self.assertIsNone(core._save_worker)
        worker.deleteLater.assert_called_once_with()

    def test_save_exit_restarts_save_when_old_snapshot_missed_new_edits(self):
        core = _core_with_unsaved_item()
        worker = mock.MagicMock()
        worker._core_save_outcome = "success"
        core._save_worker = worker
        core._quit_state = core.QUIT_SAVING
        core.save_all = mock.MagicMock(return_value=True)
        core._begin_worker_shutdown = mock.MagicMock()

        core._on_save_worker_thread_finished(worker)

        core.save_all.assert_called_once_with()
        core._begin_worker_shutdown.assert_not_called()
        self.assertEqual(core._quit_state, core.QUIT_SAVING)

    def test_save_exit_closes_after_latest_snapshot_is_clean(self):
        core = _core_with_unsaved_item()
        core.items[0]["dirty"] = False
        core.items[0]["ever_saved"] = True
        worker = mock.MagicMock()
        worker._core_save_outcome = "success"
        core._save_worker = worker
        core._quit_state = core.QUIT_SAVING
        core._begin_worker_shutdown = mock.MagicMock()

        core._on_save_worker_thread_finished(worker)

        core._begin_worker_shutdown.assert_called_once_with()

    def test_failed_exit_save_restores_interaction(self):
        core = _core_with_unsaved_item()
        worker = mock.MagicMock()
        worker._core_save_outcome = "error"
        core.main_window = mock.MagicMock()
        core.main_window._update_controller.restart_requested = True
        core._save_worker = worker
        core._quit_state = core.QUIT_SAVING
        core._pending_quit = True
        core._quit_interaction_locked = True
        core.on_quit_interaction_lock_changed = mock.MagicMock()

        core._on_save_worker_thread_finished(worker)

        self.assertEqual(core._quit_state, core.QUIT_IDLE)
        self.assertFalse(core._pending_quit)
        self.assertFalse(core._quit_interaction_locked)
        self.assertFalse(core.main_window._update_controller.restart_requested)
        core.on_quit_interaction_lock_changed.assert_called_once_with(False)

    def test_close_event_contains_no_timed_force_wait(self):
        app_source = (
            Path(__file__).resolve().parents[1] / "app.py"
        ).read_text(encoding="utf-8")
        close_source = app_source.split("def closeEvent", 1)[1].split(
            "def _any_worker_running",
            1,
        )[0]
        self.assertIn("self.core.quit_app(save_before_exit=", close_source)
        self.assertNotIn(".wait(", close_source)

    def test_main_window_disables_central_widget_during_exit(self):
        app_source = (
            Path(__file__).resolve().parents[1] / "app.py"
        ).read_text(encoding="utf-8")

        self.assertIn(
            "self.core.on_quit_interaction_lock_changed",
            app_source,
        )
        self.assertIn("central.setEnabled(not locked)", app_source)


if __name__ == "__main__":
    unittest.main()
