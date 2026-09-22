import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
from unittest.mock import Mock, patch
import numpy as np
from annotation.controller import AnnotatorCore


def dirty_core():
    core = AnnotatorCore()
    core.running = True
    core.items = [{"dirty": True, "boxes": [], "source_path": "old.png"}]
    core._sync_current_boxes_to_item = Mock()
    core.set_status = Mock()
    return core


def test_restart_cancel_preserves_session():
    core = dirty_core()
    core._msg_save_discard_cancel = Mock(return_value="cancel")
    callback = Mock()
    assert not core.request_restart(callback)
    assert core.running and core._restart_callback is None
    assert core.items[0]["dirty"]
    callback.assert_not_called()


def test_restart_detaches_inference_without_waiting_or_closing_window():
    core = dirty_core()
    core._msg_save_discard_cancel = Mock(return_value="discard")
    core.main_window = Mock()
    worker = Mock()
    worker._quit_watch_connected = False
    worker.isRunning.return_value = True
    core._ocr_worker = worker
    callback = Mock()
    queued = []
    with patch("PySide6.QtCore.QTimer.singleShot", side_effect=lambda _, cb: queued.append(cb)):
        assert core.request_restart(callback)
        assert core._quit_state == core.QUIT_STOPPING
        worker.cancel.assert_called_once()
        worker.wait.assert_not_called()
        assert len(queued) == 1
        queued.pop()()
    callback.assert_called_once()
    assert core.any_worker_running()
    assert not core.any_session_worker_running()
    assert worker in core._workers
    assert core._quit_state == core.QUIT_IDLE
    assert core.items == [] and core._ocr_worker is None
    assert not core._allow_window_close and not core._quit_interaction_locked
    core.main_window.close.assert_not_called()


def test_restart_save_failure_cancels_pending_restart():
    core = dirty_core()
    core._msg_save_discard_cancel = Mock(return_value="save")
    core.save_all = Mock(return_value=True)
    callback = Mock()
    assert core.request_restart(callback)
    worker = Mock()
    worker._core_save_outcome = "stopped"
    core._save_worker = worker
    core._on_save_worker_thread_finished(worker)
    assert core._restart_callback is None
    assert core._quit_state == core.QUIT_IDLE and core.running
    callback.assert_not_called()


def test_restart_save_success_continues_only_after_saved_worker_finishes():
    core = dirty_core()
    core._msg_save_discard_cancel = Mock(return_value="save")
    core.save_all = Mock(return_value=True)
    callback = Mock()
    assert core.request_restart(callback)
    worker = Mock()
    worker.isRunning.return_value = False
    worker._core_save_outcome = "success"
    core._save_worker = worker
    core.items[0]["dirty"] = False
    callback.assert_not_called()
    with patch("PySide6.QtCore.QTimer.singleShot", side_effect=lambda _, cb: cb()):
        core._on_save_worker_thread_finished(worker)
    callback.assert_called_once()


def test_ui_keeps_new_paths_pending_until_old_session_has_stopped(tmp_path):
    from PySide6.QtWidgets import QApplication
    from ui.window import MainWindow
    qt = QApplication.instance() or QApplication([])
    window = MainWindow()
    core = window.core
    core.running = True
    core.input_dir = "old-input"
    core.output_dir = "old-output"
    window.in_edit.setText(str(tmp_path))
    window.out_edit.setText(str(tmp_path / "new-output"))
    window.level_combo.setCurrentText("line")
    core.request_restart = Mock(return_value=True)
    core.prepare_items = Mock(return_value=True)
    core.start = Mock(return_value=True)
    window._start()
    assert core.input_dir == "old-input" and core.output_dir == "old-output"
    assert core.level == "char"
    callback = core.request_restart.call_args.args[0]
    core.running = False
    callback()
    assert core.output_dir == str(tmp_path / "new-output") and core.level == "line"
    assert window.start_button.isEnabled()
    window.deleteLater()
    qt.processEvents()


def test_new_geometry_revision_rejects_old_tight_box_context():
    from utils.storage import annotation_context_matches
    core = AnnotatorCore()
    current = core._current_annotation_context()
    previous = {key: value for key, value in current.items() if key != "char_box_revision"}
    assert not annotation_context_matches(previous, current)
    assert annotation_context_matches(current, current)


def test_window_close_still_waits_for_retired_inference():
    core = AnnotatorCore()
    worker = Mock()
    worker.isRunning.return_value = True
    worker._retired_session = True
    core._workers = [worker]
    core.main_window = Mock()
    core.set_status = Mock()
    core.quit_app()
    assert core._quit_state == core.QUIT_STOPPING
    assert not core.can_close_window()
    core.main_window.close.assert_not_called()


def test_old_prefetch_error_cannot_interrupt_new_session():
    core = AnnotatorCore()
    worker = Mock()
    worker._retired_session = True
    core._on_worker_error = Mock()
    core._on_prefetch_error(worker, "old error")
    core._on_worker_error.assert_not_called()
