"""Native updater must respect the real annotator close state machine."""
from PySide6.QtWidgets import QApplication
from PySide6.QtTest import QTest
from hanzi_update import attach_updates
from ui.window import MainWindow


def test_native_install_waits_for_host_close_policy(tmp_path, monkeypatch):
    monkeypatch.setattr('hanzi_update.native_ui.QStandardPaths.writableLocation', lambda _: str(tmp_path))
    app = QApplication.instance() or QApplication([])
    window = MainWindow()
    updater = attach_updates(window, 'char-detection-main')
    updater.startup_timer.stop()
    window.show()
    try:
        updater.prepare_close()
        for _ in range(20):
            QTest.qWait(10)
            if updater.approved:
                break
        assert updater.approved
        assert window.isVisible()
        assert window.core.can_close_window()
        updater.finish_close()
        assert not window.isVisible()
    finally:
        updater.shutdown()
        window.removeEventFilter(updater)
        window.close()
