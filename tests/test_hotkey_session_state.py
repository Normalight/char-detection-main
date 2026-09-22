# -*- coding: utf-8 -*-
"""会话开始前后的快捷键启停回归。"""

import os
import unittest
from unittest import mock

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication, QWidget

from annotation.controller import AnnotatorCore


class HotkeySessionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.window = QWidget()
        self.core = AnnotatorCore(self.window)
        self.core.register_hotkeys(self.window)

    def tearDown(self):
        self.window.deleteLater()

    def test_before_session_only_exit_shortcuts_are_enabled(self):
        enabled = [shortcut.isEnabled() for shortcut in self.core._hotkeys]

        self.assertEqual(sum(enabled), len(self.core._exit_hotkeys))
        self.assertGreater(len(self.core._exit_hotkeys), 0)

    def test_model_ready_enables_annotation_shortcuts(self):
        worker = mock.MagicMock()
        self.core._model_worker = worker
        self.core._session_generation = 1
        self.core._load_item = mock.MagicMock(return_value=True)
        self.core._refresh_canvas = mock.MagicMock()

        self.core._on_model_ready(worker, 1, mock.MagicMock())

        self.assertTrue(all(sc.isEnabled() for sc in self.core._hotkeys))

    def test_final_close_disables_every_shortcut(self):
        self.core.set_hotkeys_enabled(True)
        self.core._known_workers = mock.MagicMock(return_value=[])
        self.window.close = mock.MagicMock()

        self.core._really_quit()

        self.assertTrue(all(not sc.isEnabled() for sc in self.core._hotkeys))

    def test_before_session_quit_has_no_save_prompt(self):
        self.core._msg_save_discard_cancel = mock.MagicMock()
        self.core._begin_worker_shutdown = mock.MagicMock()

        self.assertTrue(self.core.quit_app())

        self.core._msg_save_discard_cancel.assert_not_called()
        self.core._begin_worker_shutdown.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
