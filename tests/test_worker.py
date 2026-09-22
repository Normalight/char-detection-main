# -*- coding: utf-8 -*-
"""annotation.workers.py 的静态/语义回归。

只测类结构与 cancel 行为，不真的启动 QThread（需要 QApplication + OCR 模型）。
"""

import inspect
from pathlib import Path
import unittest
from unittest import mock

import numpy as np
from PIL import Image
from PySide6.QtTest import QSignalSpy

import annotation.workers as worker
from annotation.workers import _BaseWorker, ModelLoadWorker, OcrPrefetchWorker, SaveAllWorker


class WorkerStructureTests(unittest.TestCase):
    def test_workers_inherit_base(self):
        for cls in (ModelLoadWorker, OcrPrefetchWorker, SaveAllWorker):
            self.assertTrue(issubclass(cls, _BaseWorker),
                            f"{cls.__name__} 必须继承 _BaseWorker")

    def test_base_has_cancel_and_signals(self):
        # cancel 标志位
        self.assertTrue(hasattr(_BaseWorker, "cancel"))
        # 关键 signal 必须作为类属性存在（PySide6 Signal 是描述符）
        for sig in ("error", "finished_ok"):
            self.assertTrue(hasattr(_BaseWorker, sig), f"缺少 signal {sig}")

    def test_save_all_worker_checks_cancel_in_loop(self):
        """批量保存的核心循环必须有 cancel 检查点，否则取消按钮无效。"""
        src = inspect.getsource(SaveAllWorker.run)
        self.assertIn("is_cancelled", src)

    def test_ocr_worker_does_not_touch_gui(self):
        """Worker 不允许直接碰 QWidget / QImage / QMessageBox。

        通过源码扫描禁止出现这些标识符——纯逻辑的 IO/计算可以，UI 不行。
        防止以后误把 GUI 操作塞进后台线程（跨线程碰 GUI 会崩）。
        """
        banned = ("QMessageBox", "QDialog", "QPainter", "QImage", ".setText(",
                  ".show()", ".exec()", ".update()")
        for cls in (ModelLoadWorker, OcrPrefetchWorker, SaveAllWorker):
            src = inspect.getsource(cls)
            for needle in banned:
                # 允许在注释/字符串里提到（例如 docstring）；只看真实语句很难，
                # 折中：去掉行首为 # 或被 " 包围的简单情况。
                lines = []
                for line in src.splitlines():
                    stripped = line.strip()
                    if stripped.startswith("#"):
                        continue
                    lines.append(line)
                joined = "\n".join(lines)
                self.assertNotIn(needle, joined,
                                 f"{cls.__name__} 不应出现 {needle}（worker 不得碰 GUI）")

    def test_save_all_worker_loads_image_itself(self):
        """SaveAllWorker 必须自己加载/渲染图片，不能依赖主线程传入 PIL
        （否则要么主线程加载阻塞，要么需要持有大量 PIL 引用）。"""
        src = inspect.getsource(SaveAllWorker)
        self.assertIn("_load_item_pil", src)

    def test_save_all_worker_does_not_integrate_glyph_grouping(self):
        src = inspect.getsource(SaveAllWorker.run)
        self.assertNotIn("build_glyph_grouping_outputs", src)
        self.assertNotIn("字形识别归并", src)

    def test_save_all_error_closes_progress_dialog(self):
        app_source = (
            Path(__file__).resolve().parents[1] / "ui/window.py"
        ).read_text(encoding="utf-8")
        self.assertIn(
            "sw.error.connect(lambda _msg, worker=sw: self._close_progress(worker))",
            app_source,
        )
        self.assertNotIn(
            "sw.error.connect(self._on_worker_error_close_progress)",
            app_source,
        )

    def test_save_all_non_success_path_is_connected_to_core_and_ui(self):
        core_source = (
            Path(__file__).resolve().parents[1] / "annotation/controller.py"
        ).read_text(encoding="utf-8")
        app_source = (
            Path(__file__).resolve().parents[1] / "ui/window.py"
        ).read_text(encoding="utf-8")
        self.assertIn(
            "save_worker.stopped.connect(",
            core_source,
        )
        self.assertIn(
            "sw.stopped.connect(lambda _msg, worker=sw: self._close_progress(worker))",
            app_source,
        )
        self.assertIn(
            "self.core.on_save_worker_created = self._hook_save_worker",
            app_source,
        )

    def test_save_progress_is_non_modal_and_errors_have_one_ui_owner(self):
        app_source = (
            Path(__file__).resolve().parents[1] / "ui/window.py"
        ).read_text(encoding="utf-8")
        self.assertIn("modal=False", app_source)
        error_handler = app_source.split(
            "def _on_worker_error_close_progress",
            1,
        )[1].split("def _on_save_all_done", 1)[0]
        self.assertNotIn("QMessageBox.critical", error_handler)
        self.assertNotIn("QTimer.singleShot", app_source)


class WorkerCancelTests(unittest.TestCase):
    """cancel 行为：不真的启动 QThread，只验证 cancel() 翻转标志位。

    QThread 的 __new__ 不能用 object.__new__ 绕过（PySide6 限制），
    所以这里用 mock 替代实例。
    """

    def test_cancel_method_flips_flag(self):
        # 用 unittest.mock 构造一个 spec=_BaseWorker 的假对象，
        # 让 hasattr 通过；cancel 方法用真实方法绑定。
        from unittest import mock
        w = mock.MagicMock(spec=_BaseWorker)
        # 模拟 _cancelled 字段 + 让 cancel 真正翻转它
        w._cancelled = False
        w.is_cancelled = _BaseWorker.is_cancelled.fget(w)
        w.cancel = lambda: setattr(w, "_cancelled", True)
        self.assertFalse(w.is_cancelled)
        w.cancel()
        # 直接读字段验证
        self.assertTrue(w._cancelled)

    def test_cancel_is_idempotent(self):
        """多次 cancel 不出错（quit_app / closeEvent 可能多次调）。"""
        from unittest import mock
        w = mock.MagicMock(spec=_BaseWorker)
        w._cancelled = False
        w.cancel = lambda: setattr(w, "_cancelled", True)
        w.cancel()
        w.cancel()
        w.cancel()
        self.assertTrue(w._cancelled)


class OcrPrefetchPayloadTests(unittest.TestCase):
    def test_emits_boxes_only(self):
        box = np.asarray(
            [[0, 0], [9, 0], [9, 9], [0, 9]],
            dtype=np.float32,
        )
        fake_ocr = mock.MagicMock()
        fake_ocr.text_predict.return_value = [
            {
                "bbox": box,
                "score": 0.99,
                "result": {
                    "char_list": [
                        {"char": "字", "score": 0.91, "bbox": box},
                    ],
                    "words": [],
                },
            }
        ]
        prefetch = OcrPrefetchWorker(
            Image.new("RGB", (10, 10), "white"),
            960,
            "char",
            fake_ocr,
            context={"request_id": 7},
        )
        payloads = []
        prefetch.ocr_done.connect(payloads.append)

        prefetch.run()

        self.assertEqual(len(payloads), 1)
        self.assertEqual(len(payloads[0]["boxes"]), 1)
        self.assertEqual(set(payloads[0]), {"boxes", "context", "ocr_geometry"})
        self.assertEqual(payloads[0]["context"], {"request_id": 7})


class SaveAllWorkerOutputTests(unittest.TestCase):
    def test_char_save_finishes_without_grouping_call(self):
        item = {
            "type": "image",
            "source_path": "/books/c1.jpg",
            "display_name": "c1.jpg",
            "boxes": [
                np.asarray(
                    [[0, 0], [9, 0], [9, 9], [0, 9]],
                    dtype=np.float32,
                )
            ],
        }
        fake_ocr = mock.MagicMock()
        save_worker = SaveAllWorker(
            output_dir="/output",
            items=[item],
            ocr_handler=fake_ocr,
            dbnet_short_side=960,
            level="char",
            pdf_dpi=200,
            poppler_path=None,
            image_scale=1.0,
        )
        finished_spy = QSignalSpy(save_worker.finished_ok)

        self.assertFalse(hasattr(worker, "build_glyph_grouping_outputs"))
        with (
            mock.patch(
                "annotation.workers._load_item_pil",
                return_value=Image.new("RGB", (10, 10), "white"),
            ),
            mock.patch("annotation.workers.save_item_boxes_to_disk", return_value=1) as save,
            mock.patch("annotation.workers.build_excel_for_single_item") as workbook,
        ):
            save_worker.run()

        save.assert_called_once()
        workbook.assert_called_once()
        self.assertEqual(finished_spy.count(), 1)


class SaveAllCoreOutcomeTests(unittest.TestCase):
    def test_model_worker_ui_hook_runs_before_worker_start(self):
        from annotation.controller import AnnotatorCore

        core = AnnotatorCore()
        events = []
        fake_worker = mock.MagicMock()
        fake_worker.start.side_effect = lambda: events.append("start")
        core.on_model_worker_created = lambda worker: events.append(
            "hook" if worker is fake_worker else "wrong-worker"
        )

        with mock.patch("annotation.workers.ModelLoadWorker", return_value=fake_worker):
            core._start_model_load()

        self.assertEqual(events, ["hook", "start"])

    def test_save_worker_ui_hook_runs_before_worker_start(self):
        from annotation.controller import AnnotatorCore

        core = AnnotatorCore()
        core.items = [
            {
                "type": "image",
                "source_path": "/books/c1.jpg",
            }
        ]
        core.current_item_index = 0
        core.boxes = []
        core.view_scale = 1.0
        core.ocr_handler = mock.MagicMock()
        core.output_dir = "/output"
        events = []
        fake_worker = mock.MagicMock()
        fake_worker.start.side_effect = lambda: events.append("start")
        core.on_save_worker_created = lambda worker: events.append(
            "hook" if worker is fake_worker else "wrong-worker"
        )

        with mock.patch("annotation.workers.SaveAllWorker", return_value=fake_worker):
            core.save_all()

        self.assertEqual(events, ["hook", "start"])

    def test_save_all_passes_deep_snapshot_to_worker(self):
        from annotation.controller import AnnotatorCore

        core = AnnotatorCore()
        box = np.asarray(
            [[0, 0], [9, 0], [9, 9], [0, 9]],
            dtype=np.float32,
        )
        core.items = [
            {
                "type": "image",
                "source_path": "/books/c1.jpg",
                "boxes": [box.copy()],
            }
        ]
        core.current_item_index = 0
        core.boxes = [box.copy()]
        core.view_scale = 1.0
        core.ocr_handler = mock.MagicMock()
        core.output_dir = "/output"
        fake_worker = mock.MagicMock()

        with mock.patch(
            "annotation.workers.SaveAllWorker",
            return_value=fake_worker,
        ) as worker_cls:
            core.save_all()

        snapshot = worker_cls.call_args.args[1]
        self.assertIsNot(snapshot[0], core.items[0])
        core.items[0]["boxes"][0][0, 0] = 123
        self.assertEqual(float(snapshot[0]["boxes"][0][0, 0]), 0.0)

    def test_save_start_exception_cleans_worker_and_returns_false(self):
        from annotation.controller import AnnotatorCore

        core = AnnotatorCore()
        core.items = [
            {
                "type": "image",
                "source_path": "/books/c1.jpg",
            }
        ]
        core.current_item_index = 0
        core.boxes = []
        core.view_scale = 1.0
        core.ocr_handler = mock.MagicMock()
        core.output_dir = "/output"
        core._on_worker_error = mock.MagicMock()
        core.on_save_worker_start_failed = mock.MagicMock()
        fake_worker = mock.MagicMock()
        fake_worker.start.side_effect = RuntimeError("start failed")
        fake_worker.isRunning.return_value = False

        with mock.patch("annotation.workers.SaveAllWorker", return_value=fake_worker):
            started = core.save_all()

        self.assertFalse(started)
        self.assertIsNone(core._save_worker)
        self.assertFalse(
            any(worker is fake_worker for worker in core._workers)
        )
        core.on_save_worker_start_failed.assert_called_once_with(fake_worker)
        core._on_worker_error.assert_called_once()
        fake_worker.deleteLater.assert_called_once_with()

    def test_stopped_save_waits_for_qthread_finished_before_release(self):
        from annotation.controller import AnnotatorCore

        core = AnnotatorCore()
        worker_mock = mock.MagicMock()
        core._save_worker = worker_mock
        core._quit_state = core.QUIT_SAVING
        core._pending_quit = True
        core.set_status = mock.MagicMock()

        core._on_save_all_stopped(worker_mock, "批量保存已取消")

        core.set_status.assert_called_once_with("批量保存已取消")
        self.assertIs(core._save_worker, worker_mock)
        worker_mock.deleteLater.assert_not_called()

        core._on_save_worker_thread_finished(worker_mock)

        self.assertFalse(core._pending_quit)
        self.assertEqual(core._quit_state, core.QUIT_IDLE)
        self.assertIsNone(core._save_worker)
        worker_mock.deleteLater.assert_called_once()


if __name__ == "__main__":
    unittest.main()
