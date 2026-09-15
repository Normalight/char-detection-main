# -*- coding: utf-8 -*-
"""
annotator_core.py

当前主线 GUI 交互核心（PySide6 版）。

UI 接入层从 Tk + cv2.imshow 迁到 Qt：
- 画布：AnnotatorCanvas（QPainter 绘制，鼠标/滚轮走 Qt 事件）
- 快捷键：QShortcut（直接吃 WM_KEYDOWN，IME 激活时也照常触发）
- 主循环：Qt 事件循环，不再 while + cv2.waitKeyEx + root.update 双循环
- 对话框：QMessageBox（模态，事件循环天然支持）

业务核心（坐标系、boxes、保存、Excel、OCR）保持不变。

本版核心修复（继承自历史版本）：
1. 将"显示缩放 view_scale"和"真实处理/保存坐标"彻底解耦；
2. boxes 始终存储在 base_image_bgr 坐标系，不再随放大、缩小、适配窗口而改变；
3. 保存切片始终从 base_image_bgr 裁剪，避免 Excel/切片质量受界面缩放影响；
4. Excel 生成失败不再静默吞掉，会打印并显示状态；
5. 批量保存对未访问页面具备 JSON/OCR 兜底；
6. 保持 app.py 原有工具栏调用接口不变。
"""

import copy
import logging
import math
import os
from glob import glob
from typing import List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

from cutbatch import OcrHandle
from pdf_utils import get_pdf_page_count, render_pdf_page_to_pil
from storage_utils import (
    annotation_context_matches,
    load_item_annotation_record_if_exists,
    normalize_boxes,
    normalize_annotation_context,
)


class AnnotatorCore:
    APP_WINDOW_TITLE = "Interactive OCR Annotator - 控制面板"
    WINDOW_NAME = "Interactive OCR Annotator (Batch Mode)"
    HANDLE_RADIUS = 8

    MIN_VIEW_SCALE = 0.05
    MAX_VIEW_SCALE = 12.0

    QUIT_IDLE = "idle"
    QUIT_SAVING = "saving"
    QUIT_STOPPING = "stopping"
    QUIT_CLOSING = "closing"

    def __init__(self, main_window=None):
        # main_window：QMainWindow（或任意 QWidget），用作 QShortcut 宿主和消息框父窗口。
        # 可为 None，便于纯逻辑单元测试通过 object.__new__ 绕过 __init__。
        self.main_window = main_window
        self.input_dir = ""
        self.output_dir = ""
        self.level = "char"
        self.dbnet_short_side = 960
        self.pdf_dpi = 200
        self.image_scale = 1.0
        self.poppler_path = None

        self.items = []
        self.current_item_index = 0
        self.total_items = 0

        # base_image_bgr：真实处理图像，坐标系用于 OCR、编辑框、切片保存。
        # view_image_bgr：仅用于显示，由 base_image_bgr 按 view_scale 缩放得到。
        self.base_image_bgr = None
        self.view_image_bgr = None
        self.base_size = (0, 0)  # (w, h)
        self.view_scale = 1.0
        # view_image_bgr 版本号：每次 _rebuild_view_image 自增，canvas 用它做 QImage 缓存指纹。
        self._view_version = 0

        self.VIEW_W = 800
        self.VIEW_H = 600
        self.viewport_offset = np.array([0, 0], dtype=np.int32)  # view image 坐标系
        self.draw_origin = np.array([0, 0], dtype=np.int32)      # view image 小于窗口时的居中偏移

        # boxes 始终在 base_image_bgr 坐标系。
        self.boxes: List[np.ndarray] = []
        self.selected_box_index = 0
        self.selected_handle_index = -1
        self.drawing = False

        self.creation_mode = False
        self.creation_start_point = None

        self.rotate_mode = False
        self.rotating = False
        self.rotate_center = None
        self.rotate_start_vec = None
        self.rotate_base_box = None

        self.nav_mode = False
        self.panning = False
        self.last_mouse = (0, 0)

        self.current_item_title = ""
        self.status_label = None

        self.ocr_handler = None
        self.running = False

        self.canvas = None
        self._hotkeys_registered = False
        self._hotkeys = []
        self._exit_hotkeys = []

        # 后台 worker 引用（持有以免被 GC；同一时刻每类最多一个）
        self._model_worker = None
        self._ocr_worker = None
        self._save_worker = None
        # 空闲预取 worker：后台顺序 OCR 未访问页，切页时结果已在内存
        self._prefetch_worker = None
        self._workers = []
        self._load_worker = None
        self._navigation_target = None
        self._session_generation = 0
        self._ocr_target_index = -1
        self._next_ocr_request_id = 1
        self._quit_state = self.QUIT_IDLE
        self._allow_window_close = False
        # 兼容已有调用和测试；新的退出流程以 _quit_state 为唯一状态源。
        self._pending_quit = False
        # app.py 可注册回调；每次 SaveAllWorker 构造后、start 前同步调用，
        # 以确保进度/取消信号不会因轮询时序而漏接。
        self.on_model_worker_created = None
        self.on_save_worker_created = None
        self.on_save_worker_start_failed = None
        self.on_session_start_aborted = None
        self._restart_callback = None
        self.on_quit_interaction_lock_changed = None
        self._quit_interaction_locked = False

    # ---------- 外部接口 ----------
    def set_status_label(self, lbl):
        self.status_label = lbl

    def set_status(self, txt):
        logger.info(txt)
        if self.status_label:
            try:
                # Qt：QStatusBar.showMessage 或 QLabel.setText 都行，统一走 setText 兼容。
                if hasattr(self.status_label, "setText"):
                    self.status_label.setText(txt)
                elif hasattr(self.status_label, "showMessage"):
                    self.status_label.showMessage(txt)
            except Exception as e:
                logger.debug("更新状态标签失败：%s", e)

    def set_input_dir(self, p):
        self.input_dir = p
        self.set_status(f"输入: {p}")

    def set_output_dir(self, p):
        self.output_dir = p
        os.makedirs(p, exist_ok=True)
        self.set_status(f"输出: {p}")

    def set_params(self, level, short_side, dpi, img_scale, poppler):
        level = (level or "char").strip().lower()
        if level not in ("char", "word", "line"):
            raise ValueError("Level 只能是 char、word 或 line")
        self.level = level
        self.dbnet_short_side = int(short_side)
        self.pdf_dpi = int(dpi)
        self.image_scale = float(img_scale)
        self.poppler_path = poppler if poppler else None

    def _current_annotation_context(self):
        import config as app_config

        return normalize_annotation_context(
            {
                "level": self.level,
                "char_box_revision": "8",
                "dbnet_short_side": self.dbnet_short_side,
                "pdf_dpi": self.pdf_dpi,
                "image_scale": self.image_scale,
                "ocr_max_side": getattr(app_config, "ocr_max_side", 2560),
                "det_model": getattr(app_config, "det_model", "legacy"),
                "rec_model": getattr(app_config, "rec_model", "crnn"),
                "char_box_strategy": getattr(
                    app_config,
                    "char_box_strategy",
                    "ctc-midpoint",
                ),
            }
        )

    def attach_canvas(self, canvas):
        """注册 Qt 画布，并同步初始尺寸。"""
        self.canvas = canvas
        self.VIEW_W = max(1, canvas.width())
        self.VIEW_H = max(1, canvas.height())

    def register_hotkeys(self, main_window=None):
        """用 QShortcut 注册快捷键。

        关键：QShortcut 监听的是 QKeyEvent.key()（= Win32 WM_KEYDOWN 的 vkCode），
        不经 TranslateMessage / IME，所以中文输入法激活时也能触发。

        main_window 不传则用 __init__ 时传入的 self.main_window。
        """
        if main_window is None:
            main_window = self.main_window
        if main_window is None:
            logger.warning("register_hotkeys 跳过：无 main_window 可绑定")
            return
        if self._hotkeys_registered:
            return
        self._hotkeys_registered = True

        from PySide6.QtCore import Qt
        from PySide6.QtGui import QKeySequence, QShortcut

        # 大小写各注册一次（QKeySequence("C") 严格匹配 Shift 状态）。
        # ApplicationShortcut：焦点在任何子 widget 都生效。
        bindings = [
            (("C", "Shift+C"),       self.toggle_create_mode),
            (("R", "Shift+R"),       self.toggle_rotate_mode),
            (("O", "Shift+O"),       self.toggle_nav_mode),
            (("+", "="),             lambda: self.zoom_center(1.25)),
            (("-", "_"),             lambda: self.zoom_center(0.8)),
            (("N", "Shift+N"),       self.next_box),
            (("B", "Shift+B"),       self.prev_box),
            (("S", "Shift+S"),       self.save_current),
            (("A", "Shift+A"),       self.prev_item),
            (("D", "Shift+D"),       self.next_item),
            (("X", "Shift+X", "Delete"), self.delete_box),
            (("Q", "Shift+Q", "Escape"), self.quit_app),
        ]
        for keys, cb in bindings:
            for k in keys:
                sc = QShortcut(QKeySequence(k), main_window)
                sc.setContext(Qt.ApplicationShortcut)
                sc.activated.connect(cb)
                # 会话开始前只保留退出键；其它标注快捷键要等模型就绪。
                is_exit_hotkey = cb == self.quit_app
                sc.setEnabled(is_exit_hotkey)
                self._hotkeys.append(sc)
                if is_exit_hotkey:
                    self._exit_hotkeys.append(sc)

    def set_hotkeys_enabled(self, enabled):
        for shortcut in getattr(self, "_hotkeys", []):
            try:
                is_exit_hotkey = any(
                    shortcut is exit_shortcut
                    for exit_shortcut in getattr(self, "_exit_hotkeys", [])
                )
                shortcut.setEnabled(bool(enabled) or is_exit_hotkey)
            except Exception as e:
                logger.debug("切换快捷键状态失败：%s", e)

    def disable_all_hotkeys(self):
        for shortcut in getattr(self, "_hotkeys", []):
            try:
                shortcut.setEnabled(False)
            except Exception as e:
                logger.debug("禁用快捷键失败：%s", e)

    def _set_quit_interaction_locked(self, locked):
        locked = bool(locked)
        if self._quit_interaction_locked == locked:
            return
        self._quit_interaction_locked = locked
        if locked:
            self.disable_all_hotkeys()
        else:
            self.set_hotkeys_enabled(self.ocr_handler is not None and self.running)
        callback = getattr(self, "on_quit_interaction_lock_changed", None)
        if callable(callback):
            try:
                callback(locked)
            except Exception as e:
                logger.warning("切换退出交互锁失败：%s", e)

    def prepare_items(self):
        """收集输入项；不加载模型（模型加载异步走 ModelLoadWorker）。"""
        if self.running or self.any_session_worker_running():
            self._msg_warning("提示", "当前会话或后台任务仍在运行，不能重复开始")
            return False
        if not os.path.isdir(self.input_dir):
            self._msg_critical("错误", "输入目录无效")
            return False
        self.items = self._collect_input_items()
        if not self.items:
            self._msg_warning("提示", "输入目录中没有符合的图片/PDF")
            return False
        annotation_context = self._current_annotation_context()
        for item in self.items:
            item["annotation_context"] = copy.deepcopy(annotation_context)
            item["box_context"] = None
        self.total_items = len(self.items)
        os.makedirs(self.output_dir, exist_ok=True)
        self.set_status(f"找到 {self.total_items} 个项目，正在初始化 OCR 模型...")
        return True

    def start(self):
        """启动标注会话。

        流程（全部异步，主线程不阻塞）：
        1) ModelLoadWorker 加载 ONNX（~2-5s）→ on_model_ready
        2) on_model_ready 切到第一页 _load_item（已访问/有 JSON 立即显示）
        3) 若该页没 boxes，启动 OcrPrefetchWorker 跑预标注 → on_ocr_done 合并
        """
        if not self.items or self.running or self.any_session_worker_running():
            return False
        self._session_generation += 1
        self.running = True
        try:
            self._start_model_load()
        except Exception as e:
            self.running = False
            worker = self._model_worker
            self._model_worker = None
            self._untrack_worker(worker)
            try:
                if worker is not None:
                    worker.deleteLater()
            except Exception:
                pass
            logger.exception("启动模型任务失败：%s", e)
            self._msg_critical("启动失败", str(e))
            return False
        return True

    # ---------- 异步任务编排 ----------
    def _track_worker(self, worker):
        if worker is None or any(existing is worker for existing in self._workers):
            return worker
        self._workers.append(worker)
        worker.finished.connect(
            lambda tracked=worker: self._untrack_worker(tracked)
        )
        # worker 启动前就接好退出等待监听，避免退出流程中临时连接时
        # 恰好错过 finished 信号。
        self._watch_worker_for_quit(worker)
        return worker

    def _untrack_worker(self, worker):
        self._workers = [
            existing for existing in self._workers if existing is not worker
        ]

    def active_workers(self):
        return [
            worker
            for worker in self._known_workers()
            if self._worker_is_running(worker)
        ]

    def any_worker_running(self):
        return bool(self.active_workers())

    def any_session_worker_running(self):
        return any(getattr(worker, "_retired_session", False) is not True
                   for worker in self.active_workers())

    def cancel_all_workers(self):
        for worker in self.active_workers():
            try:
                worker.cancel()
            except Exception as e:
                logger.debug("请求取消 worker 失败：%s", e)

    def _start_model_load(self):
        from worker import ModelLoadWorker
        generation = self._session_generation
        model_worker = ModelLoadWorker(parent=self.main_window)
        self._model_worker = model_worker
        model_worker.ocr_ready.connect(
            lambda ocr_handle, worker=model_worker, token=generation: self._on_model_ready(
                worker,
                token,
                ocr_handle,
            )
        )
        model_worker.error.connect(
            lambda message, worker=model_worker, token=generation: self._on_model_error(
                worker,
                token,
                message,
            )
        )
        model_worker.finished.connect(
            lambda worker=model_worker: self._on_model_worker_thread_finished(
                worker
            )
        )
        self._track_worker(model_worker)
        worker_created_hook = getattr(
            self,
            "on_model_worker_created",
            None,
        )
        if callable(worker_created_hook):
            try:
                worker_created_hook(model_worker)
            except Exception as e:
                logger.warning("模型 worker UI 挂接失败：%s", e)
        model_worker.start()

    def _on_model_ready(self, worker, generation, ocr_handle):
        if (
            worker is not self._model_worker
            or generation != self._session_generation
            or self._quit_state != self.QUIT_IDLE
        ):
            logger.debug("忽略旧模型任务结果：generation=%s", generation)
            return
        self.ocr_handler = ocr_handle
        self.ocr_handler.ocr_image_scale = self.image_scale
        self.set_hotkeys_enabled(True)
        self.set_status("OCR 模型就绪，正在加载第一页...")
        self.current_item_index = 0
        if not self._load_item():
            self.running = False
            self.ocr_handler = None
            self.set_hotkeys_enabled(False)
            callback = getattr(self, "on_session_start_aborted", None)
            if callable(callback):
                try:
                    callback()
                except Exception as e:
                    logger.warning("恢复首次加载后的启动入口失败：%s", e)
            self.set_status("首次页面未加载，会话已撤回，可修改设置后重新开始")
            return
        self._refresh_canvas()

    def _on_model_error(self, worker, generation, message):
        if (
            worker is not self._model_worker
            or generation != self._session_generation
        ):
            return
        self.running = False
        self._on_worker_error(message)

    def _on_model_worker_thread_finished(self, worker):
        if worker is self._model_worker:
            self._model_worker = None
        try:
            worker.deleteLater()
        except Exception:
            pass

    def _on_worker_error(self, msg):
        self.set_status(f"后台任务失败：{msg}")
        self._msg_critical("错误", msg)

    def _refresh_canvas(self):
        """触发画布重绘（canvas.update()，Qt 下一帧调 paintEvent）。

        getattr 兜底：兼容 object.__new__ 绕过 __init__ 的纯逻辑测试
        （这类测试不关心 Qt 画布）。
        """
        if getattr(self, "canvas", None) is not None:
            try:
                self.canvas.update()
            except Exception as e:
                logger.debug("canvas.update 异常：%s", e)

    def _update_window_title(self):
        """持续在主窗口标题显示当前项、页码和界面缩放。"""
        main_window = getattr(self, "main_window", None)
        if main_window is None or not hasattr(main_window, "setWindowTitle"):
            return
        current_title = getattr(self, "current_item_title", "")
        total = int(getattr(self, "total_items", 0) or 0)
        if current_title and total > 0:
            current = int(getattr(self, "current_item_index", 0)) + 1
            scale = float(getattr(self, "view_scale", 1.0))
            title = (
                f"{self.APP_WINDOW_TITLE} - ({current}/{total}) "
                f"{current_title} [View:{scale:.2f}]"
            )
        else:
            title = self.APP_WINDOW_TITLE
        try:
            main_window.setWindowTitle(title)
        except Exception as e:
            logger.debug("更新窗口标题失败：%s", e)

    # ---------- Qt 对话框封装 ----------
    def _msg_critical(self, title, text):
        self._msg_box("critical", title, text)

    def _msg_warning(self, title, text):
        self._msg_box("warning", title, text)

    def _msg_question(self, title, text) -> bool:
        """返回 True 表示用户选 Yes。"""
        return self._msg_box("question", title, text)

    def _msg_save_discard_cancel(
        self,
        title,
        text,
        save_label="保存并继续",
        discard_label="不保存",
    ) -> str:
        """显示保存 / 放弃 / 取消三选项，关闭对话框等同取消。"""
        if self.main_window is None:
            logger.warning("[choice] %s: %s", title, text)
            return "cancel"
        try:
            from PySide6.QtWidgets import QMessageBox
        except Exception as e:
            logger.error("QMessageBox 导入失败：%s；消息 %s: %s", e, title, text)
            return "cancel"

        box = QMessageBox(self.main_window)
        box.setIcon(QMessageBox.Question)
        box.setWindowTitle(title)
        box.setText(text)
        save_button = box.addButton(save_label, QMessageBox.AcceptRole)
        discard_button = box.addButton(discard_label, QMessageBox.DestructiveRole)
        cancel_button = box.addButton("取消", QMessageBox.RejectRole)
        box.setDefaultButton(save_button)
        box.setEscapeButton(cancel_button)
        box.exec()
        clicked = box.clickedButton()
        if clicked is save_button:
            return "save"
        if clicked is discard_button:
            return "discard"
        return "cancel"

    def _msg_box(self, kind, title, text):
        """惰性导入 QMessageBox，避免顶层强依赖 PySide6（保留纯逻辑可测性）。

        若 Qt 未就绪（main_window 为 None 或 import 失败），退化到 stderr 提示。
        """
        if self.main_window is None:
            logger.warning("[%s] %s: %s", kind, title, text)
            return kind != "question" or False
        try:
            from PySide6.QtWidgets import QMessageBox
        except Exception as e:
            logger.error("QMessageBox 导入失败：%s；消息 [%s] %s: %s", e, kind, title, text)
            return kind != "question" or False

        icon = {
            "critical": QMessageBox.Critical,
            "warning": QMessageBox.Warning,
            "question": QMessageBox.Question,
        }.get(kind, QMessageBox.Information)
        if kind == "question":
            box = QMessageBox(self.main_window)
            box.setIcon(icon)
            box.setWindowTitle(title)
            box.setText(text)
            yes = box.addButton("是", QMessageBox.YesRole)
            box.addButton("否", QMessageBox.NoRole)
            box.exec()
            return box.clickedButton() is yes
        msg = QMessageBox(self.main_window)
        msg.setIcon(icon)
        msg.setWindowTitle(title)
        msg.setText(text)
        msg.exec()
        return True

    # ---------- 工具栏/快捷键动作 ----------
    def toggle_create_mode(self):
        self.creation_mode = not self.creation_mode
        if self.creation_mode:
            self.rotate_mode = False
            self.nav_mode = False
        self.set_status(f"创建模式: {'开' if self.creation_mode else '关'}")
        self._refresh_canvas()

    def toggle_rotate_mode(self):
        self.rotate_mode = not self.rotate_mode
        if self.rotate_mode:
            self.creation_mode = False
            self.nav_mode = False
        self.set_status(f"旋转模式: {'开' if self.rotate_mode else '关'}")
        self._refresh_canvas()

    def toggle_nav_mode(self):
        self.nav_mode = not self.nav_mode
        if self.nav_mode:
            self.creation_mode = False
            self.rotate_mode = False
        self.set_status(f"导航模式: {'开(左键拖动, 滚轮缩放)' if self.nav_mode else '关'}")
        self._refresh_canvas()

    def zoom_center(self, factor):
        self._zoom_at_screen(factor, (self.VIEW_W // 2, self.VIEW_H // 2))

    def fit_to_window(self):
        if self.base_image_bgr is None:
            return
        bw, bh = self.base_size
        if bw <= 0 or bh <= 0:
            return
        new_scale = min(self.VIEW_W / bw, self.VIEW_H / bh, 1.0)
        self._set_view_scale(float(np.clip(new_scale, self.MIN_VIEW_SCALE, self.MAX_VIEW_SCALE)), center=True)

    def one_to_one(self):
        self._set_view_scale(1.0, center=True)

    def prev_box(self):
        if self.boxes:
            self.selected_box_index = (self.selected_box_index - 1 + len(self.boxes)) % len(self.boxes)
            self._refresh_canvas()

    def next_box(self):
        if self.boxes:
            self.selected_box_index = (self.selected_box_index + 1) % len(self.boxes)
            self._refresh_canvas()

    def delete_box(self):
        if self.boxes:
            self.boxes.pop(self.selected_box_index)
            if self.selected_box_index >= len(self.boxes):
                self.selected_box_index = max(0, len(self.boxes) - 1)
            self._mark_dirty()
            self.set_status("已删除当前框")
            self._refresh_canvas()

    def save_current(self):
        """Start a snapshot save; completion, not this return value, allows paging."""
        if self.base_image_bgr is None or not self.items:
            return False
        if self._worker_is_running(self._save_worker) or self._load_worker is not None:
            self.set_status('正在保存或加载页面，请稍候')
            return False
        item = self.items[self.current_item_index]
        if item.get('ocr_pending', False) and not item.get('dirty', False):
            self._msg_warning('识别尚未完成', '当前页仍在 OCR 预标注，请等待识别完成后保存。')
            return False
        self._cancel_background_prefetch()
        self._sync_current_boxes_to_item()
        from worker import SaveCurrentWorker
        worker = SaveCurrentWorker(self.output_dir, item, self.base_image_bgr, parent=self.main_window)
        self._save_worker = worker
        worker.progress.connect(lambda done,total,message: self.set_status(f'{message} {done}/{total}')
                                if worker is self._save_worker else None)
        worker.page_saved.connect(self._on_save_page_saved)
        worker.finished_ok.connect(lambda: self._on_save_all_finished(worker))
        worker.error.connect(lambda message: self._on_save_all_error(worker, message))
        worker.stopped.connect(lambda message: self._on_save_all_stopped(worker, message))
        worker.finished.connect(lambda: self._on_save_worker_thread_finished(worker))
        self._track_worker(worker)
        try:
            hook = getattr(self, 'on_save_worker_created', None)
            if callable(hook):
                hook(worker)
            worker.start()
        except Exception as error:
            self._untrack_worker(worker)
            self._release_save_worker(worker)
            hook = getattr(self, 'on_save_worker_start_failed', None)
            if callable(hook):
                hook(worker)
            self._on_worker_error(f'无法启动保存：{error}')
            return False
        self.set_status('正在后台保存当前页：原图切片、JSON、Excel')
        return True

    def save_all(self):
        """批量保存，全部走后台线程。app.py 监听 save_worker 的 signal 显示进度。

        注意：本方法立即返回，不等保存完成。`quit_app` 流程由 app.py
        的 closeEvent / 进度对话框关闭来决定真正退出时机。
        """
        if not self.items:
            return False
        if self.ocr_handler is None:
            self._msg_warning("提示", "OCR 模型尚未加载完成，请稍后再试")
            return False
        if self._save_worker is not None and self._save_worker.isRunning():
            self.set_status("已有批量保存任务在运行")
            return True

        # 保存任务自己会做 OCR 兜底，先停掉空闲预取避免抢 CPU；
        # 保存结束后 _on_save_all_finished 会重新启动预取。
        self._cancel_background_prefetch()

        # 先同步当前页（纯内存操作，不阻塞）
        self._sync_current_boxes_to_item()

        save_worker = None
        try:
            from worker import SaveAllWorker
            # 后台线程只操作深快照；保存期间主线程仍可安全编辑当前页。
            items_snapshot = copy.deepcopy(self.items)
            save_worker = SaveAllWorker(
                self.output_dir, items_snapshot,
                self.ocr_handler, self.dbnet_short_side, self.level,
                self.pdf_dpi, self.poppler_path, self.image_scale,
                parent=self.main_window,
            )
            self._save_worker = save_worker
            # 默认信号接到状态栏
            save_worker.progress.connect(
                lambda done, total, name: self.set_status(f"批量保存 {done}/{total}：{name}")
            )
            save_worker.error.connect(
                lambda message, worker=save_worker: self._on_save_all_error(
                    worker,
                    message,
                )
            )
            save_worker.stopped.connect(
                lambda message, worker=save_worker: self._on_save_all_stopped(
                    worker,
                    message,
                )
            )
            save_worker.finished_ok.connect(
                lambda worker=save_worker: self._on_save_all_finished(worker)
            )
            save_worker.page_saved.connect(self._on_save_page_saved)
            save_worker.finished.connect(
                lambda worker=save_worker: self._on_save_worker_thread_finished(
                    worker
                )
            )
            self._track_worker(save_worker)
            worker_created_hook = getattr(
                self,
                "on_save_worker_created",
                None,
            )
            if callable(worker_created_hook):
                try:
                    worker_created_hook(save_worker)
                except Exception as e:
                    logger.warning("保存 worker UI 挂接失败：%s", e)
            save_worker.start()
        except Exception as e:
            logger.exception("启动批量保存任务失败：%s", e)
            if save_worker is not None and self._worker_is_running(save_worker):
                # 极少数 start() 已启动线程后才抛错的实现仍按运行中任务
                # 管理，结束回调会负责恢复退出状态和释放对象。
                self._on_save_all_error(
                    save_worker,
                    f"批量保存启动后异常：{e}",
                )
                return True
            if save_worker is self._save_worker:
                self._save_worker = None
            self._untrack_worker(save_worker)
            failed_hook = getattr(
                self,
                "on_save_worker_start_failed",
                None,
            )
            if callable(failed_hook):
                try:
                    failed_hook(save_worker)
                except Exception as hook_error:
                    logger.warning("清理保存启动 UI 失败：%s", hook_error)
            try:
                if save_worker is not None:
                    save_worker.deleteLater()
            except Exception:
                pass
            self._on_worker_error(f"无法启动批量保存：{e}")
            return False
        self.set_status("批量保存已启动（后台执行，可继续编辑当前页）")
        return True

    @staticmethod
    def _item_identity(item):
        source_path = os.path.normcase(
            os.path.realpath(os.path.abspath(item.get("source_path", "")))
        )
        return (
            item.get("type"),
            source_path,
            item.get("page_index"),
        )

    @staticmethod
    def _boxes_geometry_equal(left, right):
        left_boxes = normalize_boxes(left)
        right_boxes = normalize_boxes(right)
        return len(left_boxes) == len(right_boxes) and all(
            np.allclose(a, b, rtol=0.0, atol=1e-3)
            for a, b in zip(left_boxes, right_boxes)
        )

    def _on_save_page_saved(self, saved_item):
        """安全合并后台保存结果，不覆盖保存期间发生的编辑。"""

        saved_identity = self._item_identity(saved_item)
        live_index = next(
            (
                index
                for index, live_item in enumerate(self.items)
                if self._item_identity(live_item) == saved_identity
            ),
            None,
        )
        if live_index is None:
            return

        live_item = self.items[live_index]
        saved_boxes = normalize_boxes(saved_item.get("boxes", []))
        same_geometry = self._boxes_geometry_equal(
            live_item.get("boxes", []),
            saved_boxes,
        )
        same_revision = int(live_item.get("_edit_revision", 0)) == int(
            saved_item.get("_edit_revision", 0)
        )
        untouched_unvisited = (
            not live_item.get("visited")
            and not live_item.get("dirty")
        )
        untouched_ocr_pending = (
            live_item.get("ocr_pending", False)
            and not live_item.get("dirty", False)
            and same_revision
        )
        live_item["ever_saved"] = True

        if (
            (not same_geometry or not same_revision)
            and not untouched_unvisited
            and not untouched_ocr_pending
        ):
            # 保存的是旧快照；当前框仍是未保存的新版本。
            live_item["dirty"] = True
            return

        if untouched_unvisited or untouched_ocr_pending:
            live_item["boxes"] = [box.copy() for box in saved_boxes]
            live_item["box_context"] = copy.deepcopy(
                saved_item.get("box_context")
            )
            live_item["box_source"] = "ocr"
        if untouched_ocr_pending:
            # 后台保存已为这页得到并持久化了权威框；使仍在运行的预取 OCR
            # 请求失效，避免它稍后把已保存状态覆盖回“未保存”。
            live_item["ocr_pending"] = False
            live_item["ocr_completed"] = True
            live_item["ocr_request_id"] = None
        live_item["dirty"] = False
        live_item["nav_scale"] = 1.0

    def _on_save_all_finished(self, worker=None):
        worker = worker or self._save_worker
        if worker is None:
            return
        worker._core_save_outcome = "success"
        if worker is not self._save_worker:
            return
        if getattr(worker, '_single_page', False) is True:
            self.set_status('当前页保存完成：原图切片、JSON、Excel')
        else:
            self.set_status(f"全部保存完成：{self.total_items} / {self.total_items}")
        # 保存期间预取被暂停；若有仍未标注的页面，继续空闲预取。
        self._maybe_start_background_prefetch()

    def _on_save_all_stopped(self, worker, message=None):
        """取消或部分失败：结束任务，但不宣告成功、也不自动退出。"""
        if message is None:
            message = worker
            worker = self._save_worker
        if worker is None:
            return
        worker._core_save_outcome = "stopped"
        worker._core_save_message = message
        if worker is not self._save_worker:
            return
        self.set_status(message)

    def _on_save_all_error(self, worker, message=None):
        """保存失败：显示错误并清理任务，保留窗口供用户处理。"""
        if message is None:
            message = worker
            worker = self._save_worker
        if worker is None:
            return
        worker._core_save_outcome = "error"
        worker._core_save_message = message
        if worker is not self._save_worker:
            return
        self._on_worker_error(message)

    def _on_save_worker_thread_finished(self, worker):
        """QThread 真正结束后再释放，避免销毁仍在运行的线程。"""
        if worker is not self._save_worker:
            self._release_save_worker(worker)
            return
        outcome = getattr(worker, "_core_save_outcome", "stopped")
        should_quit = (
            outcome == "success"
            and self._quit_state == self.QUIT_SAVING
        )
        if outcome != "success" and self._quit_state == self.QUIT_SAVING:
            self._restart_callback = None
            self._quit_state = self.QUIT_IDLE
            self._pending_quit = False
            self._set_quit_interaction_locked(False)
        self._release_save_worker(worker)
        if getattr(worker, '_single_page', False) is True and self._navigation_target is not None:
            target = self._navigation_target
            self._navigation_target = None
            if outcome == 'success' and self._quit_state == self.QUIT_IDLE:
                item = self.items[self.current_item_index]
                if item.get('dirty'):
                    self.set_status('保存期间有新编辑，已保留当前页；请再次保存后翻页')
                else:
                    self._load_item(target)
            elif self._quit_state == self.QUIT_IDLE:
                self.set_status('保存未完成，已保留当前页')
        if should_quit:
            # 用户可能在这个 worker 的旧快照保存期间继续编辑。退出前重新
            # 检查实时状态；若仍有未保存内容，用最新快照再保存一轮。
            if self._has_unsaved_items():
                if self.save_all():
                    self.set_status("检测到保存期间的新编辑，正在保存最新版本...")
                    return
                self._quit_state = self.QUIT_IDLE
                self._pending_quit = False
                self._set_quit_interaction_locked(False)
                self._restart_callback = None
                self.set_status("无法保存最新编辑，已取消后续操作")
                return
            self._begin_worker_shutdown()
        elif self._quit_state == self.QUIT_IDLE:
            self._maybe_start_background_prefetch()

    def _release_save_worker(self, worker=None):
        """释放已结束的保存 worker 引用。"""
        worker = worker or self._save_worker
        if worker is self._save_worker:
            self._save_worker = None
        try:
            if worker is not None:
                worker.deleteLater()
        except Exception:
            pass

    def cancel_save_all(self):
        if self._save_worker is not None and self._save_worker.isRunning():
            self._save_worker.cancel()
            self.set_status("正在取消后续保存或翻页，当前页写入完成后停止...")

    def _process_events(self):
        """在长任务（save_all）里泵一次事件循环，避免界面卡死。

        惰性导入 Qt，无 QApplication 时直接跳过（便于纯逻辑单测）。
        """
        try:
            from PySide6.QtWidgets import QApplication
            app = QApplication.instance()
            if app is not None:
                app.processEvents()
        except Exception as e:
            logger.debug("processEvents 异常：%s", e)

    def _has_unsaved_items(self):
        return any(
            item.get("dirty", False)
            or (
                item.get("ocr_completed", False)
                and not item.get("ever_saved", False)
            )
            for item in self.items
        )

    def request_restart(self, callback):
        """Reuse the save/stop lifecycle, then start again without closing Qt."""
        if self._quit_state != self.QUIT_IDLE:
            return False
        self._restart_callback = callback
        if not self.quit_app():
            self._restart_callback = None
            self.set_status("已取消重新开始，保留当前会话")
            return False
        return True

    def _complete_restart(self):
        callback = self._restart_callback
        if callback is None or self._worker_is_running(self._save_worker):
            return
        self._restart_callback = None
        # Invalidate queued results before replacing the session. Request IDs
        # stay monotonic so an old OCR signal cannot match a new request.
        self._session_generation += 1
        # Keep strong references until finished. Retired inference cannot block
        # a new session, but window closing still waits for these QThreads.
        self._workers = self._known_workers()
        for worker in self._workers:
            worker._retired_session = True
        self._model_worker = self._ocr_worker = self._save_worker = None
        self._prefetch_worker = None
        self._load_worker = None
        self._navigation_target = None
        self.ocr_handler = None
        self.items = []
        self.total_items = 0
        self.current_item_index = 0
        self.boxes = []
        self.base_image_bgr = self.view_image_bgr = None
        self._view_version += 1
        self.selected_box_index = 0
        self.selected_handle_index = -1
        self.drawing = self.rotating = self.panning = False
        self.creation_mode = self.rotate_mode = self.nav_mode = False
        self.creation_start_point = self.rotate_center = None
        self.rotate_start_vec = self.rotate_base_box = None
        self._quit_state = self.QUIT_IDLE
        self._pending_quit = self._allow_window_close = False
        self.running = False
        self._set_quit_interaction_locked(False)
        self._refresh_canvas()
        callback()

    def quit_app(self):
        """统一处理所有退出入口，并在任务结束后真正关闭窗口。"""
        if self._quit_state != self.QUIT_IDLE:
            self.set_status("退出请求正在处理中...")
            return False

        if self.items:
            self._sync_current_boxes_to_item()

        if not self._has_unsaved_items():
            self._set_quit_interaction_locked(True)
            self._begin_worker_shutdown()
            return True

        restarting = self._restart_callback is not None
        action = "重新开始" if restarting else "退出"
        choice = self._msg_save_discard_cancel(
            action + "确认",
            f"当前会话存在未保存内容。请选择{action}方式：",
            save_label="保存并" + action,
            discard_label="不保存" + action,
        )
        if choice == "cancel":
            self.set_status("已取消退出")
            return False
        if choice == "discard":
            self._set_quit_interaction_locked(True)
            self._begin_worker_shutdown()
            return True

        self._quit_state = self.QUIT_SAVING
        self._pending_quit = True
        self._set_quit_interaction_locked(True)
        if self.save_all():
            self.set_status("正在保存，完成后将" + ("重新开始..." if self._restart_callback else "退出..."))
            return True

        self._quit_state = self.QUIT_IDLE
        self._pending_quit = False
        self._set_quit_interaction_locked(False)
        self.set_status("无法启动保存，已取消退出")
        return False

    def _known_workers(self):
        workers = []
        seen_ids = set()
        for worker in getattr(self, "_workers", []):
            if worker is not None and id(worker) not in seen_ids:
                workers.append(worker)
                seen_ids.add(id(worker))
        for attr in ("_model_worker", "_ocr_worker", "_save_worker", "_load_worker"):
            worker = getattr(self, attr, None)
            if worker is not None and id(worker) not in seen_ids:
                workers.append(worker)
                seen_ids.add(id(worker))
        return workers

    @staticmethod
    def _worker_is_running(worker):
        try:
            return bool(worker is not None and worker.isRunning())
        except Exception:
            return False

    def _watch_worker_for_quit(self, worker):
        if worker is None or getattr(worker, "_quit_watch_connected", False):
            return
        try:
            worker._quit_watch_connected = True
            worker.finished.connect(self._maybe_finish_quit)
        except Exception as e:
            logger.debug("监听 worker 结束失败：%s", e)

    def _begin_worker_shutdown(self):
        """请求所有任务停止；线程真正结束前不关闭窗口。"""
        self._set_quit_interaction_locked(True)
        self._quit_state = self.QUIT_STOPPING
        self._pending_quit = False
        self.running = False
        running_workers = [
            worker
            for worker in self._known_workers()
            if self._worker_is_running(worker)
        ]
        for worker in running_workers:
            self._watch_worker_for_quit(worker)
            try:
                worker.cancel()
            except Exception as e:
                logger.debug("请求取消 worker 失败：%s", e)

        if self._restart_callback is not None and not self._worker_is_running(self._save_worker):
            from PySide6.QtCore import QTimer
            QTimer.singleShot(0, self._complete_restart)
            return

        # cancel() 可能同步或极快地让 worker 结束，因此不能继续使用
        # 取消前捕获的 running_workers 快照。
        if self.any_worker_running():
            self.set_status("正在等待后台任务安全结束后" + ("重新开始..." if self._restart_callback else "退出..."))
            return
        self._really_quit()

    def _maybe_finish_quit(self):
        if self._quit_state != self.QUIT_STOPPING:
            return
        if any(
            self._worker_is_running(worker)
            for worker in self._known_workers()
        ):
            return
        self._really_quit()

    def _really_quit(self):
        """允许并重新触发窗口关闭；只能在线程全部停止后调用。"""
        if any(
            self._worker_is_running(worker)
            for worker in self._known_workers()
        ):
            self._begin_worker_shutdown()
            return False

        if self._restart_callback is not None:
            # Run after all finished-signal cleanup callbacks have been drained.
            from PySide6.QtCore import QTimer
            QTimer.singleShot(0, self._complete_restart)
            return True
        self._pending_quit = False
        self._quit_state = self.QUIT_CLOSING
        self._allow_window_close = True
        self.running = False
        self.disable_all_hotkeys()
        main_window = self.main_window
        if main_window is not None and hasattr(main_window, "close"):
            try:
                from PySide6.QtCore import QTimer

                QTimer.singleShot(0, main_window.close)
            except Exception as e:
                logger.debug("调度窗口关闭失败，改为直接关闭：%s", e)
                main_window.close()
        return True

    def can_close_window(self):
        return bool(self._allow_window_close)

    def _request_navigation(self, target):
        if not 0 <= target < self.total_items or self._quit_state != self.QUIT_IDLE:
            return False
        if self._navigation_target is not None or self._load_worker is not None:
            self.set_status('正在保存或加载页面，请勿重复翻页')
            return False
        self._navigation_target = target
        if not self._prompt_save_current_if_needed():
            self._navigation_target = None
            return False
        if getattr(self._save_worker, '_single_page', False) is True:
            return True  # Only the successful finished callback can continue.
        self._navigation_target = None
        return self._load_item(target)

    def prev_item(self):
        return self._request_navigation(self.current_item_index - 1)

    def next_item(self):
        return self._request_navigation(self.current_item_index + 1)

    # ---------- 文件/项目加载 ----------
    def _collect_input_items(self):
        image_exts = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tif", "*.tiff")
        imgs = []
        for ext in image_exts:
            imgs.extend(glob(os.path.join(self.input_dir, ext)))
            imgs.extend(glob(os.path.join(self.input_dir, ext.upper())))
        imgs = sorted(set(imgs))

        pdfs = sorted(set(glob(os.path.join(self.input_dir, "*.pdf")) + glob(os.path.join(self.input_dir, "*.PDF"))))
        items = []
        for p in imgs:
            items.append({
                "type": "image",
                "source_path": p,
                "page_index": None,
                "display_name": os.path.basename(p),
                "visited": False,
                "boxes": [],
                "dirty": False,
                "_edit_revision": 0,
                "ever_saved": False,
                "ocr_pending": False,
                "ocr_completed": False,
                "ocr_request_id": None,
                "view_scale": None,
                "nav_scale": 1.0,
            })

        for pdf in pdfs:
            try:
                page_count = get_pdf_page_count(pdf, poppler_path=self.poppler_path)
            except Exception as e:
                logger.warning("跳过 PDF：%s，无法读取页数：%s", pdf, e)
                continue
            for pi in range(page_count):
                items.append({
                    "type": "pdf_page",
                    "source_path": pdf,
                    "page_index": pi,
                    "display_name": f"{os.path.basename(pdf)} [Page {pi + 1}]",
                    "visited": False,
                    "boxes": [],
                    "dirty": False,
                    "_edit_revision": 0,
                    "ever_saved": False,
                    "ocr_pending": False,
                    "ocr_completed": False,
                    "ocr_request_id": None,
                    "view_scale": None,
                    "nav_scale": 1.0,
                })
        return items

    def _load_base_pil_for_item(self, item) -> Image.Image:
        if item.get("type") == "image":
            pil_rgb = Image.open(item["source_path"]).convert("RGB")
        else:
            pil_rgb = render_pdf_page_to_pil(
                item["source_path"], item["page_index"], dpi=self.pdf_dpi, poppler_path=self.poppler_path
            )

        # Original pixels are the annotation and export coordinate system.
        return pil_rgb.convert("RGB")

    def _load_item(self, target_index=None):
        """Start background original-page loading; keep the displayed page intact."""
        target = self.current_item_index if target_index is None else target_index
        if not 0 <= target < len(self.items):
            return False
        if self._load_worker is not None:
            self._load_worker.cancel()
        self._cancel_background_prefetch()
        from worker import PageLoadWorker
        context = {'target_index': target, 'generation': self._session_generation,
                   'item_identity': self._item_identity(self.items[target]),
                   'departing_index': self.current_item_index,
                   'departing_revision': int(self.items[self.current_item_index].get('_edit_revision', 0))}
        worker = PageLoadWorker(self.items[target], self.output_dir, self.pdf_dpi,
                                self.poppler_path, context, parent=self.main_window)
        self._load_worker = worker
        worker.loaded.connect(lambda payload: self._on_page_loaded(worker, payload))
        worker.error.connect(lambda message: self._on_page_load_error(worker, message))
        worker.finished.connect(lambda: self._on_page_load_finished(worker))
        self._track_worker(worker)
        try:
            worker.start()
        except Exception as error:
            self._load_worker = None
            self._untrack_worker(worker)
            worker.deleteLater()
            self._msg_critical('加载失败', str(error))
            return False
        self.set_status(f'正在后台加载原图：{self.items[target].get("display_name", "")}')
        return True

    def _on_page_loaded(self, worker, payload):
        context = payload['context']
        if (worker is not self._load_worker or context['generation'] != self._session_generation
                or self._quit_state != self.QUIT_IDLE):
            return
        target = context['target_index']
        if not 0 <= target < len(self.items) or self._item_identity(self.items[target]) != context['item_identity']:
            return
        previous = self.current_item_index
        if (target != previous and context.get('departing_index') == previous
                and int(self.items[previous].get('_edit_revision', 0)) != context['departing_revision']):
            self.set_status('加载期间有新编辑，已保留当前页；请保存后再次翻页')
            return
        self.current_item_index = target
        if not self._apply_loaded_item(payload['pil'], payload['bgr'], payload['record']):
            self.current_item_index = previous
            self._abort_initial_load()

    def _abort_initial_load(self):
        if self.base_image_bgr is None:
            self.running = False
            self.ocr_handler = None
            self.set_hotkeys_enabled(False)
            callback = getattr(self, 'on_session_start_aborted', None)
            if callable(callback):
                callback()

    def _on_page_load_error(self, worker, message):
        if worker is self._load_worker and self._quit_state == self.QUIT_IDLE:
            self._msg_critical('加载失败', message)
            self.set_status('页面加载失败，已保留当前页')
            self._abort_initial_load()

    def _on_page_load_finished(self, worker):
        if worker is self._load_worker:
            self._load_worker = None
            self._maybe_start_background_prefetch()
        worker.deleteLater()

    def _apply_loaded_item(self, pil_rgb, base_bgr, record):
        """UI-only merge of already decoded pixels and already read JSON."""
        item = self.items[self.current_item_index]
        # boxes 优先级：会话内已访问 boxes > JSON > OCR（异步）。
        boxes_cand = None
        box_source = None
        if item.get("visited"):
            has_authoritative_state = (
                bool(item.get("boxes"))
                or item.get("dirty", False)
                or item.get("ocr_pending", False)
                or item.get("ocr_completed", False)
                or item.get("ever_saved", False)
            )
            if has_authoritative_state:
                boxes_cand = normalize_boxes(item.get("boxes"))
                box_source = item.get("box_source", "session")
        else:
            if record is not None:
                saved_context = record.get("annotation_context", {})
                current_context = item.get("annotation_context", {})
                use_saved = annotation_context_matches(
                    saved_context,
                    current_context,
                )
                if not use_saved:
                    choice = self._msg_save_discard_cancel(
                        "已有标注与当前设置不一致",
                        self._annotation_context_mismatch_text(
                            saved_context,
                            current_context,
                        ),
                        save_label="使用旧框",
                        discard_label="重新 OCR",
                    )
                    if choice == "cancel":
                        self.set_status("已取消加载该页标注")
                        return False
                    use_saved = choice == "save"
                if use_saved:
                    boxes_cand = normalize_boxes(record.get("boxes"))
                    item["box_context"] = normalize_annotation_context(
                        saved_context
                    )
                    item["box_source"] = "json"
                    box_source = self._annotation_context_label(
                        "JSON",
                        saved_context,
                    )
                    item["ever_saved"] = True
                    item["dirty"] = False
                    item["ocr_pending"] = False
                    item["ocr_completed"] = False
                    item["ocr_request_id"] = None
                    logger.info("已从 JSON 载入标注：%d 个", len(boxes_cand))
            elif item.get("ocr_completed") and item.get("boxes"):
                # 后台预取已完成该页：直接显示内存结果，无需再跑 OCR。
                boxes_cand = normalize_boxes(item.get("boxes"))
                item["box_source"] = "ocr"
                box_source = self._annotation_context_label(
                    "后台预标注",
                    item.get("box_context"),
                )
        item["visited"] = True

        self.base_image_bgr = base_bgr
        self.base_size = pil_rgb.size
        self.current_item_title = item.get("display_name", "")

        if boxes_cand is None:
            # 没有 JSON 也没访问过：先空 boxes 显示，OCR 在后台补上。
            self.boxes = []
            self.selected_box_index = 0
            self._apply_view_for_current_item()
            self._refresh_canvas()
            if item.get("ocr_pending"):
                # 该页正被后台预取处理：抢占批量任务，为当前页单独重跑，
                # 让用户正看着的页最快出结果（迟到的预取结果会被丢弃）。
                self._cancel_background_prefetch()
                self.set_status(
                    f"图片已加载，正在 OCR 预标注：{self.current_item_title}"
                )
                self._start_ocr_prefetch(pil_rgb)
            else:
                self.set_status(
                    f"图片已加载，正在 OCR 预标注：{self.current_item_title}"
                )
                self._start_ocr_prefetch(pil_rgb)
            return True

        self.boxes = [b.copy().astype(np.float32) for b in boxes_cand]
        self.selected_box_index = 0
        self._apply_view_for_current_item()
        self._refresh_canvas()
        source_label = box_source or "会话"
        if "Level=" not in source_label and "Level 未记录" not in source_label:
            source_label = self._annotation_context_label(
                source_label,
                item.get("box_context"),
            )
        self.set_status(
            f"已加载 ({self.current_item_index + 1}/{self.total_items}) {self.current_item_title} | "
            f"框数: {len(self.boxes)} | 来源: {source_label} | "
            f"显示缩放: {self.view_scale:.2f}"
        )
        # 当前页立即可见后，空闲时间继续预取后续页面。
        self._maybe_start_background_prefetch()
        return True

    @staticmethod
    def _annotation_context_label(source, context):
        normalized = normalize_annotation_context(context)
        if not normalized:
            return f"{source}（旧版，Level 未记录）"
        return (
            f"{source}（Level={normalized.get('level', '未知')}, "
            f"det={normalized.get('det_model', '未知')}, "
            f"rec={normalized.get('rec_model', '未知')}）"
        )

    @staticmethod
    def _annotation_context_mismatch_text(saved_context, current_context):
        saved = normalize_annotation_context(saved_context)
        current = normalize_annotation_context(current_context)

        def describe(context):
            if not context:
                return "旧版 JSON（未记录识别设置）"
            return (
                f"Level={context.get('level', '未知')}, "
                f"det={context.get('det_model', '未知')}, "
                f"rec={context.get('rec_model', '未知')}, "
                f"char_box={context.get('char_box_strategy', '未知')}, "
                f"框版本={context.get('char_box_revision', '旧版')}, "
                f"检测尺寸={context.get('dbnet_short_side', '未记录')}, "
                f"DPI={context.get('pdf_dpi', '未记录')}, "
                f"图像缩放={context.get('image_scale', '未记录')}"
            )

        return (
            "保存的标注框与当前识别设置不一致：\n\n"
            f"已有：{describe(saved)}\n"
            f"当前：{describe(current)}\n\n"
            "使用旧框会保留原来的框粒度；重新 OCR 会按当前设置生成新框。"
        )

    def _apply_view_for_current_item(self):
        """根据当前 base_image 重建 view image + 恢复 view_scale。"""
        restored_scale = None
        if 0 <= self.current_item_index < len(self.items):
            it = self.items[self.current_item_index]
            restored_scale = it.get("view_scale")
            if restored_scale is None:
                restored_scale = it.get("nav_scale") if it.get("nav_scale") not in (None, 1.0) else None
        if restored_scale is None:
            self.view_scale = self._fit_scale_for_current_base()
        else:
            self.view_scale = float(np.clip(float(restored_scale), self.MIN_VIEW_SCALE, self.MAX_VIEW_SCALE))
        self._rebuild_view_image()
        self._center_view_if_large()
        self._compute_draw_and_clamp()
        self._update_window_title()

    def _start_ocr_prefetch(self, pil_rgb):
        if self.ocr_handler is None:
            # 模型还没加载完（极少见，正常流程下 start() 先加载模型再 load_item）。
            self.set_status("OCR 模型尚未就绪，请稍候")
            return
        # 当前页识别优先于后台预取：先停掉批量预取，避免两个 OCR 抢 CPU。
        self._cancel_background_prefetch()
        from worker import OcrPrefetchWorker

        target_index = self.current_item_index
        item = self.items[target_index]
        request_id = self._next_ocr_request_id
        self._next_ocr_request_id += 1
        context = {
            "target_index": target_index,
            "item_identity": self._item_identity(item),
            "edit_revision": int(item.get("_edit_revision", 0)),
            "request_id": request_id,
            "annotation_context": copy.deepcopy(
                item.get("annotation_context", {})
            ),
        }
        item["ocr_pending"] = True
        item["ocr_completed"] = False
        item["ocr_request_id"] = request_id
        # 旧字段保留用于诊断，不再作为结果归属依据。
        self._ocr_target_index = target_index
        ocr_worker = OcrPrefetchWorker(
            pil_rgb, self.dbnet_short_side, self.level,
            self.ocr_handler, context=context, parent=self.main_window,
        )
        self._ocr_worker = ocr_worker
        ocr_worker.progress.connect(
            lambda message, worker=ocr_worker: self.set_status(message)
            if worker is self._ocr_worker else None
        )
        ocr_worker.ocr_done.connect(
            lambda payload, worker=ocr_worker: self._on_ocr_done(
                worker,
                payload,
            )
        )
        ocr_worker.error.connect(
            lambda message, worker=ocr_worker: self._on_ocr_error(
                worker,
                message,
            )
        )
        ocr_worker.finished.connect(
            lambda worker=ocr_worker: self._on_ocr_worker_thread_finished(
                worker
            )
        )
        self._track_worker(ocr_worker)
        ocr_worker.start()

    # ---------- 空闲预取：后台顺序 OCR 未访问页 ----------
    def _item_needs_background_ocr(self, item):
        """该页是否值得后台预标注：未访问、无任何权威标注来源。"""
        if item.get("visited"):
            return False
        if item.get("boxes"):
            return False
        if item.get("dirty") or item.get("ever_saved"):
            return False
        if item.get("ocr_pending") or item.get("ocr_completed"):
            return False
        if item.get("ocr_failed"):
            return False
        try:
            has_json = (
                load_item_annotation_record_if_exists(
                    self.output_dir, item
                )
                is not None
            )
        except Exception as e:
            logger.debug("检查标注 JSON 失败，跳过预取 %s：%s",
                         item.get("display_name", ""), e)
            return False
        return not has_json

    def _reset_pending_prefetch_requests(self, worker):
        """把指定预取批次里仍未完成的页面恢复为可重新调度状态。"""
        request_ids = getattr(worker, "prefetch_request_ids", None) or ()
        for item in self.items:
            if (
                item.get("ocr_pending")
                and item.get("ocr_request_id") in request_ids
            ):
                item["ocr_pending"] = False
                item["ocr_request_id"] = None

    def _cancel_background_prefetch(self):
        worker = self._prefetch_worker
        if worker is None:
            return
        self._prefetch_worker = None
        try:
            worker.cancel()
        except Exception as e:
            logger.debug("请求取消后台预取失败：%s", e)
        self._reset_pending_prefetch_requests(worker)

    def _maybe_start_background_prefetch(self):
        """条件满足时启动（或继续）后台批量预标注。

        启动条件：模型就绪、非退出流程、无保存任务、当前页 OCR 不在跑
        （当前页永远优先）、没有已在跑的预取。候选顺序从当前页向后绕环。
        """
        if self.ocr_handler is None or not self.items:
            return
        if self._load_worker is not None or self._navigation_target is not None:
            return
        if self._quit_state != self.QUIT_IDLE:
            return
        if self._prefetch_worker is not None:
            return
        if self._save_worker is not None and self._worker_is_running(
            self._save_worker
        ):
            return
        current = (
            self.items[self.current_item_index]
            if 0 <= self.current_item_index < len(self.items)
            else None
        )
        if current is not None and current.get("ocr_pending"):
            return

        total = len(self.items)
        order = [
            (self.current_item_index + step) % total
            for step in range(1, total + 1)
        ]
        jobs = []
        for idx in order:
            item = self.items[idx]
            if not self._item_needs_background_ocr(item):
                continue
            request_id = self._next_ocr_request_id
            self._next_ocr_request_id += 1
            context = {
                "target_index": idx,
                "item_identity": self._item_identity(item),
                "edit_revision": int(item.get("_edit_revision", 0)),
                "request_id": request_id,
                "annotation_context": copy.deepcopy(
                    item.get("annotation_context", {})
                ),
            }
            item["ocr_pending"] = True
            item["ocr_completed"] = False
            item["ocr_request_id"] = request_id
            jobs.append(
                {
                    "type": item.get("type"),
                    "source_path": item.get("source_path"),
                    "page_index": item.get("page_index"),
                    "display_name": item.get("display_name", ""),
                    "context": context,
                }
            )
        if not jobs:
            return

        from worker import OcrBatchPrefetchWorker
        prefetch_worker = OcrBatchPrefetchWorker(
            jobs,
            self.ocr_handler,
            self.dbnet_short_side,
            self.level,
            self.pdf_dpi,
            self.poppler_path,
            self.image_scale,
            parent=self.main_window,
        )
        prefetch_worker.prefetch_request_ids = frozenset(
            job["context"]["request_id"] for job in jobs
        )
        try:
            prefetch_worker.page_ocr_done.connect(
                lambda payload, worker=prefetch_worker: self._on_ocr_done(
                    worker, payload
                )
            )
            prefetch_worker.page_ocr_failed.connect(
                lambda payload, worker=prefetch_worker: (
                    self._on_prefetch_page_failed(payload)
                )
            )
            prefetch_worker.error.connect(
                lambda message, worker=prefetch_worker: (
                    self._on_prefetch_error(worker, message)
                )
            )
            prefetch_worker.finished.connect(
                lambda worker=prefetch_worker: (
                    self._on_prefetch_thread_finished(worker)
                )
            )
            self._prefetch_worker = prefetch_worker
            self._track_worker(prefetch_worker)
            prefetch_worker.start()
            logger.info("后台预取已启动：%d 个未标注页", len(jobs))
        except Exception as e:
            logger.exception("启动后台预取失败：%s", e)
            if self._prefetch_worker is prefetch_worker:
                self._prefetch_worker = None
            self._untrack_worker(prefetch_worker)
            self._reset_pending_prefetch_requests(prefetch_worker)
            try:
                prefetch_worker.deleteLater()
            except Exception:
                pass

    def _on_prefetch_page_failed(self, payload):
        """单页预取失败：标记后不再重试，保持无感（不弹错误框）。"""
        context = payload.get("context", {}) if isinstance(payload, dict) else {}
        identity = context.get("item_identity")
        idx = (
            self._find_item_index_by_identity(identity)
            if identity is not None
            else context.get("target_index", -1)
        )
        if idx is None or not (0 <= idx < len(self.items)):
            return
        item = self.items[idx]
        if item.get("ocr_request_id") != context.get("request_id"):
            return
        item["ocr_pending"] = False
        item["ocr_request_id"] = None
        item["ocr_failed"] = True
        logger.debug(
            "页面预取失败，已标记跳过：%s", item.get("display_name", "")
        )

    def _on_prefetch_error(self, worker, message):
        if getattr(worker, "_retired_session", False) is True:
            return
        self._reset_pending_prefetch_requests(worker)
        if self._quit_state in (self.QUIT_STOPPING, self.QUIT_CLOSING):
            self.set_status(f"后台预取已停止：{message}")
            return
        self._on_worker_error(f"后台预取失败：{message}")

    def _on_prefetch_thread_finished(self, worker):
        if worker is not self._prefetch_worker:
            return
        self._prefetch_worker = None
        self._reset_pending_prefetch_requests(worker)
        # 批次结束（如保存任务刚完成）后继续预取剩余页面。
        if self._quit_state == self.QUIT_IDLE:
            self._maybe_start_background_prefetch()

    def _find_item_index_by_identity(self, identity):
        return next(
            (
                index
                for index, item in enumerate(self.items)
                if self._item_identity(item) == tuple(identity)
            ),
            None,
        )

    def _on_ocr_done(self, worker, payload=None):
        """只把 OCR 结果写回任务启动时绑定且未被编辑的页面。"""
        if payload is None:
            payload = worker
            worker = self._ocr_worker
        payload = payload if isinstance(payload, dict) else {"boxes": payload}
        context = payload.get("context", {}) or getattr(
            worker,
            "_context",
            {},
        )
        identity = context.get("item_identity")
        idx = (
            self._find_item_index_by_identity(identity)
            if identity is not None
            else context.get("target_index", -1)
        )
        if idx is None or not (0 <= idx < len(self.items)):
            return
        item = self.items[idx]
        request_id = context.get("request_id")
        if item.get("ocr_request_id") != request_id:
            logger.debug("丢弃已失效的 OCR 结果：request=%s", request_id)
            return

        item["ocr_pending"] = False
        item["ocr_request_id"] = None
        started_revision = int(context.get("edit_revision", 0))
        current_revision = int(item.get("_edit_revision", 0))
        if item.get("dirty", False) or current_revision != started_revision:
            if idx == self.current_item_index:
                self.set_status("OCR 已完成；检测到手动编辑，已保留当前标注")
            return

        boxes = payload.get("boxes", [])
        normalized = normalize_boxes(boxes)
        item["boxes"] = [b.copy().astype(np.float32) for b in normalized]
        item["ever_saved"] = False
        item["dirty"] = False
        item["ocr_completed"] = True
        item["box_context"] = normalize_annotation_context(
            context.get("annotation_context")
        )
        item["box_source"] = "ocr"
        if idx == self.current_item_index:
            # 还在当前页：合并显示
            self.boxes = [b.copy().astype(np.float32) for b in normalized]
            self.selected_box_index = 0
            self._refresh_canvas()
            self.set_status(
                f"OCR 完成 ({self.current_item_index + 1}/{self.total_items}) {self.current_item_title} | "
                f"框数: {len(self.boxes)} | 来源: "
                f"{self._annotation_context_label('本次 OCR', item.get('box_context'))} | "
                f"显示缩放: {self.view_scale:.2f}"
            )
        else:
            logger.debug("OCR 完成但用户已切到第 %d 页，仅写入 items[%d]",
                         self.current_item_index + 1, idx + 1)
        geometry = payload.get('ocr_geometry')
        if geometry:
            item['ocr_geometry'] = copy.deepcopy(geometry)
            if idx == self.current_item_index:
                source = '×'.join(map(str, geometry['source_size']))
                proxy = '×'.join(map(str, geometry['ocr_size']))
                self.set_status(f'OCR 完成：{len(self.boxes)} 个框 | 原图 {source} → 推理 {proxy} | 坐标已回映原图')
        # 当前页识别结束（或预取结果落盘）后，空闲继续预取剩余页面。
        self._maybe_start_background_prefetch()

    def _on_ocr_error(self, worker, message):
        context = getattr(worker, "_context", {}) or {}
        identity = context.get("item_identity")
        request_id = context.get("request_id")
        idx = (
            self._find_item_index_by_identity(identity)
            if identity is not None
            else context.get("target_index", -1)
        )
        if request_id is not None:
            if idx is None or not (0 <= idx < len(self.items)):
                logger.debug("忽略已失去页面归属的 OCR 错误：%s", message)
                return
            item = self.items[idx]
            if item.get("ocr_request_id") != request_id:
                logger.debug(
                    "忽略旧 OCR 请求错误：request_id=%s，当前=%s",
                    request_id,
                    item.get("ocr_request_id"),
                )
                return
            item["ocr_pending"] = False
            item["ocr_request_id"] = None
        if self._quit_state in (self.QUIT_STOPPING, self.QUIT_CLOSING):
            self.set_status(f"OCR 已停止：{message}")
            return
        self._on_worker_error(message)

    def _on_ocr_worker_thread_finished(self, worker):
        if worker is self._ocr_worker:
            self._ocr_worker = None
        try:
            worker.deleteLater()
        except Exception:
            pass

    # ---------- 显示缩放/坐标映射 ----------
    def _fit_scale_for_current_base(self) -> float:
        if self.base_image_bgr is None:
            return 1.0
        bw, bh = self.base_size
        if bw <= 0 or bh <= 0 or self.VIEW_W <= 0 or self.VIEW_H <= 0:
            return 1.0
        return float(np.clip(min(self.VIEW_W / bw, self.VIEW_H / bh, 1.0), self.MIN_VIEW_SCALE, self.MAX_VIEW_SCALE))

    def _set_view_scale(self, new_scale: float, center: bool = False):
        if self.base_image_bgr is None:
            return
        self.view_scale = float(np.clip(new_scale, self.MIN_VIEW_SCALE, self.MAX_VIEW_SCALE))
        self._rebuild_view_image()
        if center:
            self._center_view_if_large()
        self._compute_draw_and_clamp()
        self._mark_view_state()
        self._update_window_title()
        self.set_status(f"显示缩放: {self.view_scale:.2f}（不影响保存清晰度）")

    def _rebuild_view_image(self):
        if self.base_image_bgr is None:
            return
        if abs(self.view_scale - 1.0) < 1e-6:
            self.view_image_bgr = self.base_image_bgr.copy()
        else:
            h0, w0 = self.base_image_bgr.shape[:2]
            new_w = max(1, int(round(w0 * self.view_scale)))
            new_h = max(1, int(round(h0 * self.view_scale)))
            interp = cv2.INTER_AREA if self.view_scale < 1.0 else cv2.INTER_CUBIC
            self.view_image_bgr = cv2.resize(self.base_image_bgr, (new_w, new_h), interpolation=interp)
        # getattr 兜底：兼容 object.__new__ 绕过 __init__ 的纯逻辑测试。
        self._view_version = getattr(self, "_view_version", 0) + 1
        self._refresh_canvas()

    def _compute_draw_and_clamp(self):
        if self.view_image_bgr is None:
            return
        H, W = self.view_image_bgr.shape[:2]
        if W <= self.VIEW_W and H <= self.VIEW_H:
            dx = (self.VIEW_W - W) // 2
            dy = (self.VIEW_H - H) // 2
            self.draw_origin = np.array([dx, dy], dtype=np.int32)
            self.viewport_offset[:] = 0
        else:
            self.draw_origin[:] = 0
            max_x = max(0, W - self.VIEW_W)
            max_y = max(0, H - self.VIEW_H)
            self.viewport_offset[0] = int(np.clip(self.viewport_offset[0], 0, max_x))
            self.viewport_offset[1] = int(np.clip(self.viewport_offset[1], 0, max_y))

    def _center_view_if_large(self):
        if self.view_image_bgr is None:
            return
        H, W = self.view_image_bgr.shape[:2]
        self.viewport_offset[0] = max(0, (W - self.VIEW_W) // 2)
        self.viewport_offset[1] = max(0, (H - self.VIEW_H) // 2)

    def _base_to_screen(self, pt) -> Tuple[int, int]:
        vx = float(pt[0]) * self.view_scale
        vy = float(pt[1]) * self.view_scale
        H, W = self.view_image_bgr.shape[:2]
        if W <= self.VIEW_W and H <= self.VIEW_H:
            return int(round(vx + self.draw_origin[0])), int(round(vy + self.draw_origin[1]))
        return int(round(vx - self.viewport_offset[0])), int(round(vy - self.viewport_offset[1]))

    def _screen_to_base(self, sx, sy) -> Optional[np.ndarray]:
        if self.view_image_bgr is None or self.view_scale <= 0:
            return None
        H, W = self.view_image_bgr.shape[:2]
        if W <= self.VIEW_W and H <= self.VIEW_H:
            vx = sx - self.draw_origin[0]
            vy = sy - self.draw_origin[1]
        else:
            vx = self.viewport_offset[0] + sx
            vy = self.viewport_offset[1] + sy
        if vx < 0 or vy < 0 or vx >= W or vy >= H:
            return None
        return np.array([vx / self.view_scale, vy / self.view_scale], dtype=np.float32)

    def _zoom_at_screen(self, factor, screen_pt):
        if self.base_image_bgr is None or self.view_image_bgr is None:
            return
        old_scale = self.view_scale
        new_scale = float(np.clip(old_scale * float(factor), self.MIN_VIEW_SCALE, self.MAX_VIEW_SCALE))
        if abs(new_scale - old_scale) < 1e-6:
            return

        anchor_base = self._screen_to_base(screen_pt[0], screen_pt[1])
        if anchor_base is None:
            bw, bh = self.base_size
            anchor_base = np.array([bw / 2.0, bh / 2.0], dtype=np.float32)

        self.view_scale = new_scale
        self._rebuild_view_image()
        self._compute_draw_and_clamp()

        H, W = self.view_image_bgr.shape[:2]
        if W > self.VIEW_W or H > self.VIEW_H:
            anchor_view = anchor_base * self.view_scale
            self.viewport_offset[0] = int(round(anchor_view[0] - screen_pt[0]))
            self.viewport_offset[1] = int(round(anchor_view[1] - screen_pt[1]))
            self._compute_draw_and_clamp()
        self._mark_view_state()
        self._update_window_title()

    # ---------- 鼠标交互（Qt 事件分发后的入口） ----------
    # 这些方法由 canvas_widget.AnnotatorCanvas 的 mousePress/Move/Release/wheelEvent
    # 调用；逻辑等价于旧 _mouse_cb，只是把 cv2 EVENT_* 拆成了独立方法。
    def _on_wheel(self, factor, x, y):
        if self.view_image_bgr is None:
            return
        self._zoom_at_screen(factor, (x, y))
        self._refresh_canvas()

    def _on_mouse_left_down(self, x, y, alt):
        if self.view_image_bgr is None:
            return

        if self.nav_mode or alt:
            self.panning = True
            self.last_mouse = (x, y)
            return

        if self.rotate_mode:
            base_pt = self._screen_to_base(x, y)
            if self.boxes and base_pt is not None:
                self.rotating = True
                if self.selected_box_index < 0 or self.selected_box_index >= len(self.boxes):
                    self.selected_box_index = 0
                self.rotate_base_box = self.boxes[self.selected_box_index].copy()
                self.rotate_center = self.rotate_base_box.mean(axis=0)
                self.rotate_start_vec = base_pt - self.rotate_center
                if np.linalg.norm(self.rotate_start_vec) < 1e-6:
                    self.rotate_start_vec = np.array([1.0, 0.0], dtype=np.float32)
            return

        if self.creation_mode:
            base_pt = self._screen_to_base(x, y)
            if base_pt is not None:
                self.drawing = True
                self.creation_start_point = base_pt
                self.boxes.append(np.array([self.creation_start_point] * 4, dtype=np.float32))
                self.selected_box_index = len(self.boxes) - 1
                self._refresh_canvas()
            return

        # 默认：选框 / 拖 handle
        is_handle = False
        if self.boxes and self.selected_box_index < len(self.boxes):
            for j, point in enumerate(self.boxes[self.selected_box_index]):
                sx, sy = self._base_to_screen(point)
                if np.hypot(x - sx, y - sy) < self.HANDLE_RADIUS:
                    self.drawing = True
                    self.selected_handle_index = j
                    is_handle = True
                    break
        if not is_handle:
            base_pt = self._screen_to_base(x, y)
            if base_pt is not None:
                for i, box in enumerate(self.boxes):
                    if cv2.pointPolygonTest(box.astype(np.int32), (int(base_pt[0]), int(base_pt[1])), False) >= 0:
                        self.selected_box_index = i
                        self.drawing = False
                        self.selected_handle_index = -1
                        self._refresh_canvas()
                        break

    def _on_mouse_move(self, x, y, alt):
        if self.view_image_bgr is None:
            return

        if self.panning:
            dx = x - self.last_mouse[0]
            dy = y - self.last_mouse[1]
            self.last_mouse = (x, y)
            H, W = self.view_image_bgr.shape[:2]
            if W > self.VIEW_W or H > self.VIEW_H:
                self.viewport_offset[0] -= dx
                self.viewport_offset[1] -= dy
                self._compute_draw_and_clamp()
                self._mark_view_state()
                self._refresh_canvas()
            return

        if self.rotate_mode and self.rotating:
            base_pt = self._screen_to_base(x, y)
            if base_pt is None:
                return
            cur_vec = base_pt - self.rotate_center
            if np.linalg.norm(cur_vec) < 1e-6:
                return
            ang0 = math.atan2(self.rotate_start_vec[1], self.rotate_start_vec[0])
            ang1 = math.atan2(cur_vec[1], cur_vec[0])
            delta = ang1 - ang0
            self.boxes[self.selected_box_index] = self._rotate_pts(self.rotate_base_box, self.rotate_center, delta)
            self._mark_dirty()
            self._refresh_canvas()
            return

        if self.creation_mode and self.drawing:
            base_pt = self._screen_to_base(x, y)
            if base_pt is None:
                return
            p1 = self.creation_start_point
            p3 = base_pt
            self.boxes[self.selected_box_index] = np.array(
                [p1, [p3[0], p1[1]], p3, [p1[0], p3[1]]], dtype=np.float32
            )
            self._mark_dirty()
            self._refresh_canvas()
            return

        if self.drawing and self.selected_handle_index != -1:
            base_pt = self._screen_to_base(x, y)
            if base_pt is None or not self.boxes:
                return
            box = self.boxes[self.selected_box_index]
            d = self.selected_handle_index
            a = (d + 2) % 4
            a1 = (d + 1) % 4
            a2 = (d + 3) % 4
            p_anchor = box[a]
            v1 = box[a1] - p_anchor
            v2 = box[a2] - p_anchor
            n1 = np.linalg.norm(v1)
            n2 = np.linalg.norm(v2)
            if n1 < 1e-6 or n2 < 1e-6:
                return
            u1 = v1 / n1
            u2 = v2 / n2
            v_mouse = base_pt - p_anchor
            p1n = p_anchor + np.dot(v_mouse, u1) * u1
            p2n = p_anchor + np.dot(v_mouse, u2) * u2
            box[a1] = p1n
            box[a2] = p2n
            box[d] = p_anchor + (p1n - p_anchor) + (p2n - p_anchor)
            self._mark_dirty()
            self._refresh_canvas()

    def _on_mouse_left_up(self, x, y):
        if self.view_image_bgr is None:
            return
        if self.rotate_mode:
            self.rotating = False
            self._refresh_canvas()
            return
        if self.creation_mode and self.drawing:
            self.drawing = False
            self.creation_mode = False
            self.creation_start_point = None
            if (
                0 <= self.selected_box_index < len(self.boxes)
                and not normalize_boxes([self.boxes[self.selected_box_index]])
            ):
                self.boxes.pop(self.selected_box_index)
                self.selected_box_index = max(0, len(self.boxes) - 1)
                self.set_status("未创建标注框：框的宽度或高度过小")
            self._mark_dirty()
            self._refresh_canvas()
            return
        if self.panning:
            self.panning = False
            return
        if self.drawing:
            self.drawing = False
            self.selected_handle_index = -1
            self._refresh_canvas()

    # ---------- 小工具 ----------
    def _rotate_pts(self, pts, center, ang):
        c = np.array(center, dtype=np.float32)
        rot = np.array([[math.cos(ang), -math.sin(ang)], [math.sin(ang), math.cos(ang)]], dtype=np.float32)
        return ((pts - c) @ rot.T) + c

    def _sync_current_boxes_to_item(self):
        if 0 <= self.current_item_index < len(self.items):
            item = self.items[self.current_item_index]
            item["boxes"] = [b.copy().astype(np.float32) for b in self.boxes]
            item["view_scale"] = self.view_scale
            item["nav_scale"] = 1.0

    def _mark_dirty(self):
        if 0 <= self.current_item_index < len(self.items):
            item = self.items[self.current_item_index]
            item["dirty"] = True
            box_context = item.get("box_context")
            if box_context is None:
                box_context = (
                    item.get("annotation_context")
                    or self._current_annotation_context()
                )
            item["box_context"] = copy.deepcopy(
                box_context
            )
            item["box_source"] = "manual"
            item["_edit_revision"] = int(
                item.get("_edit_revision", 0)
            ) + 1
            self._sync_current_boxes_to_item()

    def _mark_view_state(self):
        if 0 <= self.current_item_index < len(self.items):
            self.items[self.current_item_index]["view_scale"] = self.view_scale

    def _prompt_save_current_if_needed(self):
        if not self.items:
            return True
        self._sync_current_boxes_to_item()
        item = self.items[self.current_item_index]
        # 翻页询问与旧版一致：手动编辑过，或存在尚未落盘的框（含纯
        # OCR / 后台预取结果）。OCR 尚未返回、画布为空的页没有可保存
        # 的内容，不打扰。
        has_unsaved_boxes = bool(item.get("boxes")) and not item.get(
            "ever_saved", False
        )
        needs_save = item.get("dirty", False) or has_unsaved_boxes
        if needs_save:
            choice = self._msg_save_discard_cancel(
                "保存确认",
                f"当前页面存在未保存内容：\n  {item.get('display_name', '')}\n请选择操作：",
                save_label="保存并翻页",
                discard_label="不保存并翻页",
            )
            if choice == "cancel":
                self.set_status("已取消翻页")
                return False
            if choice == "save":
                return bool(self.save_current())
        return True
