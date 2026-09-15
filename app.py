# -*- coding: utf-8 -*-
"""app.py — PySide6 版主窗口。

替代原 Tk 版的 app.py。改动要点：
- Tk → QMainWindow + QSplitter + QScrollArea；
- 画布用 canvas_widget.AnnotatorCanvas（QPainter 绘制）；
- 快捷键用 QShortcut（直接吃 WM_KEYDOWN，IME 激活时也照常触发）；
- 主循环交给 Qt 事件循环（app.exec），不再 while + cv2.waitKey + root.update；
- 对话框走 QMessageBox。
"""

import os
import sys

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QProgressDialog,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSplitter,
    QStatusBar,
    QVBoxLayout,
    QWidget,
)

from annotator_core import AnnotatorCore
from canvas_widget import AnnotatorCanvas

SIDEBAR_WIDTH = 320
BUTTON_FEEDBACK_STYLE = """
    QPushButton:hover {
        background-color: palette(midlight);
        border: 1px solid palette(mid);
    }
    QPushButton:pressed {
        background-color: #707070;
        color: white;
        border: 1px solid #454545;
    }
"""
# 不设硬性最小宽度：让内容 widget 自己的 sizeHint 通过 layout 链向上传递，
# splitter 拖动时会自然停在内容"刚好完整显示"的位置。
# 配合 setCollapsible(0, False) 只是禁止折叠到 0，不限制用户往宽拖。


def _parse_int(value, name, min_value=None, max_value=None):
    try:
        out = int(value)
    except Exception:
        raise ValueError(f"{name} 必须是整数")
    if min_value is not None and out < min_value:
        raise ValueError(f"{name} 不能小于 {min_value}")
    if max_value is not None and out > max_value:
        raise ValueError(f"{name} 不能大于 {max_value}")
    return out


def _parse_float(value, name, min_value=None, max_value=None):
    try:
        out = float(value)
    except Exception:
        raise ValueError(f"{name} 必须是数字")
    if min_value is not None and out < min_value:
        raise ValueError(f"{name} 不能小于 {min_value}")
    if max_value is not None and out > max_value:
        raise ValueError(f"{name} 不能大于 {max_value}")
    return out


class _MinHintScrollArea(QScrollArea):
    """QScrollArea whose minimum width follows the inner widget's needs.

    默认 QScrollArea 报告的最小宽度是 0（"反正可以滚动"），这会让外层
    QSplitter 把侧栏拖到极窄、压坏内容。这里重写 minimumSizeHint，让它
    返回内部 widget 的最小宽度 + 垂直滚动条宽度，splitter 就会自然把
    分隔条停在"内容刚好完整显示"的位置——不需要硬编码最小宽度。
    """

    def __init__(self, inner_widget, parent=None):
        super().__init__(parent)
        self.setWidgetResizable(True)
        self.setFrameShape(QScrollArea.NoFrame)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOn)
        # ScrollBarAlwaysOn 只保证滚动条 widget 存在；但内容刚好等于视口高时，
        # 滚动条 maximum=0，部分 Qt 主题会把它渲染成灰色细条，看起来像没显示。
        # 通过样式表强制滚动条始终可见、有明确宽度，跟原 Tk 版侧栏观感一致。
        self.setStyleSheet("""
            QScrollBar:vertical {
                background: transparent;
                width: 12px;
                margin: 0;
            }
            QScrollBar::handle:vertical {
                background: rgba(0, 0, 0, 0.35);
                min-height: 24px;
                border-radius: 4px;
            }
            QScrollBar::handle:vertical:hover {
                background: rgba(0, 0, 0, 0.55);
            }
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
                height: 0;
            }
            QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {
                background: transparent;
            }
        """)
        inner_widget.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
        self.setWidget(inner_widget)
        self._inner = inner_widget

    def minimumSizeHint(self):
        from PySide6.QtCore import QSize
        base = super().minimumSizeHint()
        inner_min = self._inner.minimumSizeHint().width()
        # 视口至少容纳内容最小宽 + 滚动条占位 + 2px 余量
        vsb = self.verticalScrollBar().sizeHint().width() if self.verticalScrollBar() else 0
        return QSize(max(base.width(), inner_min + vsb + 2),
                     base.height())

    def sizeHint(self):
        from PySide6.QtCore import QSize
        base = super().sizeHint()
        inner_hint = self._inner.sizeHint().width()
        vsb = self.verticalScrollBar().sizeHint().width() if self.verticalScrollBar() else 0
        return QSize(max(base.width(), inner_hint + vsb + 2),
                     base.height())

    def resizeEvent(self, event):
        """保证内部 widget 至少比视口高 1px。

        ScrollBarAlwaysOn 让滚动条 widget 存在，但当内容刚好等于视口高时
        滚动条 maximum=0，部分主题下 handle 不绘制，看起来像没显示。
        强制 minimumHeight = viewport + 1，让滚动条永远有非零范围、handle
        始终可见——这是"常驻"在视觉上的真正含义。
        """
        super().resizeEvent(event)
        vp_h = self.viewport().height()
        try:
            self._inner.setMinimumHeight(vp_h + 1)
        except Exception:
            pass


def _make_scroll_panel(inner_widget):
    """把任意 widget 包进 _MinHintScrollArea。

    垂直滚动条**始终显示**（ScrollBarAlwaysOn），与原 Tk 版一致——
    按需出现会让侧栏宽度在内容多/少时来回跳，影响布局稳定。
    """
    return _MinHintScrollArea(inner_widget)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(self.core_title())
        # Qt 默认按压反馈受系统主题影响很大，显式给所有按钮一个
        # 高对比度 pressed 状态，让鼠标操作有清晰的即时反馈。
        self.setStyleSheet(BUTTON_FEEDBACK_STYLE)
        self.resize(1400, 900)

        # 状态栏
        self.status_label = QLabel("就绪")
        self.status_bar = QStatusBar()
        self.status_bar.addWidget(self.status_label)
        self.setStatusBar(self.status_bar)

        # core
        self.core = AnnotatorCore(self)
        self.core.set_status_label(self.status_label)
        self.core.on_model_worker_created = self._hook_model_worker
        self.core.on_save_worker_created = self._hook_save_worker
        self.core.on_save_worker_start_failed = self._close_progress
        self.core.on_session_start_aborted = self._on_session_start_aborted

        # 中央 splitter：左侧栏 + 右画布
        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(self._build_sidebar())
        splitter.addWidget(self._build_canvas_area())
        # 禁止折叠到 0（不是禁拖动）。分隔条的最左位置由侧栏内容的
        # minimumSizeHint 决定——内容会告诉 splitter "我最少要多宽"。
        splitter.setCollapsible(0, False)
        splitter.setCollapsible(1, False)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([SIDEBAR_WIDTH, 1400 - SIDEBAR_WIDTH])
        self.setCentralWidget(splitter)
        self.core.on_quit_interaction_lock_changed = (
            self._set_quit_interaction_locked
        )

        # 侧栏内容（依赖 core.canvas 已存在）
        self._build_path_form()
        self._build_param_form()
        self._build_start_button()
        self._build_toolbar()
        self._build_hint()

        # 快捷键：QShortcut 走 WM_KEYDOWN，中文输入法激活时也生效。
        self.core.register_hotkeys(self)

        # 后台任务进度对话框（同一个，按需复用）
        self.progress_dlg = None
        self._progress_worker = None
        # 启动后让画布拥有焦点（快捷键 ApplicationShortcut 之外的第二重保险）
        self.canvas.setFocus()

    @staticmethod
    def core_title():
        return AnnotatorCore.APP_WINDOW_TITLE

    def _set_quit_interaction_locked(self, locked):
        central = self.centralWidget()
        if central is not None:
            central.setEnabled(not locked)

    # ---------- 后台任务进度对话框 ----------
    def _show_progress(
        self,
        label_text,
        cancelable=True,
        max_value=0,
        worker=None,
        modal=True,
    ):
        """显示进度对话框。max_value=0 表示不确定模式（busy）。"""
        if self.progress_dlg is not None:
            try:
                self.progress_dlg.close()
                self.progress_dlg.deleteLater()
            except Exception:
                pass
        dlg = QProgressDialog(label_text, "取消" if cancelable else "", 0,
                              max_value if max_value > 0 else 0, self)
        dlg.setWindowTitle("处理中")
        dlg.setWindowModality(Qt.ApplicationModal if modal else Qt.NonModal)
        dlg.setMinimumDuration(0)
        dlg.setAutoClose(False)
        dlg.setAutoReset(False)
        if max_value <= 0:
            dlg.setBar(QProgressBar())   # 不确定模式：滚动条
        dlg.show()
        self.progress_dlg = dlg
        self._progress_worker = worker
        return dlg

    def _close_progress(self, worker=None):
        if worker is not None and worker is not self._progress_worker:
            return
        if self.progress_dlg is not None:
            try:
                self.progress_dlg.close()
                self.progress_dlg.deleteLater()
            except Exception:
                pass
            self.progress_dlg = None
            self._progress_worker = None

    def _hook_save_worker(self, sw):
        """在当前页或批量保存 worker.start 前挂接进度、取消和结束信号。"""
        if sw is None or getattr(sw, "_ui_hooked", False):
            return
        sw._ui_hooked = True
        dlg = self._show_progress(
            "保存当前页..." if getattr(sw, "_single_page", False) is True else "批量保存中...",
            cancelable=True,
            max_value=max(1, self.core.total_items),
            worker=sw,
            modal=False,
        )
        dlg.canceled.connect(self.core.cancel_save_all)

        def _on_progress(done, total, name):
            if (
                self.progress_dlg is not None
                and self._progress_worker is sw
            ):
                self.progress_dlg.setMaximum(max(1, total))
                self.progress_dlg.setLabelText(
                    f"保存 {done}/{total}：{name}"
                )
                self.progress_dlg.setValue(done)

        sw.progress.connect(_on_progress)
        sw.finished_ok.connect(lambda worker=sw: self._on_save_all_done(worker))
        # 错误内容已由 core._on_worker_error 唯一展示；app 只负责
        # 关闭进度框，避免同一个后台错误弹出两次 QMessageBox。
        sw.error.connect(lambda _msg, worker=sw: self._close_progress(worker))
        sw.stopped.connect(lambda _msg, worker=sw: self._close_progress(worker))

    def _hook_model_worker(self, model_worker):
        """在 ModelLoadWorker.start 前确定地挂接进度 UI。"""
        if model_worker is None or getattr(model_worker, "_ui_hooked", False):
            return
        model_worker._ui_hooked = True
        self._show_progress(
            "正在初始化 OCR 模型...",
            cancelable=False,
            worker=model_worker,
        )
        model_worker.finished_ok.connect(
            lambda worker=model_worker: self._close_progress(worker)
        )
        model_worker.error.connect(
            lambda message, worker=model_worker: self._on_worker_error_close_progress(
                message,
                worker,
            )
        )

    def _on_worker_error_close_progress(self, msg, worker=None):
        """错误框由 core 统一显示；App 只清理进度 UI。"""
        self._close_progress(worker)
        if not self.core.running:
            self.start_button.setEnabled(True)

    def _on_session_start_aborted(self):
        """首次页面取消或加载失败后，允许用户调整设置并重试。"""
        self.start_button.setEnabled(True)

    def _on_save_all_done(self, worker=None):
        self._close_progress(worker)
        # 如果是 quit_app 触发，core._really_quit 已在 _on_save_all_finished 里被调，
        # 这里只负责关闭对话框；若窗口已被 closeEvent 标记要退，QApplication 会自动退出。

    # ---------- 侧栏 ----------
    def _build_sidebar(self):
        v_split = QSplitter(Qt.Vertical)
        self.top_panel = QWidget()
        self.top_panel.setLayout(QVBoxLayout(self.top_panel))
        self.top_panel.layout().setContentsMargins(8, 8, 8, 8)

        self.tools_panel = QWidget()
        self.tools_panel.setLayout(QVBoxLayout(self.tools_panel))
        self.tools_panel.layout().setContentsMargins(8, 8, 8, 8)

        v_split.addWidget(_make_scroll_panel(self.top_panel))
        v_split.addWidget(_make_scroll_panel(self.tools_panel))
        # 上下两段也不允许折叠（addWidget 之后再设，否则 index 越界）。
        v_split.setCollapsible(0, False)
        v_split.setCollapsible(1, False)
        v_split.setStretchFactor(0, 1)
        v_split.setStretchFactor(1, 1)
        v_split.setSizes([500, 400])
        return v_split

    def _build_canvas_area(self):
        self.canvas = AnnotatorCanvas(self.core)
        self.core.attach_canvas(self.canvas)
        return self.canvas

    def _build_path_form(self):
        """路径分组：按钮 + 输入框，垂直独占一行（与原 Tk LabelFrame 一致）。"""
        group = QGroupBox("路径")
        v = QVBoxLayout(group)
        v.setContentsMargins(8, 6, 8, 6)
        v.setSpacing(4)

        self.in_edit = self._make_line_edit()
        self.out_edit = self._make_line_edit()

        btn_in = QPushButton("选择输入文件夹")
        btn_out = QPushButton("选择输出文件夹")
        btn_in.clicked.connect(self._choose_input)
        btn_out.clicked.connect(self._choose_output)

        v.addWidget(btn_in)
        v.addWidget(self.in_edit)
        v.addWidget(btn_out)
        v.addWidget(self.out_edit)

        self.top_panel.layout().addWidget(group)

    def _build_param_form(self):
        """参数分组：每项 label 在上、输入框独占一行。

        故意不用 QFormLayout（左右两列）：原 Tk 版是 ttk.Label + ttk.Entry 上下排，
        label 可以写得很长（"DBNet短边（建议 320–2048）"）也不会被截断或挤窄输入框。
        """
        group = QGroupBox("参数")
        v = QVBoxLayout(group)
        v.setContentsMargins(8, 6, 8, 6)
        v.setSpacing(6)

        self.level_combo = QComboBox()
        self.level_combo.addItems(["char", "word", "line"])
        self.level_combo.setCurrentText("char")
        self.level_combo.currentIndexChanged.connect(lambda *_: self.canvas.setFocus())

        self.size_edit = self._make_line_edit("960")
        self.dpi_edit = self._make_line_edit("200")
        self.img_scale_edit = self._make_line_edit("1.0")
        self.poppler_edit = self._make_line_edit("")

        def _row(label_text, widget):
            """label 在上（左对齐）、widget 在下，独占整行。

            setWordWrap(True)：让 label 在侧栏窄时自动换行，**不再把整行文字
            宽度顶上去当最小宽度**。否则最长 label（"基准缩放 image_scale
            （影响OCR/保存，建议 0.2–3.0）"）会强制侧栏至少 300+ 像素，
            实际根本不需要那么宽。
            """
            lbl = QLabel(label_text)
            lbl.setWordWrap(True)
            lbl.setOpenExternalLinks(False)
            v.addWidget(lbl)
            v.addWidget(widget)

        _row("切分层级 Level", self.level_combo)
        _row("DBNet短边（建议 320–2048）", self.size_edit)
        _row("PDF DPI（建议 100–400）", self.dpi_edit)
        _row("OCR 缩放（0.05–1.0，原图和切片不变）", self.img_scale_edit)
        _row("poppler_path（可选，仅 pdf2image 需要）", self.poppler_edit)

        self.top_panel.layout().addWidget(group)

    def _make_line_edit(self, text="") -> QLineEdit:
        """构造一个 QLineEdit，回车后自动把焦点还给画布。

        关键：QLineEdit 获焦时，字母键会被它当作文本输入而不触发
        QShortcut（即便 ApplicationShortcut 也不行）。所以约定用户敲完
        回车 / 点别处后，焦点自动回归画布，快捷键立刻恢复。
        """
        edit = QLineEdit(text)
        edit.setClearButtonEnabled(True)
        edit.returnPressed.connect(lambda: self.canvas.setFocus())
        return edit

    def _build_start_button(self):
        self.start_button = QPushButton("开始")
        self.start_button.setSizePolicy(
            QSizePolicy.Expanding,
            QSizePolicy.Fixed,
        )
        self.start_button.setMinimumHeight(34)
        self.start_button.setStyleSheet(
            "QPushButton { font-weight: bold; padding: 4px; }"
        )
        self.start_button.clicked.connect(self._start)
        self.top_panel.layout().addWidget(self.start_button)

    def _build_toolbar(self):
        title = QLabel("工具栏（等同快捷键）")
        title.setStyleSheet("font-weight: bold;")
        self.tools_panel.layout().addWidget(title)

        grid_host = QWidget()
        grid = QGridLayout(grid_host)
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setHorizontalSpacing(6)
        grid.setVerticalSpacing(6)
        # 两列均分剩余宽度；按钮自己的最小宽度仍由内容决定（setMinimumWidth=0），
        # 避免某个按钮文字特别长（如"删除框 (x/Delete)"）把整列拉宽、压坏另一列。
        grid.setColumnStretch(0, 1)
        grid.setColumnStretch(1, 1)

        buttons = [
            ("创建模式 (c)",   self.core.toggle_create_mode),
            ("旋转模式 (r)",   self.core.toggle_rotate_mode),
            ("导航模式 (O)",   self.core.toggle_nav_mode),
            ("放大 (+)",       lambda: self.core.zoom_center(1.25)),
            ("缩小 (-)",       lambda: self.core.zoom_center(0.8)),
            ("适配窗口 (Fit)", self.core.fit_to_window),
            ("1:1",            self.core.one_to_one),
            ("上一项 (a)",     self.core.prev_item),
            ("下一项 (d)",     self.core.next_item),
            ("上一个框 (b)",   self.core.prev_box),
            ("下一个框 (n)",   self.core.next_box),
            ("删除框 (x/Delete)", self.core.delete_box),
            ("保存当前 (s)",   self.core.save_current),
            ("全部保存",       self.core.save_all),
            ("退出 (Esc/q)",   self.core.quit_app),
        ]
        for i, (txt, cb) in enumerate(buttons):
            r, c = divmod(i, 2)
            b = self._make_tool_button(txt)
            b.clicked.connect(cb)
            grid.addWidget(b, r, c)

        self.tools_panel.layout().addWidget(grid_host)
        # 不再加 addStretch(1)：原来这行把工具栏顶到上沿、hint 被挤很远；
        # 现在让 hint 紧跟工具栏，整列自然顺排。

    def _make_tool_button(self, text: str) -> QPushButton:
        """工具栏按钮：按内容自适应宽度，不被网格强行等宽拉宽/压窄。

        - Minimum 水平策略：按钮只取内容所需宽度，剩余空间让给同列其它按钮；
        - setMinimumWidth(0)：解除 QPushButton 默认 75px 最小宽限制；
        - 固定高度 + padding：触摸友好，列表视觉对齐。
        """
        b = QPushButton(text)
        b.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Fixed)
        b.setMinimumWidth(0)
        b.setMinimumHeight(30)
        b.setFixedHeight(30)
        # 内边距：紧贴文字，不让按钮看起来空荡；同时保留触摸友好高度
        b.setStyleSheet("QPushButton { padding: 2px 4px; }")
        return b

    def _build_hint(self):
        hint = QLabel("说明：放大/缩小只影响界面显示，不影响最终切片清晰度。")
        hint.setWordWrap(True)
        hint.setStyleSheet("color: #555555;")
        self.tools_panel.layout().addWidget(hint)

    # ---------- 交互回调 ----------
    def _choose_input(self):
        p = QFileDialog.getExistingDirectory(self, "选择输入文件夹")
        if p:
            self.in_edit.setText(p)

    def _choose_output(self):
        p = QFileDialog.getExistingDirectory(self, "选择输出文件夹")
        if p:
            self.out_edit.setText(p)

    def _start(self):
        if self.core._quit_state != self.core.QUIT_IDLE:
            return
        input_dir = self.in_edit.text().strip() or self.core.input_dir
        output_dir = self.out_edit.text().strip() or self.core.output_dir
        if not input_dir or not os.path.isdir(input_dir):
            QMessageBox.critical(self, "错误", "请先选择有效的输入文件夹")
            return
        if not output_dir:
            QMessageBox.critical(self, "错误", "请先选择输出文件夹")
            return
        try:
            level = self.level_combo.currentText().strip().lower()
            if level not in ("char", "word", "line"):
                raise ValueError("Level 只能选择 char、word 或 line")
            short_side = _parse_int(self.size_edit.text(), "DBNet短边", 64, 6000)
            pdf_dpi = _parse_int(self.dpi_edit.text(), "PDF DPI", 50, 800)
            image_scale = _parse_float(self.img_scale_edit.text(), "OCR 缩放", 0.05, 1.0)
        except Exception as e:
            QMessageBox.critical(self, "参数错误", str(e))
            return
        # Capture new settings without mutating the old session's save paths.
        settings = (input_dir, output_dir, level, short_side, pdf_dpi,
                    image_scale, self.poppler_edit.text().strip())
        if self.core.running or self.core.any_worker_running():
            self.core.request_restart(lambda: self._start_session(settings))
        else:
            self._start_session(settings)

    def _start_session(self, settings):
        input_dir, output_dir, *params = settings
        self.core.set_input_dir(input_dir)
        self.core.set_output_dir(output_dir)
        self.core.set_params(*params)
        if not self.core.prepare_items():
            return
        if not self.core.start():
            QMessageBox.warning(self, "无法开始", "当前状态不允许启动新会话")
            return
        # Like the original UI, Start remains available to apply new settings.
        self.start_button.setEnabled(True)
        self.canvas.setFocus()

    # ---------- 关闭 ----------
    def closeEvent(self, event):
        """所有关闭入口统一交给 core 的异步退出状态机。"""
        if self.core.can_close_window():
            event.accept()
            return
        event.ignore()
        self.core.quit_app()

    def _any_worker_running(self):
        return self.core.any_worker_running()


def main():
    app = QApplication(sys.argv)
    # PySide6 6.5+ 默认启用高 DPI 缩放，无需手动 setHighDpiScaleFactorRoundingPolicy。
    w = MainWindow()
    w.showMaximized()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
