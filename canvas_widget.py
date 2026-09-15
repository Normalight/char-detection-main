# -*- coding: utf-8 -*-
"""canvas_widget.py

Qt 画布，替代原 cv2.imshow + annotator_core._redraw + annotator_core._mouse_cb。

设计要点：
1. 本 widget 只负责"绘制"和"鼠标事件路由"，状态/逻辑全部留在 AnnotatorCore；
2. QImage 做**内容指纹**缓存（不用 id()，避免 CPython 地址复用导致误命中）；
3. setMouseTracking(True) 让 mouseMoveEvent 不必按住按键也触发（对应 cv2 EVENT_MOUSEMOVE）；
4. mouseMove 节流：~16ms（≈60fps）一次回调，拖框 / 平移时避免每像素触发重绘；
5. wheelEvent.accept()：防止 Windows 触摸板把滚动手势重复投递；
6. 颜色常量在 RGB 空间定义（QColor 用 RGB），不要和 annotator_core 里的 BGR 元组混用。
"""

import logging
import time
from typing import Optional

import cv2
import numpy as np
from PySide6.QtCore import Qt, QPoint, QPointF, QSize, QRect, QRectF
from PySide6.QtGui import QColor, QImage, QPainter, QPen, QPolygonF
from PySide6.QtWidgets import QWidget

logger = logging.getLogger(__name__)

# RGB（QColor 空间）。annotator_core 里的 COLOR_* 是 BGR，不要直接拿来用。
_RGB_UNSELECTED = QColor(0, 150, 255)     # 橙
_RGB_SELECTED = QColor(0, 255, 0)         # 绿
_RGB_HANDLE = QColor(255, 0, 0)           # 红
_HANDLE_RADIUS = 8                         # 与原 annotator_core.HANDLE_RADIUS 一致

# mouseMove 节流间隔（秒）。人眼 60fps 足够，过密会和高 DPI 屏打架。
_MOVE_THROTTLE_SEC = 0.016


class AnnotatorCanvas(QWidget):
    """标注画布。

    用法：AnnotatorCore.attach_canvas(canvas) 后，core 会通过 _refresh_canvas
    触发本 widget 重绘；鼠标/滚轮事件本 widget 回调 core 的对应方法。
    """

    def __init__(self, core, parent=None):
        super().__init__(parent)
        self.core = core
        self.setMinimumSize(400, 300)
        self.setMouseTracking(True)        # 无按键 mouseMove 也触发
        self.setFocusPolicy(Qt.StrongFocus)
        self.setAttribute(Qt.WA_OpaquePaintEvent, True)
        self.setAutoFillBackground(False)

        # view_image → QImage 缓存：内容指纹 = (shape, dtype, version, bytes 头部 hash)
        self._qimg: Optional[QImage] = None
        self._qimg_fingerprint = None

        # mouseMove 节流
        self._last_move_ts = 0.0
        self._pending_move = None   # 缓存的最后一次 (x, y, alt)，下次节流窗打开时回放

    # ---------- Qt 标准事件 ----------
    def sizeHint(self) -> QSize:
        return QSize(1280, 720)

    def resizeEvent(self, event):
        # 原 _on_viewer_resize：通知 core 画布大小变了。
        self.core.VIEW_W = max(1, self.width())
        self.core.VIEW_H = max(1, self.height())
        self.core._compute_draw_and_clamp()
        self.update()
        super().resizeEvent(event)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.fillRect(self.rect(), Qt.black)
        # 旧 cv2 画布的 LINE_AA 会平滑旋转框、斜线和圆形 handle；
        # Qt 需显式开启对应的矢量抗锯齿。
        painter.setRenderHint(QPainter.Antialiasing, True)

        view = self.core.view_image_bgr
        if view is None:
            return

        self.core._compute_draw_and_clamp()
        H, W = view.shape[:2]
        qimg = self._ensure_qimage(view)

        if W <= self.core.VIEW_W and H <= self.core.VIEW_H:
            # 图小于视口：居中贴
            dx = int(self.core.draw_origin[0])
            dy = int(self.core.draw_origin[1])
            painter.drawImage(dx, dy, qimg)
        else:
            # 图大于视口：只画 viewport_offset 起的可见部分
            vx = int(self.core.viewport_offset[0])
            vy = int(self.core.viewport_offset[1])
            w_roi = min(self.core.VIEW_W, W - vx)
            h_roi = min(self.core.VIEW_H, H - vy)
            if w_roi > 0 and h_roi > 0:
                src = QRect(vx, vy, w_roi, h_roi)
                painter.drawImage(QPoint(0, 0), qimg, src)

        self._draw_boxes(painter)

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            alt = bool(event.modifiers() & Qt.AltModifier)
            p = event.position().toPoint()
            self.core._on_mouse_left_down(p.x(), p.y(), alt)
            # 按下后立即 grab 键盘焦点，保证后续快捷键稳定生效
            self.setFocus()
            event.accept()

    def mouseMoveEvent(self, event):
        alt = bool(event.modifiers() & Qt.AltModifier)
        p = event.position().toPoint()
        x, y = p.x(), p.y()
        now = time.monotonic()
        # 节流：第一次立即回放；之后每 _MOVE_THROTTLE_SEC 至少一次。
        # 末次缓存的 move 必须被回放（保证 release 前落点准确）。
        if self._last_move_ts == 0.0 or now - self._last_move_ts >= _MOVE_THROTTLE_SEC:
            self.core._on_mouse_move(x, y, alt)
            self._last_move_ts = now
            self._pending_move = None
        else:
            self._pending_move = (x, y, alt)
        event.accept()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton:
            # 先把节流期间残留的最后一帧 move 回放，确保 release 时位置准确
            if self._pending_move is not None:
                x, y, alt = self._pending_move
                self.core._on_mouse_move(x, y, alt)
                self._pending_move = None
                self._last_move_ts = time.monotonic()
            p = event.position().toPoint()
            self.core._on_mouse_left_up(p.x(), p.y())
            event.accept()

    def wheelEvent(self, event):
        # 只处理离散 angleDelta；像素 deltas（某些触摸板）忽略避免重复。
        delta = event.angleDelta().y()
        if delta == 0:
            event.ignore()
            return
        p = event.position().toPoint()
        if delta > 0:
            self.core._on_wheel(1.25, p.x(), p.y())
        else:
            self.core._on_wheel(0.8, p.x(), p.y())
        event.accept()

    def focusInEvent(self, event):
        # 画布获得焦点时 grabKeyboard，让 IME 下字母键也能被 QShortcut 捕获
        # （ApplicationShortcut 已经够用，这里只是双保险）。
        self.grabKeyboard()
        super().focusInEvent(event)

    def focusOutEvent(self, event):
        self.releaseKeyboard()
        super().focusOutEvent(event)

    # ---------- 内部 ----------
    def _fingerprint(self, view_bgr: np.ndarray):
        """计算内容指纹。不用 id()——CPython 内存地址会被复用，缓存会误命中。

        采用 (shape, dtype, version, 头部 N 字节 hash)：
        - shape / dtype 区分尺寸/类型；
        - version 区分"同一张图被 in-place 改了"（虽然本项目不这么做，但保险）；
        - 头部 hash 区分"内容真的不同"（用首行少量字节，O(1) 代价）。
        """
        h, w = view_bgr.shape[:2]
        # 取首行首通道前 32 字节做指纹；空数组退化为 0。
        if view_bgr.size > 0:
            sample = bytes(view_bgr[0, :min(w, 32)].tobytes())
        else:
            sample = b""
        return (view_bgr.shape, view_bgr.dtype,
                getattr(self.core, "_view_version", 0),
                hash(sample))

    def _ensure_qimage(self, view_bgr: np.ndarray) -> QImage:
        """BGR numpy → QImage (RGB888)，带内容指纹缓存。"""
        fp = self._fingerprint(view_bgr)
        if fp == self._qimg_fingerprint and self._qimg is not None:
            return self._qimg

        rgb = cv2.cvtColor(view_bgr, cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]
        # bytesPerLine 显式给，避免 QImage 默认 4 字节对齐导致斜图
        qimg = QImage(rgb.data, w, h, rgb.strides[0], QImage.Format_RGB888).copy()
        # .copy() 深拷贝像素数据，rgb 局部变量释放后 QImage 仍安全
        self._qimg = qimg
        self._qimg_fingerprint = fp
        return qimg

    def _draw_boxes(self, painter: QPainter):
        """画所有 box；选中框额外画 4 个 handle 和编号。

        与原 _redraw 保持一致：不在画布上画 CREATE/ROTATE/NAV MODE 文字（历史 bug）。
        """
        if not self.core.boxes:
            return

        for i, box in enumerate(self.core.boxes):
            is_sel = (i == self.core.selected_box_index)
            color = _RGB_SELECTED if is_sel else _RGB_UNSELECTED
            pts = [self.core._base_to_screen(p) for p in box]
            poly = QPolygonF([QPointF(x, y) for x, y in pts])

            pen = QPen(color, 2)
            pen.setCosmetic(True)
            painter.setPen(pen)
            painter.setBrush(Qt.NoBrush)
            painter.drawPolygon(poly)

            if is_sel and not self.core.creation_mode:
                painter.setPen(QColor(255, 255, 255))
                painter.setBrush(_RGB_HANDLE)
                for j, (sx, sy) in enumerate(pts):
                    painter.drawEllipse(QPointF(sx, sy), _HANDLE_RADIUS, _HANDLE_RADIUS)
                    painter.drawText(QPoint(int(sx) - 10, int(sy) - 10), str(j))
