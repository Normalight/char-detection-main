# -*- coding: utf-8 -*-
"""canvas_widget 的静态/语义回归。

只测源码层面的不变量，避免：
- 启动真实 QApplication（CI 可能没显示环境）
- 触发 annotator_core → ocr_pipeline → onnxruntime 依赖链
"""

import inspect
import unittest

import ui.canvas as canvas_widget
class CanvasWidgetSourceTests(unittest.TestCase):
    def test_paint_event_does_not_render_mode_labels(self):
        """历史 bug：原 cv2 _redraw 在画布写 CREATE/ROTATE/NAV MODE 文字。

        Qt 版 paintEvent / _draw_boxes 不得调用 drawText 画这类大字状态提示。
        注意：drawText 仍可用于画 handle 编号（0/1/2/3），所以不能粗暴断言
        "整个方法源码里没有 NAV MODE"——注释里提到历史 bug 是允许的。
        检查目标限定为：禁止把 "MODE" 作为 drawText 的字面量参数。
        """
        for method_name in ("paintEvent", "_draw_boxes"):
            src = inspect.getsource(getattr(canvas_widget.AnnotatorCanvas, method_name))
            # drawText 调用里若出现 *_MODE，就是回退到了画状态文字的旧习惯。
            for needle in ('"CREATE MODE"', '"ROTATE MODE"', '"NAV MODE"',
                           "'CREATE MODE'", "'ROTATE MODE'", "'NAV MODE'"):
                self.assertNotIn(needle, src, f"{method_name} 不应画 {needle}")

    def test_required_event_handlers_exist(self):
        for name in (
            "paintEvent", "resizeEvent", "mousePressEvent",
            "mouseMoveEvent", "mouseReleaseEvent", "wheelEvent",
        ):
            self.assertTrue(
                hasattr(canvas_widget.AnnotatorCanvas, name),
                f"AnnotatorCanvas 缺少事件处理器 {name}",
            )

    def test_mouse_tracking_and_focus_set_in_init(self):
        """setMouseTracking(True) 让无按键 mouseMove 也触发；StrongFocus 让画布能收键盘。"""
        src = inspect.getsource(canvas_widget.AnnotatorCanvas.__init__)
        self.assertIn("setMouseTracking(True)", src)
        self.assertIn("StrongFocus", src)

    def test_qimage_cache_fingerprint_is_content_based(self):
        """QImage 缓存指纹必须基于**内容**，不能用 id()。

        id() 是 CPython 内存地址，view_image 被重新赋值后旧地址可能被新对象
        复用，缓存会误命中、画错图。指纹应至少包含 shape + view_version。
        """
        self.assertTrue(hasattr(canvas_widget.AnnotatorCanvas, "_fingerprint"))
        src = inspect.getsource(canvas_widget.AnnotatorCanvas._fingerprint)
        # 只看实际语句，不检查 docstring/注释（docstring 会提到 id() 的禁用理由）
        code_lines = [ln for ln in src.splitlines()
                      if ln.strip() and not ln.strip().startswith("#")
                      and '"""' not in ln]
        code = "\n".join(code_lines)
        self.assertNotIn("id(view_bgr)", code,
                         "指纹不得用 id(view_bgr)，地址会被复用")
        # 必须用到 view_version + shape
        self.assertIn("_view_version", code)
        self.assertIn("shape", code)

    def test_mouse_move_has_throttle(self):
        """mouseMoveEvent 必须节流，否则高 DPI 屏 / 触摸板每像素触发一次重绘会卡。"""
        src = inspect.getsource(canvas_widget.AnnotatorCanvas.mouseMoveEvent)
        # 出现时间戳比较或节流常量引用即可
        self.assertTrue(
            "_MOVE_THROTTLE_SEC" in src or "_last_move_ts" in src,
            "mouseMoveEvent 应实现节流（_MOVE_THROTTLE_SEC / _last_move_ts）",
        )

    def test_wheel_event_accepts(self):
        """wheelEvent 必须 accept，否则 Windows 触摸板会把滚动手势重复投递。"""
        src = inspect.getsource(canvas_widget.AnnotatorCanvas.wheelEvent)
        self.assertIn("event.accept()", src)

    def test_paint_event_enables_antialiasing(self):
        """斜框和圆形 handle 应保留旧 cv2 LINE_AA 的平滑边缘。"""
        src = inspect.getsource(canvas_widget.AnnotatorCanvas.paintEvent)
        self.assertIn("setRenderHint", src)
        self.assertIn("Antialiasing", src)

    def test_focus_in_grabs_keyboard(self):
        """focusInEvent 应 grabKeyboard，让画布拥有焦点时快捷键稳定生效。"""
        src = inspect.getsource(canvas_widget.AnnotatorCanvas.focusInEvent)
        self.assertIn("grabKeyboard", src)
        # 配对 release
        src_out = inspect.getsource(canvas_widget.AnnotatorCanvas.focusOutEvent)
        self.assertIn("releaseKeyboard", src_out)


if __name__ == "__main__":
    unittest.main()
