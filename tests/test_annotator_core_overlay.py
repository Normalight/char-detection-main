import inspect
import unittest

from annotator_core import AnnotatorCore


class AnnotatorCoreOverlayTests(unittest.TestCase):
    """原 _redraw 已删除；Qt 画布 canvas_widget._draw_boxes 接手绘制。

    继续保持"画布代码里不得写死 CREATE/ROTATE/NAV MODE 文字"的历史不变量，
    但检查目标从 AnnotatorCore._redraw 换成 AnnotatorCanvas._draw_boxes，
    并顺手确认 annotator_core 里确实已经不存在 _redraw 方法（防止误回滚）。
    """

    def test_redraw_method_is_gone(self):
        self.assertFalse(
            hasattr(AnnotatorCore, "_redraw"),
            "AnnotatorCore._redraw 已废弃，不应再存在（Qt 画布接管绘制）",
        )
        self.assertFalse(
            hasattr(AnnotatorCore, "_mouse_cb"),
            "AnnotatorCore._mouse_cb 已废弃，不应再存在（Qt 鼠标事件接管）",
        )

    def test_mouse_handlers_exist(self):
        # Qt 版拆出来的三个入口必须存在
        for name in ("_on_mouse_left_down", "_on_mouse_move", "_on_mouse_left_up", "_on_wheel"):
            self.assertTrue(hasattr(AnnotatorCore, name), f"缺少 {name}")

    def test_visual_state_actions_request_repaint(self):
        """Qt 没有旧 cv2 的循环重绘，状态变化必须主动刷新画布。"""
        for name in (
            "toggle_create_mode",
            "toggle_rotate_mode",
            "toggle_nav_mode",
            "prev_box",
            "next_box",
            "delete_box",
        ):
            src = inspect.getsource(getattr(AnnotatorCore, name))
            self.assertIn("_refresh_canvas()", src, f"{name} 应立即刷新画布")

    def test_window_title_keeps_item_and_zoom_context(self):
        src = inspect.getsource(AnnotatorCore._update_window_title)
        self.assertIn("current_item_title", src)
        self.assertIn("current_item_index", src)
        self.assertIn("view_scale", src)


if __name__ == "__main__":
    unittest.main()
