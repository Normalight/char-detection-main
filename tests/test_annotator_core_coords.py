# -*- coding: utf-8 -*-
"""annotator_core 坐标变换单元测试。

只测纯数学方法（_base_to_screen / _screen_to_base / _fit_scale_for_current_base /
_zoom_at_screen），不测任何依赖 Tk 或 cv2 窗口的方法。通过 object.__new__
绕过 __init__（它需要 tk_root），手动设置需要的属性。
"""

import unittest

import numpy as np

from annotation.controller import AnnotatorCore


def _make_core(view_w=400, view_h=300):
    """构造一个不依赖 Tk 的 AnnotatorCore 测试实例。"""
    core = object.__new__(AnnotatorCore)
    core.VIEW_W = view_w
    core.VIEW_H = view_h
    core.view_scale = 1.0
    core.MIN_VIEW_SCALE = 0.05
    core.MAX_VIEW_SCALE = 12.0
    core.viewport_offset = np.array([0, 0], dtype=np.int32)
    core.draw_origin = np.array([0, 0], dtype=np.int32)
    core.base_size = (0, 0)
    core.base_image_bgr = None
    core.view_image_bgr = None
    core.items = []
    core.current_item_index = -1
    # set_status / _mark_view_state 等方法会访问这些属性
    core.status_label = None
    return core


def _set_image(core, w, h, view_scale=1.0):
    """注入一张虚拟图（全黑）和对应缩放。"""
    core.base_image_bgr = np.zeros((h, w, 3), dtype=np.uint8)
    core.base_size = (w, h)
    new_w = max(1, int(round(w * view_scale)))
    new_h = max(1, int(round(h * view_scale)))
    core.view_image_bgr = np.zeros((new_h, new_w, 3), dtype=np.uint8)
    core.view_scale = view_scale


class BaseToScreenTests(unittest.TestCase):
    def test_identity_scale_small_image(self):
        # 图小于窗口：图像居中显示，坐标加上 draw_origin 偏移
        core = _make_core(view_w=400, view_h=300)
        _set_image(core, w=100, h=80, view_scale=1.0)
        # 手动模拟 _compute_draw_and_clamp 的居中行为
        core.draw_origin = np.array([(400 - 100) // 2, (300 - 80) // 2], dtype=np.int32)
        sx, sy = core._base_to_screen((10, 20))
        self.assertEqual(sx, 10 + (400 - 100) // 2)
        self.assertEqual(sy, 20 + (300 - 80) // 2)

    def test_scale_2x_doubles_coordinates(self):
        core = _make_core(view_w=400, view_h=300)
        _set_image(core, w=100, h=80, view_scale=2.0)
        # 图放大到 200x160，仍小于窗口 400x300 -> 走居中分支
        core.draw_origin = np.array([0, 0], dtype=np.int32)
        sx, sy = core._base_to_screen((10, 20))
        self.assertEqual(sx, 20)
        self.assertEqual(sy, 40)


class ScreenToBaseTests(unittest.TestCase):
    def test_returns_none_when_no_image(self):
        core = _make_core()
        self.assertIsNone(core._screen_to_base(10, 10))

    def test_returns_none_when_scale_zero(self):
        core = _make_core()
        _set_image(core, w=100, h=100, view_scale=0.0)
        self.assertIsNone(core._screen_to_base(10, 10))

    def test_round_trip_small_image(self):
        # 图 < 窗口：base -> screen -> base 应能还原
        core = _make_core(view_w=400, view_h=300)
        _set_image(core, w=100, h=80, view_scale=1.0)
        core.draw_origin = np.array([10, 20], dtype=np.int32)

        base_pt = (15, 25)
        sx, sy = core._base_to_screen(base_pt)
        back = core._screen_to_base(sx, sy)
        self.assertIsNotNone(back)
        self.assertAlmostEqual(back[0], base_pt[0], places=0)
        self.assertAlmostEqual(back[1], base_pt[1], places=0)

    def test_returns_none_for_point_outside_image(self):
        core = _make_core(view_w=400, view_h=300)
        _set_image(core, w=100, h=80, view_scale=1.0)
        core.draw_origin = np.array([0, 0], dtype=np.int32)
        # 屏幕点 (200, 200) 在窗口内但图只有 100x80
        self.assertIsNone(core._screen_to_base(200, 200))


class FitScaleTests(unittest.TestCase):
    def test_returns_1_when_image_already_small(self):
        core = _make_core(view_w=400, view_h=300)
        _set_image(core, w=100, h=80, view_scale=1.0)
        # 图小于窗口 -> fit 后仍是 1.0（被 min(..., 1.0) 钳制）
        scale = core._fit_scale_for_current_base()
        self.assertAlmostEqual(scale, 1.0)

    def test_returns_fit_ratio_when_image_larger(self):
        core = _make_core(view_w=400, view_h=300)
        _set_image(core, w=800, h=800, view_scale=1.0)
        # 800x800 图放进 400x300 窗口：取较短边约束 min(400/800, 300/800, 1.0) = 0.375
        scale = core._fit_scale_for_current_base()
        self.assertAlmostEqual(scale, 0.375, places=2)

    def test_returns_1_when_no_image(self):
        core = _make_core()
        self.assertEqual(core._fit_scale_for_current_base(), 1.0)


class ZoomAtScreenTests(unittest.TestCase):
    def test_zoom_in_increases_scale(self):
        core = _make_core(view_w=400, view_h=300)
        _set_image(core, w=800, h=800, view_scale=0.5)
        # 在屏幕中心放大 1.25 倍
        core._zoom_at_screen(1.25, (200, 150))
        self.assertGreater(core.view_scale, 0.5)

    def test_zoom_clamped_to_max(self):
        core = _make_core(view_w=400, view_h=300)
        _set_image(core, w=100, h=100, view_scale=11.0)
        # 已经接近上限 12.0，再放大应被钳制
        core._zoom_at_screen(10.0, (50, 50))
        self.assertLessEqual(core.view_scale, 12.0)


class SetViewScaleTests(unittest.TestCase):
    def test_clamps_to_min(self):
        core = _make_core(view_w=400, view_h=300)
        _set_image(core, w=100, h=100, view_scale=1.0)
        core._set_view_scale(0.001)
        self.assertGreaterEqual(core.view_scale, 0.05)

    def test_clamps_to_max(self):
        core = _make_core(view_w=400, view_h=300)
        _set_image(core, w=100, h=100, view_scale=1.0)
        core._set_view_scale(999.0)
        self.assertLessEqual(core.view_scale, 12.0)


if __name__ == "__main__":
    unittest.main()
