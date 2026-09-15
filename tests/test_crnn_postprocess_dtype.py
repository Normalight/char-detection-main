# -*- coding: utf-8 -*-
"""CRNN _postprocess dtype 回归测试。

历史上 reverse_rotate_crop_image 用 `points = np.float32(points)` 复用调用方数组，
当 bbox 已是 float32 时不复制，后续 `points[:, 0] -= left` 直接改坏调用方 bbox。
表现：用 float32 bbox 调用 predict_rbg 后，所有字框都堆在图像原点（y=0-行高）。

int16 / float64 输入因 dtype 不同恰好触发了 np.float32 的复制路径，掩盖了这个 bug，
导致 bench/compare_e2e.py（float32 bbox）和生产 cutbatch（int16 bbox）行为不一致。

本测试不加载 ONNX 模型，直接对 reverse_rotate_crop_image 验证两点：
  1) 函数不修改入参 points
  2) 同一 bbox 多次调用结果一致（第一次和第二次必须相同）
"""

import unittest

import numpy as np
from PIL import Image

from crnn.CRNN import CRNNHandle, reverse_rotate_crop_image


class ReverseRotateCropImageDtypeTest(unittest.TestCase):
    def _bbox_horizontal(self, dtype):
        return np.array([[26, 455], [619, 455], [619, 479], [26, 479]], dtype=dtype)

    def _bbox_vertical(self, dtype):
        # 竖排：高/宽 >= 1.5 会走 Srotate(-90) 分支
        return np.array([[100, 100], [124, 100], [124, 460], [100, 460]], dtype=dtype)

    def test_does_not_mutate_int16_bbox(self):
        bbox = self._bbox_horizontal(np.int16)
        before = bbox.copy()
        reverse_rotate_crop_image(None, None, bbox, [[0, 0], [0, 24]], (0, 0))
        np.testing.assert_array_equal(bbox, before)

    def test_does_not_mutate_float32_bbox(self):
        """历史 bug 现场：float32 bbox 被原地修改。"""
        bbox = self._bbox_horizontal(np.float32)
        before = bbox.copy()
        reverse_rotate_crop_image(None, None, bbox, [[0, 0], [0, 24]], (0, 0))
        np.testing.assert_array_equal(bbox, before)

    def test_does_not_mutate_float64_bbox(self):
        bbox = self._bbox_horizontal(np.float64)
        before = bbox.copy()
        reverse_rotate_crop_image(None, None, bbox, [[0, 0], [0, 24]], (0, 0))
        np.testing.assert_array_equal(bbox, before)

    def test_repeated_calls_return_same_result_float32(self):
        """同一 bbox 连续调用两次，第二次必须返回和第一次相同的结果。

        历史 bug：第一次调用改坏了 bbox，第二次拿到的已经是局部坐标，
        所有字框全堆到 (0, 0) 附近。
        """
        bbox = self._bbox_horizontal(np.float32)
        out1 = reverse_rotate_crop_image(None, None, bbox, [[6, 0], [6, 24]], (0, 0))
        out2 = reverse_rotate_crop_image(None, None, bbox, [[6, 0], [6, 24]], (0, 0))
        self.assertEqual(out1, out2)
        # 横排第一行 bbox left=26 top=455，crop_left=6 反算后 x 应在 26+6=32 附近
        self.assertAlmostEqual(out1[0][0], 32, delta=2)
        self.assertAlmostEqual(out1[0][1], 455, delta=2)

    def test_dtype_invariance_horizontal(self):
        """int16 / float32 / float64 输入必须得到相同结果（横排）。"""
        part_points = [[6, 0], [6, 24]]
        out_int = reverse_rotate_crop_image(
            None, None, self._bbox_horizontal(np.int16), part_points, (0, 0))
        out_f32 = reverse_rotate_crop_image(
            None, None, self._bbox_horizontal(np.float32), part_points, (0, 0))
        out_f64 = reverse_rotate_crop_image(
            None, None, self._bbox_horizontal(np.float64), part_points, (0, 0))
        self.assertEqual(out_int, out_f32)
        self.assertEqual(out_int, out_f64)

    def test_dtype_invariance_vertical(self):
        """竖排（走 Srotate 分支）也必须 dtype 无关。"""
        part_points = [[0, 0], [0, 30]]
        out_int = reverse_rotate_crop_image(
            None, None, self._bbox_vertical(np.int16), part_points, (0, 0))
        out_f32 = reverse_rotate_crop_image(
            None, None, self._bbox_vertical(np.float32), part_points, (0, 0))
        self.assertEqual(out_int, out_f32)


class CrnnConfidenceTests(unittest.TestCase):
    def test_postprocess_adds_character_and_line_confidence(self):
        handle = object.__new__(CRNNHandle)
        preds = np.asarray(
            [
                [0.0, 4.0, 0.0],
                [4.0, 0.0, 0.0],
                [0.0, 0.0, 2.0],
            ],
            dtype=np.float32,
        )
        crop = Image.new("RGB", (12, 24), "white")
        bbox = np.asarray(
            [[0, 0], [12, 0], [12, 24], [0, 24]],
            dtype=np.float32,
        )

        result = handle._postprocess(
            preds,
            scale=1.0,
            w=12,
            crop_im=crop,
            bbox=bbox,
        )

        self.assertEqual(len(result["char_list"]), 2)
        self.assertTrue(all(
            0.0 < item["score"] <= 1.0
            for item in result["char_list"]
        ))
        self.assertAlmostEqual(
            result["score"],
            float(np.mean([
                item["score"] for item in result["char_list"]
            ])),
        )

    def test_probability_output_is_not_softmaxed_twice(self):
        handle = object.__new__(CRNNHandle)
        preds = np.asarray(
            [[0.05, 0.9, 0.05]],
            dtype=np.float32,
        )
        crop = Image.new("RGB", (8, 8), "white")
        bbox = np.asarray(
            [[0, 0], [8, 0], [8, 8], [0, 8]],
            dtype=np.float32,
        )

        result = handle._postprocess(
            preds,
            scale=1.0,
            w=8,
            crop_im=crop,
            bbox=bbox,
        )

        self.assertAlmostEqual(result["char_list"][0]["score"], 0.9)


if __name__ == "__main__":
    unittest.main()
