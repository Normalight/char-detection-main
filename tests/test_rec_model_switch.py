# -*- coding: utf-8 -*-
"""识别模型切换（rec_model 开关）回归测试。

验证 cutbatch._build_rec_handle 按 config.rec_model 正确派发到对应的 handle 类，
并拦截非法取值。不需要真的 ONNX 模型文件——通过 monkey-patch handle 构造函数。
"""

import os
import unittest
from unittest.mock import patch

import config
import cutbatch


class RecModelSwitchTest(unittest.TestCase):
    def setUp(self):
        # 备份 config 里相关字段
        self._orig_rec_model = getattr(config, "rec_model", "crnn")
        self._orig_cutbatch_rec_model = getattr(cutbatch, "rec_model", "crnn")
        self._orig_char_box_strategy = getattr(
            cutbatch, "char_box_strategy", "ctc-midpoint"
        )

    def tearDown(self):
        config.rec_model = self._orig_rec_model
        cutbatch.rec_model = self._orig_cutbatch_rec_model
        cutbatch.char_box_strategy = self._orig_char_box_strategy

    def test_default_is_v6_small(self):
        """config.py 顶层默认 rec_model 必须是 v6-small（用户决策：替换 CRNN）。"""
        self.assertEqual(config.rec_model, "v6-small",
                         "config.rec_model 默认值变了；如非故意请改回 v6-small")

    def test_crnn_dispatch(self):
        """rec_model='crnn' 时返回 CRNNHandle，name='CRNN'。"""
        cutbatch.rec_model = "crnn"
        with patch("cutbatch.CRNNHandle") as mock_crnn:
            mock_crnn.return_value = "MOCK_CRNN_HANDLE"
            handle, name = cutbatch._build_rec_handle()
        mock_crnn.assert_called_once()
        self.assertEqual(name, "CRNN")
        self.assertEqual(handle, "MOCK_CRNN_HANDLE")

    def test_v6_small_dispatch(self):
        """rec_model='v6-small' 时返回 PPOcrV6RecHandle，name='v6-small'。"""
        cutbatch.rec_model = "v6-small"
        cutbatch.char_box_strategy = "polarity-aware"
        # 模拟 ppocrv6.PPOcrV6RecHandle 已被 import 到 cutbatch 命名空间
        class FakeV6:
            def __init__(self, path, char_box_strategy):
                self.path = path
                self.char_box_strategy = char_box_strategy
        with patch.dict(cutbatch.__dict__, {"PPOcrV6RecHandle": FakeV6}):
            fake_path = "/tmp/fake_v6_small.onnx"
            with patch.object(config, "ppocrv6_rec_model_path",
                              {"small": fake_path, "tiny": "", "medium": ""}):
                with patch("cutbatch.ppocrv6_rec_model_path",
                           {"small": fake_path, "tiny": "", "medium": ""}):
                    with patch("os.path.exists", return_value=True):
                        handle, name = cutbatch._build_rec_handle()
        self.assertEqual(name, "v6-small")
        self.assertIsInstance(handle, FakeV6)
        self.assertEqual(handle.path, fake_path)
        self.assertEqual(handle.char_box_strategy, "polarity-aware")

    def test_v6_unknown_tier_raises(self):
        cutbatch.rec_model = "v6-huge"
        with self.assertRaises(RuntimeError) as cm:
            cutbatch._build_rec_handle()
        self.assertIn("未知 v6 档位", str(cm.exception))

    def test_v6_missing_file_raises(self):
        """v6 模型文件不存在时给出可操作的下载提示。"""
        cutbatch.rec_model = "v6-small"
        with patch.object(config, "ppocrv6_rec_model_path",
                          {"small": "/nonexistent/path.onnx", "tiny": "", "medium": ""}):
            with patch("cutbatch.ppocrv6_rec_model_path",
                       {"small": "/nonexistent/path.onnx", "tiny": "", "medium": ""}):
                with self.assertRaises(RuntimeError) as cm:
                    cutbatch._build_rec_handle()
        msg = str(cm.exception)
        self.assertIn("v6 rec 模型文件不存在", msg)
        self.assertIn("README.md", msg)

    def test_unknown_rec_model_raises(self):
        cutbatch.rec_model = "deepseek-v3"
        with self.assertRaises(RuntimeError) as cm:
            cutbatch._build_rec_handle()
        self.assertIn("未知 rec_model", str(cm.exception))
        # 错误消息要列出所有合法选项，方便用户改
        for opt in ("crnn", "v6-tiny", "v6-small", "v6-medium"):
            self.assertIn(opt, str(cm.exception))


if __name__ == "__main__":
    unittest.main()
