# -*- coding: utf-8 -*-
"""excel_utils 单元测试。

只测纯逻辑函数（assign_excel_positions、parse_crop_index、sanitize_sheet_name、
_box_features），不测真正写 xlsx 的 build_excel_for_single_item（依赖磁盘 IO
和 xlsxwriter，留给集成测试）。
"""

import os
import tempfile
import unittest

from utils.excel import (
    _box_features,
    assign_excel_positions,
    list_saved_crops_for_item,
    parse_crop_index,
    sanitize_sheet_name,
)


def _rec(cx, cy, w=10, h=10, crop_index=None, path=""):
    """构造一个最小 record。"""
    rec = {
        "cx": cx, "cy": cy, "w": w, "h": h,
        "x_min": cx - w / 2, "x_max": cx + w / 2,
        "y_min": cy - h / 2, "y_max": cy + h / 2,
        "path": path, "crop_index": crop_index if crop_index is not None else 0,
        "has_box": True,
    }
    return rec


class ParseCropIndexTests(unittest.TestCase):
    def test_extracts_from_crop_pattern(self):
        self.assertEqual(parse_crop_index("foo_crop_3.png"), 3)
        self.assertEqual(parse_crop_index("foo_crop_12.jpg"), 12)

    def test_extracts_from_char_pattern(self):
        # char_NNN 模式返回 NNN-1（兼容旧的 1-based 编号）
        self.assertEqual(parse_crop_index("char_001.png"), 0)
        self.assertEqual(parse_crop_index("char_005.jpeg"), 4)

    def test_returns_none_for_no_match(self):
        self.assertIsNone(parse_crop_index("random.png"))
        self.assertIsNone(parse_crop_index("foo_crop_.png"))

    def test_case_insensitive(self):
        self.assertEqual(parse_crop_index("FOO_CROP_7.PNG"), 7)

    def test_saved_crops_sort_zero_before_positive_indices(self):
        with tempfile.TemporaryDirectory() as tmp:
            for name in (
                "book_crop_10.png",
                "book_crop_0.png",
                "book_crop_1.png",
            ):
                open(os.path.join(tmp, name), "wb").close()

            paths = list_saved_crops_for_item(tmp)

            self.assertEqual(
                [parse_crop_index(path) for path in paths],
                [0, 1, 10],
            )


class SanitizeSheetNameTests(unittest.TestCase):
    def test_replaces_invalid_chars(self):
        # Excel sheet 名禁止的字符：[ ] : * ? / \
        self.assertEqual(sanitize_sheet_name("a[b]c"), "a_b_c")
        self.assertEqual(sanitize_sheet_name("file:name"), "file_name")
        self.assertEqual(sanitize_sheet_name("a/b\\c"), "a_b_c")

    def test_truncates_to_31_chars(self):
        long_name = "x" * 50
        self.assertEqual(len(sanitize_sheet_name(long_name)), 31)

    def test_empty_falls_back_to_sheet(self):
        self.assertEqual(sanitize_sheet_name(""), "Sheet")
        self.assertEqual(sanitize_sheet_name(None), "Sheet")


class BoxFeaturesTests(unittest.TestCase):
    def test_returns_none_for_bad_input(self):
        self.assertIsNone(_box_features(None))
        self.assertIsNone(_box_features([1, 2, 3]))

    def test_computes_center_and_size(self):
        feat = _box_features([[10, 20], [30, 20], [30, 40], [10, 40]])
        self.assertAlmostEqual(feat["cx"], 20)
        self.assertAlmostEqual(feat["cy"], 30)
        self.assertEqual(feat["w"], 20)
        self.assertEqual(feat["h"], 20)

    def test_accepts_flat_8(self):
        feat = _box_features([0, 0, 10, 0, 10, 10, 0, 10])
        self.assertAlmostEqual(feat["cx"], 5)


class AssignExcelPositionsTests(unittest.TestCase):
    def test_empty_input_returns_empty(self):
        self.assertEqual(assign_excel_positions([], "vertical_rtl"), [])

    def test_no_position_falls_back_to_sequential(self):
        # has_box=False 应触发 sequential 分支
        recs = [{"cx": 0, "cy": 0, "w": 1, "h": 1, "has_box": False,
                 "crop_index": i, "path": f"p{i}.png"} for i in range(25)]
        out = assign_excel_positions(recs, "sequential")
        self.assertEqual(len(out), 25)
        # sequential 默认 20 列：第 0 行 0-19，第 1 行 20-24
        self.assertEqual(out[0]["excel_row"], 0)
        self.assertEqual(out[0]["excel_col"], 0)
        self.assertEqual(out[20]["excel_row"], 1)
        self.assertEqual(out[20]["excel_col"], 0)

    def test_each_record_gets_unique_cell(self):
        # 4 个字符排成 2x2 网格
        recs = [
            _rec(10, 10, crop_index=0),  # 左上
            _rec(30, 10, crop_index=1),  # 右上
            _rec(10, 30, crop_index=2),  # 左下
            _rec(30, 30, crop_index=3),  # 右下
        ]
        out = assign_excel_positions(recs, "horizontal_ltr", layout_policy="aligned")
        cells = {(r["excel_row"], r["excel_col"]) for r in out}
        self.assertEqual(len(cells), 4, f"有重叠单元格: {cells}")

    def test_vertical_rtl_aligned_preserves_source_columns_and_reading_order(self):
        recs = [
            _rec(10, 10, crop_index=0),
            _rec(10, 30, crop_index=1),
            _rec(100, 10, crop_index=2),
            _rec(100, 30, crop_index=3),
        ]

        out = assign_excel_positions(
            recs,
            "vertical_rtl",
            layout_policy="aligned",
        )
        by_crop = {r["crop_index"]: r for r in out}

        self.assertEqual(by_crop[0]["excel_col"], 0)
        self.assertEqual(by_crop[2]["excel_col"], 1)
        self.assertEqual(
            [
                r["crop_index"]
                for r in sorted(out, key=lambda r: r["order_index"])
            ],
            [2, 3, 0, 1],
        )

    def test_horizontal_rtl_aligned_preserves_source_columns_and_reading_order(self):
        recs = [
            _rec(10, 10, crop_index=0),
            _rec(100, 10, crop_index=1),
        ]

        out = assign_excel_positions(
            recs,
            "horizontal_rtl",
            layout_policy="aligned",
        )
        by_crop = {r["crop_index"]: r for r in out}

        self.assertEqual(by_crop[0]["excel_col"], 0)
        self.assertEqual(by_crop[1]["excel_col"], 1)
        self.assertEqual(
            [
                r["crop_index"]
                for r in sorted(out, key=lambda r: r["order_index"])
            ],
            [1, 0],
        )

    def test_order_index_assigned(self):
        recs = [_rec(10, 10, crop_index=i) for i in range(3)]
        out = assign_excel_positions(recs, "horizontal_ltr")
        order_indices = sorted(r["order_index"] for r in out)
        self.assertEqual(order_indices, [0, 1, 2])


if __name__ == "__main__":
    unittest.main()
