# -*- coding: utf-8 -*-
"""storage_utils 单元测试。

重点回归：get_rotate_crop_image 不得对高宽比 >= 1.5 的竖排切片执行 np.rot90
（旧 utils.py 版本的"压扁 bug"，主线版本已刻意绕开）。
"""

import json
import os
import tempfile
import unittest
from unittest import mock

import numpy as np
from PIL import Image

from storage_utils import (
    _as_box_array,
    _order_points,
    clear_existing_crops,
    extract_boxes_from_ocr_results,
    get_output_dir_for_item,
    get_rotate_crop_image,
    load_item_annotation_if_exists,
    load_item_json_if_exists,
    normalize_boxes,
    save_current_slices_and_json,
    save_item_boxes_to_disk,
    save_item_json,
)


def _quad(x0, y0, x1, y1):
    """构造一个轴对齐的四点框 (4,2)。"""
    return np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], dtype=np.float32)


class BoxArrayTests(unittest.TestCase):
    def test_as_box_array_accepts_4x2(self):
        arr = _as_box_array([[0, 0], [1, 0], [1, 1], [0, 1]])
        self.assertIsNotNone(arr)
        self.assertEqual(arr.shape, (4, 2))

    def test_as_box_array_accepts_flat_8(self):
        arr = _as_box_array([0, 0, 1, 0, 1, 1, 0, 1])
        self.assertIsNotNone(arr)
        self.assertEqual(arr.shape, (4, 2))

    def test_as_box_array_rejects_bad_shape(self):
        self.assertIsNone(_as_box_array([1, 2, 3]))
        self.assertIsNone(_as_box_array("not a box"))
        self.assertIsNone(_as_box_array(None))

    def test_as_box_array_rejects_non_finite_coordinates(self):
        self.assertIsNone(_as_box_array([[0, 0], [1, 0], [1, 1], [0, np.nan]]))
        self.assertIsNone(_as_box_array([[0, 0], [1, 0], [1, 1], [0, np.inf]]))


class OrderPointsTests(unittest.TestCase):
    def test_returns_tl_tr_br_bl_order(self):
        # 传入打乱顺序的四点，期望稳定输出：左上、右上、右下、左下
        pts = np.array([[1, 1], [0, 0], [0, 1], [1, 0]], dtype=np.float32)
        ordered = _order_points(pts)
        # 左上角坐标和最小、右下角坐标和最大
        self.assertTrue(np.allclose(ordered[0], [0, 0]))
        self.assertTrue(np.allclose(ordered[2], [1, 1]))

    def test_diamond_uses_each_corner_once(self):
        diamond = np.array([[50, 0], [100, 50], [50, 100], [0, 50]], dtype=np.float32)
        ordered = _order_points(diamond)
        self.assertEqual(len({tuple(point) for point in ordered.tolist()}), 4)
        self.assertTrue(np.allclose(ordered[0], [50, 0]))
        self.assertTrue(np.allclose(ordered[1], [100, 50]))


class NormalizeBoxesTests(unittest.TestCase):
    def test_filters_invalid_and_casts_float32(self):
        mixed = [
            [[0, 0], [1, 0], [1, 1], [0, 1]],   # 合法
            [1, 2, 3],                          # 非法
            None,                                # 非法
            [0, 0, 2, 0, 2, 2, 0, 2],           # 合法扁平
        ]
        out = normalize_boxes(mixed)
        self.assertEqual(len(out), 2)
        for b in out:
            self.assertEqual(b.shape, (4, 2))
            self.assertEqual(b.dtype, np.float32)

    def test_empty_input_returns_empty_list(self):
        self.assertEqual(normalize_boxes([]), [])
        self.assertEqual(normalize_boxes(None), [])

    def test_filters_zero_area_and_non_finite_boxes(self):
        line = np.array([[10, 10], [10, 10], [10, 30], [10, 30]], dtype=np.float32)
        non_finite = _quad(0, 0, 10, 10)
        non_finite[2, 0] = np.nan
        self.assertEqual(normalize_boxes([line, non_finite]), [])


class GetRotateCropImageTests(unittest.TestCase):
    """核心回归：高宽比 >= 1.5 的竖排切片不得被 np.rot90。"""

    def test_horizontal_box_preserves_orientation(self):
        # 100x30 的横排区域
        img = np.zeros((100, 200, 3), dtype=np.uint8)
        crop = get_rotate_crop_image(img, _quad(10, 10, 110, 40))
        self.assertIsNotNone(crop)
        self.assertEqual(crop.shape[0], 30)  # h
        self.assertEqual(crop.shape[1], 100)  # w

    def test_vertical_box_not_rotated_to_horizontal(self):
        """竖排切片（h/w >= 1.5）必须保持竖向，不得被 rot90 转横。

        旧 utils.py 版本会对这种输入执行 np.rot90，导致竖排中文被压扁。
        """
        img = np.zeros((400, 100, 3), dtype=np.uint8)
        # 30 宽 x 200 高的竖排区域，h/w = 6.67 >= 1.5
        crop = get_rotate_crop_image(img, _quad(20, 50, 50, 250))
        self.assertIsNotNone(crop)
        # 输出应保持竖向：高 > 宽
        self.assertGreater(crop.shape[0], crop.shape[1],
                           f"竖排切片被错误旋转：shape={crop.shape}")

    def test_45_degree_diamond_can_be_cropped(self):
        img = np.zeros((120, 120, 3), dtype=np.uint8)
        diamond = np.array([[60, 10], [110, 60], [60, 110], [10, 60]], dtype=np.float32)
        crop = get_rotate_crop_image(img, diamond)
        self.assertIsNotNone(crop)
        self.assertGreater(crop.size, 0)

    def test_returns_none_for_degenerate_box(self):
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        # 退化成一点
        self.assertIsNone(get_rotate_crop_image(img, _quad(10, 10, 10, 10)))

    def test_returns_none_for_empty_image(self):
        self.assertIsNone(get_rotate_crop_image(None, _quad(0, 0, 10, 10)))
        empty = np.zeros((0, 0, 3), dtype=np.uint8)
        self.assertIsNone(get_rotate_crop_image(empty, _quad(0, 0, 10, 10)))


class ExtractBoxesFromOcrResultsTests(unittest.TestCase):
    def _result(self, bbox, char_list=None, words=None):
        return {
            "bbox": bbox,
            "result": {
                "line_pred": "",
                "char_list": char_list or [],
                "words": words or [],
            },
            "score": 0.9,
        }

    def test_line_level_uses_bbox(self):
        results = [self._result(_quad(0, 0, 100, 20))]
        boxes = extract_boxes_from_ocr_results(results, level="line")
        self.assertEqual(len(boxes), 1)
        self.assertEqual(boxes[0].shape, (4, 2))

    def test_char_level_extracts_char_list(self):
        results = [self._result(
            _quad(0, 0, 100, 20),
            char_list=[{"char": "字", "score": 0.91,
                        "bbox": _quad(0, 0, 20, 20)},
                       {"char": "符", "bbox": _quad(20, 0, 40, 20)}],
        )]
        boxes = extract_boxes_from_ocr_results(results, level="char")
        self.assertEqual(len(boxes), 2)

    def test_word_level_extracts_words(self):
        results = [self._result(
            _quad(0, 0, 100, 20),
            words=[{"value": "ab", "bbox": _quad(0, 0, 40, 20)}],
        )]
        boxes = extract_boxes_from_ocr_results(results, level="word")
        self.assertEqual(len(boxes), 1)

    def test_empty_results_returns_empty(self):
        self.assertEqual(extract_boxes_from_ocr_results([], "char"), [])
        self.assertEqual(extract_boxes_from_ocr_results(None, "char"), [])

    def test_sparse_cjk_horizontal_line_falls_back_to_character_boxes(self):
        line = _quad(0, 0, 160, 20)
        results = [self._result(
            line,
            char_list=[{"char": "字", "score": 0.3, "bbox": line}],
        )]
        results[0]["result"]["line_pred"] = "字"

        boxes = extract_boxes_from_ocr_results(results, "char")

        self.assertEqual(len(boxes), 4)
        for box in boxes:
            width = np.linalg.norm(box[0] - box[1])
            height = np.linalg.norm(box[0] - box[3])
            self.assertAlmostEqual(width, 40.0, delta=0.1)
            self.assertAlmostEqual(height, 20.0, delta=0.1)
        self.assertEqual(
            results[0]["result"]["char_box_fallback"]["source"],
            "bounded_geometry",
        )

    def test_sparse_cjk_vertical_column_falls_back_to_character_boxes(self):
        column = _quad(0, 0, 20, 120)
        results = [self._result(
            column,
            char_list=[{"char": "字", "score": 0.3, "bbox": column}],
        )]
        results[0]["result"]["line_pred"] = "字"

        boxes = extract_boxes_from_ocr_results(results, "char")

        self.assertEqual(len(boxes), 4)
        centers = [box.mean(axis=0) for box in boxes]
        self.assertTrue(all(
            left[1] < right[1]
            for left, right in zip(centers, centers[1:])
        ))

    def test_sparse_english_line_is_not_geometrically_fabricated(self):
        line = _quad(0, 0, 160, 20)
        results = [self._result(
            line,
            char_list=[{"char": "H", "bbox": line}],
        )]
        results[0]["result"]["line_pred"] = "H"

        boxes = extract_boxes_from_ocr_results(results, "char")

        self.assertEqual(len(boxes), 1)

    def test_high_confidence_single_cjk_in_padded_box_is_not_split(self):
        line = _quad(0, 0, 160, 20)
        results = [self._result(
            line,
            char_list=[{"char": "字", "score": 0.95, "bbox": line}],
        )]
        results[0]["result"]["line_pred"] = "字"

        boxes = extract_boxes_from_ocr_results(results, "char")

        self.assertEqual(len(boxes), 1)
        self.assertTrue(np.allclose(boxes[0], line))

    def test_single_cjk_without_confidence_is_not_split(self):
        line = _quad(0, 0, 160, 20)
        results = [self._result(
            line,
            char_list=[{"char": "字", "bbox": line}],
        )]
        results[0]["result"]["line_pred"] = "字"

        boxes = extract_boxes_from_ocr_results(results, "char")

        self.assertEqual(len(boxes), 1)

    def test_adequately_recognized_cjk_line_keeps_model_boxes(self):
        line = _quad(0, 0, 100, 20)
        model_boxes = [
            _quad(0, 0, 20, 20),
            _quad(20, 0, 40, 20),
            _quad(40, 0, 60, 20),
        ]
        results = [self._result(
            line,
            char_list=[
                {"char": char, "bbox": box}
                for char, box in zip("字符框", model_boxes)
            ],
        )]
        results[0]["result"]["line_pred"] = "字符框"

        boxes = extract_boxes_from_ocr_results(results, "char")

        self.assertEqual(len(boxes), 3)
        for actual, expected in zip(boxes, model_boxes):
            self.assertTrue(np.allclose(actual, expected))


class OutputDirTests(unittest.TestCase):
    def test_image_item_path(self):
        folder, prefix = get_output_dir_for_item(
            "/out", {"type": "image", "source_path": "/data/photo.jpg"})
        self.assertEqual(folder, os.path.join("/out", "photo"))
        self.assertEqual(prefix, "photo")

    def test_pdf_page_item_path(self):
        folder, prefix = get_output_dir_for_item(
            "/out", {"type": "pdf_page", "source_path": "/data/doc.pdf", "page_index": 2})
        # page_index 0-based，输出用 page_003
        self.assertEqual(folder, os.path.join("/out", "doc", "page_003"))
        self.assertEqual(prefix, "doc_p003")


class JsonRoundTripTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="chardet_test_")
        self.item = {"type": "image", "source_path": "/data/foo.png", "page_index": None}

    def test_save_and_load_roundtrip_preserves_boxes(self):
        boxes = [_quad(0, 0, 10, 10), _quad(20, 20, 30, 30)]
        save_item_json(self.tmp, self.item, boxes, image_size_wh=(100, 200))

        loaded = load_item_json_if_exists(self.tmp, self.item, target_size_wh=(100, 200))
        self.assertIsNotNone(loaded)
        self.assertEqual(len(loaded), 2)
        for b in loaded:
            self.assertEqual(b.shape, (4, 2))

    def test_load_rescales_when_target_size_differs(self):
        boxes = [_quad(0, 0, 50, 50)]
        save_item_json(self.tmp, self.item, boxes, image_size_wh=(100, 100))

        # 目标尺寸翻倍 -> 坐标也应翻倍
        loaded = load_item_json_if_exists(self.tmp, self.item, target_size_wh=(200, 200))
        self.assertIsNotNone(loaded)
        self.assertTrue(np.allclose(loaded[0], _quad(0, 0, 100, 100)))

    def test_load_filters_degenerate_box_from_old_json(self):
        save_item_json(
            self.tmp,
            self.item,
            [_quad(20, 20, 40, 40)],
            image_size_wh=(100, 100),
        )
        from storage_utils import get_json_path_for_item
        json_path = get_json_path_for_item(self.tmp, self.item)
        with open(json_path, "r", encoding="utf-8") as json_file:
            data = json.load(json_file)
        data["boxes"].insert(0, [[10, 10], [10, 10], [10, 30], [10, 30]])
        with open(json_path, "w", encoding="utf-8") as json_file:
            json.dump(data, json_file, ensure_ascii=False)

        loaded = load_item_annotation_if_exists(
            self.tmp,
            self.item,
            target_size_wh=(100, 100),
        )

        self.assertEqual(len(loaded), 1)
        self.assertTrue(np.allclose(loaded[0], _quad(20, 20, 40, 40)))

    def test_load_returns_none_when_no_json(self):
        loaded = load_item_json_if_exists(self.tmp, self.item, target_size_wh=(100, 100))
        self.assertIsNone(loaded)

    def test_json_contains_orientation_metadata(self):
        save_item_json(self.tmp, self.item, [_quad(0, 0, 10, 10)], image_size_wh=(50, 50))
        from storage_utils import get_json_path_for_item
        with open(get_json_path_for_item(self.tmp, self.item), "r", encoding="utf-8") as f:
            data = json.load(f)
        # 主线版本应携带这些元信息字段
        self.assertTrue(data.get("preserve_crop_orientation"))
        self.assertIn("coordinate_system", data)

    def test_old_json_hidden_fields_are_ignored(self):
        save_item_json(
            self.tmp,
            self.item,
            [_quad(0, 0, 10, 10)],
            image_size_wh=(50, 50),
        )
        from storage_utils import get_json_path_for_item
        json_path = get_json_path_for_item(self.tmp, self.item)
        with open(json_path, "r", encoding="utf-8") as json_file:
            data = json.load(json_file)
        data["box_recognitions"] = [{"text": "旧"}]
        data["box_recognition_context"] = {"recognizer_id": "old"}
        with open(json_path, "w", encoding="utf-8") as json_file:
            json.dump(data, json_file, ensure_ascii=False)

        loaded = load_item_annotation_if_exists(
            self.tmp,
            self.item,
            target_size_wh=(50, 50),
        )

        self.assertEqual(len(loaded), 1)
        self.assertEqual(loaded[0].shape, (4, 2))

    def test_new_json_omits_hidden_recognition_fields(self):
        save_item_json(
            self.tmp,
            self.item,
            [_quad(0, 0, 10, 10)],
            image_size_wh=(50, 50),
        )
        from storage_utils import get_json_path_for_item
        with open(get_json_path_for_item(self.tmp, self.item), "r", encoding="utf-8") as json_file:
            data = json.load(json_file)
        self.assertNotIn("box_recognitions", data)
        self.assertNotIn("box_recognition_context", data)


class SaveSlicesTests(unittest.TestCase):
    def test_save_writes_crops_and_returns_count(self):
        tmp = tempfile.mkdtemp(prefix="chardet_test_")
        item = {"type": "image", "source_path": "/data/x.jpg", "page_index": None}
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        boxes = [_quad(10, 10, 30, 30), _quad(40, 40, 60, 60)]

        saved = save_current_slices_and_json(tmp, item, img, boxes)
        self.assertEqual(saved, 2)

        folder, _ = get_output_dir_for_item(tmp, item)
        crops = [f for f in os.listdir(folder) if f.endswith(".png")]
        self.assertEqual(len(crops), 2)

    def test_degenerate_box_is_not_persisted_or_retried(self):
        tmp = tempfile.mkdtemp(prefix="chardet_test_")
        item = {"type": "image", "source_path": "/data/x.jpg", "page_index": None}
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        line = np.array([[10, 10], [10, 10], [10, 30], [10, 30]], dtype=np.float32)

        saved = save_current_slices_and_json(
            tmp,
            item,
            img,
            [line, _quad(40, 40, 60, 60)],
        )

        self.assertEqual(saved, 1)
        self.assertEqual(len(item["boxes"]), 1)
        loaded = load_item_annotation_if_exists(tmp, item, target_size_wh=(100, 100))
        self.assertEqual(len(loaded), 1)

    def test_unexpected_crop_failure_keeps_existing_slices(self):
        tmp = tempfile.mkdtemp(prefix="chardet_test_")
        item = {"type": "image", "source_path": "/data/x.jpg", "page_index": None}
        folder, _ = get_output_dir_for_item(tmp, item)
        os.makedirs(folder, exist_ok=True)
        old_crop = os.path.join(folder, "data-x_crop_0.png")
        with open(old_crop, "wb") as crop_file:
            crop_file.write(b"old-crop")

        with mock.patch("storage_utils.get_rotate_crop_image", return_value=None):
            with self.assertRaisesRegex(RuntimeError, "透视裁剪失败"):
                save_current_slices_and_json(
                    tmp,
                    item,
                    np.zeros((100, 100, 3), dtype=np.uint8),
                    [_quad(10, 10, 30, 30)],
                )

        with open(old_crop, "rb") as crop_file:
            self.assertEqual(crop_file.read(), b"old-crop")

    def test_second_slice_write_failure_keeps_old_output_and_json(self):
        tmp = tempfile.mkdtemp(prefix="chardet_test_")
        item = {
            "type": "image",
            "source_path": "/data/x.jpg",
            "page_index": None,
        }
        old_box = _quad(1, 1, 10, 10)
        save_current_slices_and_json(
            tmp,
            item,
            np.zeros((40, 40, 3), dtype=np.uint8),
            [old_box],
        )
        folder, _ = get_output_dir_for_item(tmp, item)
        old_crop = os.path.join(folder, "data-x_crop_0.png")
        with open(old_crop, "rb") as crop_file:
            old_crop_bytes = crop_file.read()

        with mock.patch(
            "storage_utils._safe_imwrite",
            side_effect=[True, False],
        ):
            with self.assertRaisesRegex(RuntimeError, "切片写入失败"):
                save_current_slices_and_json(
                    tmp,
                    item,
                    np.zeros((40, 40, 3), dtype=np.uint8),
                    [_quad(2, 2, 12, 12), _quad(20, 20, 30, 30)],
                )

        with open(old_crop, "rb") as crop_file:
            self.assertEqual(crop_file.read(), old_crop_bytes)
        loaded = load_item_annotation_if_exists(
            tmp,
            item,
            target_size_wh=(40, 40),
        )
        self.assertEqual(len(loaded), 1)
        self.assertTrue(np.allclose(loaded[0], old_box))

    def test_json_failure_rolls_back_replaced_slices(self):
        tmp = tempfile.mkdtemp(prefix="chardet_test_")
        item = {
            "type": "image",
            "source_path": "/data/x.jpg",
            "page_index": None,
        }
        old_box = _quad(1, 1, 10, 10)
        save_current_slices_and_json(
            tmp,
            item,
            np.zeros((40, 40, 3), dtype=np.uint8),
            [old_box],
        )
        folder, _ = get_output_dir_for_item(tmp, item)
        old_crop = os.path.join(folder, "data-x_crop_0.png")
        with open(old_crop, "rb") as crop_file:
            old_crop_bytes = crop_file.read()

        with mock.patch(
            "storage_utils.save_item_json",
            side_effect=OSError("json failed"),
        ):
            with self.assertRaisesRegex(OSError, "json failed"):
                save_current_slices_and_json(
                    tmp,
                    item,
                    np.ones((40, 40, 3), dtype=np.uint8) * 255,
                    [_quad(2, 2, 12, 12)],
                )

        with open(old_crop, "rb") as crop_file:
            self.assertEqual(crop_file.read(), old_crop_bytes)
        self.assertTrue(np.allclose(item["boxes"][0], old_box))

    def test_clear_existing_crops_removes_old(self):
        tmp = tempfile.mkdtemp(prefix="chardet_test_")
        # 放两个旧 crop
        import storage_utils as su
        from storage_utils import _crop_filename
        old = os.path.join(tmp, _crop_filename({"source_path": "/x/y.jpg"}, "y", 0))
        with open(old, "wb") as f:
            f.write(b"\x89PNG\r\n\x1a\n")
        self.assertEqual(len(os.listdir(tmp)), 1)

        clear_existing_crops(tmp)
        self.assertEqual(len(os.listdir(tmp)), 0)

    def test_clear_existing_crops_removes_legacy_char_files(self):
        tmp = tempfile.mkdtemp(prefix="chardet_test_")
        legacy_crop = os.path.join(tmp, "char_001.png")
        with open(legacy_crop, "wb") as crop_file:
            crop_file.write(b"\x89PNG\r\n\x1a\n")

        clear_existing_crops(tmp)

        self.assertFalse(os.path.exists(legacy_crop))

    def test_fallback_ocr_saves_boxes_without_hidden_recognition(self):
        tmp = tempfile.mkdtemp(prefix="chardet_test_")
        item = {
            "type": "image",
            "source_path": "/data/unvisited.jpg",
            "page_index": None,
        }
        ocr_handler = mock.MagicMock()
        ocr_handler.text_predict.return_value = [
            {
                "bbox": _quad(0, 0, 20, 20),
                "score": 0.99,
                "result": {
                    "char_list": [
                        {
                            "char": "字",
                            "score": 0.91,
                            "bbox": _quad(0, 0, 20, 20),
                        }
                    ],
                    "words": [],
                },
            }
        ]

        saved = save_item_boxes_to_disk(
            tmp,
            item,
            Image.new("RGB", (20, 20), "white"),
            ocr_handler=ocr_handler,
            level="char",
        )

        self.assertEqual(saved, 1)
        self.assertNotIn("box_recognitions", item)
        boxes = load_item_annotation_if_exists(
            tmp,
            item,
            target_size_wh=(20, 20),
        )

    def test_batch_save_reocrs_json_with_mismatched_level(self):
        tmp = tempfile.mkdtemp(prefix="chardet_test_")
        line_context = {
            "level": "line",
            "det_model": "v6-small",
            "rec_model": "v6-small",
            "char_box_strategy": "polarity-aware",
        }
        char_context = dict(line_context, level="char")
        item = {
            "type": "image",
            "source_path": "/data/context.jpg",
            "page_index": None,
            "annotation_context": line_context,
            "box_context": line_context,
        }
        save_item_json(
            tmp,
            item,
            [_quad(0, 0, 19, 19)],
            image_size_wh=(20, 20),
        )
        item = {
            "type": "image",
            "source_path": "/data/context.jpg",
            "page_index": None,
            "annotation_context": char_context,
        }
        ocr_handler = mock.MagicMock()
        ocr_handler.text_predict.return_value = [
            {
                "bbox": _quad(0, 0, 19, 19),
                "result": {
                    "char_list": [
                        {"char": "字", "bbox": _quad(0, 0, 9, 19)},
                        {"char": "符", "bbox": _quad(10, 0, 19, 19)},
                    ],
                    "words": [],
                },
            }
        ]

        saved = save_item_boxes_to_disk(
            tmp,
            item,
            Image.new("RGB", (20, 20), "white"),
            ocr_handler=ocr_handler,
            level="char",
        )

        self.assertEqual(saved, 2)
        ocr_handler.text_predict.assert_called_once()
        self.assertEqual(item["box_context"], char_context)

    def test_visited_empty_boxes_do_not_resurrect_previous_json(self):
        tmp = tempfile.mkdtemp(prefix="chardet_test_")
        box = _quad(0, 0, 19, 19)
        item = {
            "type": "image",
            "source_path": "/data/edited.jpg",
            "page_index": None,
            "boxes": [box],
        }
        save_current_slices_and_json(
            tmp,
            item,
            np.zeros((20, 20, 3), dtype=np.uint8),
            [box],
        )
        item.update(
            {
                "visited": True,
                "dirty": True,
                "boxes": [],
            }
        )
        ocr_handler = mock.MagicMock()

        saved = save_item_boxes_to_disk(
            tmp,
            item,
            Image.new("RGB", (20, 20), "white"),
            ocr_handler=ocr_handler,
            level="char",
        )

        self.assertEqual(saved, 0)
        self.assertEqual(item["boxes"], [])
        ocr_handler.text_predict.assert_not_called()
        boxes = load_item_annotation_if_exists(
            tmp,
            item,
            target_size_wh=(20, 20),
        )
        self.assertEqual(boxes, [])
        folder, _prefix = get_output_dir_for_item(tmp, item)
        self.assertEqual(
            [
                name for name in os.listdir(folder)
                if "_crop_" in name or name.startswith("char_")
            ],
            [],
        )

    def test_fallback_ocr_failure_is_not_reported_as_empty_success(self):
        tmp = tempfile.mkdtemp(prefix="chardet_test_")
        item = {
            "type": "image",
            "source_path": "/data/unvisited.jpg",
            "page_index": None,
        }
        ocr_handler = mock.MagicMock()
        ocr_handler.text_predict.side_effect = RuntimeError("ocr crashed")

        with self.assertRaisesRegex(RuntimeError, "OCR 失败"):
            save_item_boxes_to_disk(
                tmp,
                item,
                Image.new("RGB", (20, 20), "white"),
                ocr_handler=ocr_handler,
                level="char",
            )

        self.assertFalse(os.path.exists(os.path.join(tmp, "unvisited")))


if __name__ == "__main__":
    unittest.main()
