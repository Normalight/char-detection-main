import onnxruntime as ort

ort.set_default_logger_severity(3)  # ERROR level, 忽略 warning
import os
import cv2
import numpy as np
from PIL import Image
import copy
import logging

logger = logging.getLogger(__name__)

# --- 项目相关导入 ---
try:
    # 确保这些路径和变量在 config.py 中是正确的
    import config as app_config

    # 为可选参数提供默认值，以防它们在 config.py 中不存在
    # 检测模型选择："v6-tiny" / "v6-small" / "v6-medium"
    det_model = getattr(app_config, "det_model", "v6-small")
    ppocrv6_det_model_path = getattr(
        app_config, "ppocrv6_det_model_path", {}
    )
    ppocrv6_det_limit_type = getattr(
        app_config, "ppocrv6_det_limit_type", "short"
    )
    ppocrv6_det_max_side_limit = getattr(
        app_config, "ppocrv6_det_max_side_limit", 4000
    )
    ppocrv6_det_box_thresh = getattr(
        app_config, "ppocrv6_det_box_thresh", None
    )
    # 识别模型选择："v6-tiny" / "v6-small" / "v6-medium"（见 config.py 注释）
    rec_model = getattr(app_config, "rec_model", "v6-small")
    char_box_strategy = getattr(
        app_config, "char_box_strategy", "ctc-midpoint"
    )
    ppocrv6_rec_model_path = getattr(
        app_config, "ppocrv6_rec_model_path", {}
    )

    from ocr.geometry import get_rotate_crop_image
    from ocr import PPOcrV6Detector, PPOcrV6RecHandle
except ImportError as e:
    # 注意：模块顶层不能调用 exit()，否则在 import 链中（例如 GUI 入口 app.py
    # 通过 annotation.controller → ocr.pipeline 间接 import）会让整个进程静默退出，
    # 上层无法捕获。改为抛 RuntimeError，让上层显示清晰的错误提示。
    raise RuntimeError(
        f"ocr.pipeline 导入失败：{e}。"
        f"请确保 config.py / ocr 包 存在且路径正确，"
        f"并已安装 onnxruntime / opencv-python / numpy 等依赖。"
    ) from e


# --- 项目相关导入结束 ---


def _build_det_handle():
    """根据 ``config.det_model`` 构造检测器并返回 ``(handle, name)``。"""

    if det_model.startswith("v6-"):
        tier = det_model.split("-", 1)[1]
        path = ppocrv6_det_model_path.get(tier)
        if not path:
            raise RuntimeError(
                f"未知 v6 检测档位：{det_model}（可选 tiny/small/medium）"
            )
        if not os.path.exists(path):
            raise RuntimeError(
                f"v6 det 模型文件不存在：{path}\n"
                "下载方式见 README.md"
                f"（huggingface_hub 拉取 PP-OCRv6_{tier}_det_onnx）"
            )
        return (
            PPOcrV6Detector(
                path,
                limit_type=ppocrv6_det_limit_type,
                max_side_limit=ppocrv6_det_max_side_limit,
                box_thresh=ppocrv6_det_box_thresh,
            ),
            f"v6-{tier}",
        )
    raise RuntimeError(
        f"未知 det_model：{det_model}"
        "（可选 v6-tiny / v6-small / v6-medium）"
    )


def _build_rec_handle():
    """根据 config.rec_model 选择识别模型 handle。

    返回 (handle, name) —— name 仅用于日志/调试。
    所有 handle 共享同一接口：predict_rbg(crop_pil, bbox, **rec_options) -> dict
    （schema: {'raw', 'line_pred', 'char_list', 'words'}）。
    """
    if rec_model.startswith("v6-"):
        tier = rec_model.split("-", 1)[1]   # tiny / small / medium
        path = ppocrv6_rec_model_path.get(tier)
        if not path:
            raise RuntimeError(f"未知 v6 档位：{rec_model}（可选 tiny/small/medium）")
        if not os.path.exists(path):
            raise RuntimeError(
                f"v6 rec 模型文件不存在：{path}\n"
                f"下载方式见 README.md（huggingface_hub 拉取 PP-OCRv6_{tier}_rec_onnx）"
            )
        return (
            PPOcrV6RecHandle(
                path, char_box_strategy=char_box_strategy
            ),
            f"v6-{tier}",
        )
    raise RuntimeError(f"未知 rec_model：{rec_model}（可选 v6-tiny / v6-small / v6-medium）")


def _deduplicate_text_rows(boxes, scores):
    """Suppress near-identical detector proposals, keeping legitimate overlap.

    Polygon IoU and similar area are both required; shared strokes or one
    region contained in a much larger line are not treated as duplicates.
    """
    kept = []
    polygons = [cv2.convexHull(np.asarray(box, np.float32).reshape(-1, 2)) for box in boxes]
    areas = [abs(cv2.contourArea(polygon)) for polygon in polygons]
    for index in sorted(range(len(boxes)), key=lambda i: float(scores[i]), reverse=True):
        duplicate = False
        for previous in kept:
            small, large = min(areas[index], areas[previous]), max(areas[index], areas[previous])
            if small <= 0 or small / large < .85:
                continue
            intersection, _ = cv2.intersectConvexConvex(polygons[index], polygons[previous])
            union = areas[index] + areas[previous] - intersection
            if union > 0 and intersection / union >= .85:
                duplicate = True
                break
        if not duplicate:
            kept.append(index)
    kept.sort()
    return [boxes[i] for i in kept], [scores[i] for i in kept]


class OcrHandle(object):
    def __init__(self):
        self.text_handle, self.det_name = _build_det_handle()
        self.recognizer, self.rec_name = _build_rec_handle()
        logger.info(
            "OCR 后端：det=%s, rec=%s, char_box=%s",
            self.det_name,
            self.rec_name,
            char_box_strategy,
        )

    def recognize_rows(self, im_opencv_bgr, text_boxes, detection_scores, cancel_check=None):
        """文本行批量识别。

        性能：把所有 crop 出来的行一次性塞进 predict_rbg_batch，
        按宽度分桶后每个桶一次 ONNX 推理。一张 N 行文字的图，
        相比旧的 for 循环逐行推理快 3-10×（固定开销被摊掉）。

        返回结构与历史版本一致：[{'bbox', 'result', 'score'}, ...]
        """
        # 1) 先把所有行裁出来；失败的或空的，记下索引后续填空结果。
        valid_items = []   # list of (orig_index, crop_pil_rgb, bbox_deepcopy)
        empty_indices = set()
        for index, (box_coords_quad, _score) in enumerate(zip(text_boxes, detection_scores)):
            cropped_text_region_np_bgr = None
            try:
                cropped_text_region_np_bgr = get_rotate_crop_image(
                    im_opencv_bgr, box_coords_quad.astype(np.float32))
            except cv2.error as e_cv_crop:
                logger.warning("recognize_rows get_rotate_crop_image OpenCV 错误 (box %d): %s", index, e_cv_crop)
            except Exception as e_crop:
                logger.warning("recognize_rows get_rotate_crop_image 未知错误 (box %d): %s", index, e_crop)

            if cropped_text_region_np_bgr is None or cropped_text_region_np_bgr.size == 0:
                empty_indices.add(index)
                continue

            try:
                # cropped 是 BGR；识别器接收 RGB。用 [:, :, ::-1] 零拷贝反转通道，
                # 比 cvtColor 快（不发生内存复制之外的颜色矩阵计算）。
                cropped_rgb = cropped_text_region_np_bgr[:, :, ::-1]
                crop_pil_rgb = Image.fromarray(np.ascontiguousarray(cropped_rgb))
                valid_items.append((index, crop_pil_rgb, copy.deepcopy(box_coords_quad)))
            except Exception as e_conv:
                logger.warning("行 %d 转 PIL 失败：%s", index, e_conv)
                empty_indices.add(index)

        # 2) 批量推理。某一行的退化几何不应让整页失败：批量路径异常时
        # 降级成逐行识别，并只跳过真正失败的那一行。
        recognition_inputs = [(pil, bbox) for _idx, pil, bbox in valid_items]
        rec_options = ({'defer_stroke_refinement': True}
                       if isinstance(self.recognizer, PPOcrV6RecHandle) else {})
        results_by_index = {}
        if recognition_inputs:
            try:
                if cancel_check is None:
                    recognition_outputs = self.recognizer.predict_rbg_batch(recognition_inputs, **rec_options)
                else:
                    recognition_outputs = []
                    for entry in recognition_inputs:
                        if cancel_check():
                            return []
                        recognition_outputs.extend(self.recognizer.predict_rbg_batch([entry], **rec_options))
            except Exception as batch_error:
                logger.warning(
                    "批量识别失败，降级为逐行隔离：%s", batch_error
                )
                for idx, crop_pil, bbox in valid_items:
                    if cancel_check is not None and cancel_check():
                        return []
                    try:
                        results_by_index[idx] = (
                            self.recognizer.predict_rbg(crop_pil, bbox, **rec_options)
                        )
                    except Exception as row_error:
                        logger.warning(
                            "跳过识别失败的文本行 %d：%s",
                            idx,
                            row_error,
                        )
            else:
                for (idx, _pil, _bbox), result in zip(
                    valid_items, recognition_outputs
                ):
                    results_by_index[idx] = result

        # 3) 装回原顺序；空 crop / 单行失败的位置用默认空结果。
        # 用 dict 按 orig_index 存，避免 batch 返回顺序与原 boxes 顺序耦合。
        page_rows = [
            {'bbox': box, 'score': score,
             'result': results_by_index.get(index, {'line_pred':'','char_list':[],'words':[]})}
            for index, (box, score) in enumerate(zip(text_boxes, detection_scores))
        ]
        if cancel_check is not None and cancel_check():
            return []
        from ocr.page_layout import refine_regular_rubbing_page
        # Regular pages use a shared size/spacing estimate before expensive
        # row-level ownership. Other layouts keep the existing path.
        handled = refine_regular_rubbing_page(im_opencv_bgr, page_rows)
        results = []
        for index, (box_coords_quad, score) in enumerate(zip(text_boxes, detection_scores)):
            if cancel_check is not None and cancel_check():
                return []
            result = page_rows[index]['result']
            if result.get("char_box_strategy") == "stroke-ownership" and index not in handled:
                # A region rejected by both models is often a seal fragment or
                # ornament. Keep its raw recognition for diagnostics, but do
                # not manufacture an annotation from that weak agreement.
                recognition_score = result.get("score")
                if (recognition_score is not None and np.isfinite(recognition_score)
                        and float(score) < 0.60 and float(recognition_score) < 0.40):
                    result["suppressed_candidate"] = {
                        "reason": "low_detection_and_recognition_confidence",
                        "text": result.get("line_pred", ""),
                        "det_score": float(score),
                        "rec_score": float(recognition_score),
                    }
                    result["char_list"] = []
                    result["words"] = []
                    result["line_pred"] = ""
                    logger.info("忽略双低置信度文本候选：det=%.3f, rec=%.3f", score, recognition_score)
                from ocr.stroke_ownership import refine_stroke_row
                try:
                    refine_stroke_row(im_opencv_bgr, box_coords_quad, result)
                except (cv2.error, ValueError, np.linalg.LinAlgError) as error:
                    logger.warning("行 %d 的原图笔画细化失败，保留行内字框：%s", index, error)
            if result.get("char_box_strategy") in {"independent-strokes", "stroke-ownership"}:
                # Padding may extend beyond the detector row, but must stay
                # inside the original image, in the same base coordinates.
                image_h, image_w = im_opencv_bgr.shape[:2]
                for obj in result.get("char_list", []) + result.get("words", []):
                    polygon = np.asarray(obj["bbox"], dtype=np.float32).copy()
                    polygon[:, 0] = np.clip(polygon[:, 0], 0, image_w)
                    polygon[:, 1] = np.clip(polygon[:, 1], 0, image_h)
                    obj["bbox"] = polygon
            results.append({'bbox': box_coords_quad, 'result': result, 'score': score})
        return results

    def text_predict(self, pil_image_rgb, dbnet_short_size, *, cancel_check=None):
        from ocr.image import prepare_ocr_image
        if cancel_check is not None and cancel_check():
            return []
        prepared = prepare_ocr_image(pil_image_rgb, getattr(self,'ocr_image_scale',1.0),
                                     getattr(app_config,'ocr_max_side',2560))
        return self._predict_prepared(prepared, dbnet_short_size, cancel_check)

    def text_predict_file(self, path, dbnet_short_size, *, cancel_check=None):
        from ocr.image import load_ocr_image
        if cancel_check is not None and cancel_check():
            return []
        prepared = load_ocr_image(path, getattr(self,'ocr_image_scale',1.0),
                                  getattr(app_config,'ocr_max_side',2560))
        return self._predict_prepared(prepared, dbnet_short_size, cancel_check)

    def _predict_prepared(self, prepared, dbnet_short_size, cancel_check=None):
        from ocr.image import map_results_to_source
        image_rgb = np.asarray(prepared.image,dtype=np.uint8)
        if prepared.source_size != prepared.target_size:
            # Do not upscale a deliberately bounded proxy back to a huge
            # detector tensor when the requested short-side value is oversized.
            dbnet_short_size=min(dbnet_short_size,min(prepared.target_size))
        logger.info('OCR 原图 %s -> 推理 %s，检测短边 %s',
                    prepared.source_size, prepared.target_size, dbnet_short_size)
        text_boxes_detected, text_scores_detected = self.text_handle.process(
            image_rgb, dbnet_short_size
        )

        text_boxes_detected, text_scores_detected = _deduplicate_text_rows(
            text_boxes_detected, text_scores_detected)
        if cancel_check is not None and cancel_check():
            return []
        # OCR 副本给识别 crop 用：历史是 BGR。把 RGB 反转通道得到 BGR，
        # 用 ::-1 view 代替 cvtColor，零内存拷贝、结果完全一致。
        img_opencv_bgr_original = np.ascontiguousarray(image_rgb[:, :, ::-1])
        options = {"cancel_check": cancel_check} if cancel_check is not None else {}
        ocr_full_results = self.recognize_rows(
            img_opencv_bgr_original, text_boxes_detected, text_scores_detected, **options)
        return map_results_to_source(ocr_full_results, prepared)
