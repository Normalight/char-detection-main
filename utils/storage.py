# -*- coding: utf-8 -*-
"""
utils/storage.py

负责：
1. 根据图片/PDF页面 item 计算输出目录与 JSON 路径；
2. 保存/读取标注框 JSON；
3. 基于“真实处理图像坐标”保存切字图片；
4. 在批量保存时，如果 item 中没有 boxes，则从 JSON 或 OCR 结果兜底生成 boxes。

本版重点修复：
- 保存切片时不再调用 ocr.geometry.get_rotate_crop_image 中的自动旋转逻辑，避免竖排中文/碑文字框被旋转成横向扁图；
- boxes 始终位于 base_image_bgr 坐标系，即原始图片 / PDF 渲染页的坐标系；
- 画布显示缩放不写入 boxes，也不影响最终切片清晰度；
- 保存前清理旧 crop 文件，避免删除框后旧切片残留影响 Excel。
"""

from utils.paths import get_output_dir_for_item, get_json_candidates as _legacy_json_candidates

import glob
import json
import logging
import os
import re
import uuid
from typing import Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)

SUPPORTED_CROP_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")
ANNOTATION_SCHEMA_VERSION = 2
ANNOTATION_CONTEXT_KEYS = (
    "level",
    "det_model",
    "rec_model",
    "char_box_strategy",
    "char_box_revision",
    "dbnet_short_side",
    "pdf_dpi",
    "image_scale",
    "ocr_max_side",
)


def _as_box_array(points) -> Optional[np.ndarray]:
    """将四点框转换为 (4, 2) float32。"""
    try:
        arr = np.asarray(points, dtype=np.float32)
        if arr.shape == (4, 2):
            return arr if np.isfinite(arr).all() else None
        if arr.ndim == 1 and arr.size == 8:
            arr = arr.reshape(4, 2)
            return arr if np.isfinite(arr).all() else None
    except Exception as e:
        logger.debug("box 转换失败：%s", e)
        return None
    return None


def _json_box(box: np.ndarray) -> List[List[float]]:
    return [
        [float(point[0]), float(point[1])]
        for point in np.asarray(box, dtype=np.float32).tolist()
    ]


def normalize_annotation_context(context) -> dict:
    context = context if isinstance(context, dict) else {}
    return {
        key: str(context.get(key, "")).strip().lower()
        for key in ANNOTATION_CONTEXT_KEYS
        if str(context.get(key, "")).strip()
    }


def annotation_context_matches(saved_context, current_context) -> bool:
    """只有完整且逐项一致的上下文才可静默复用已有框。"""
    saved = normalize_annotation_context(saved_context)
    current = normalize_annotation_context(current_context)
    return bool(current) and all(
        saved.get(key) == value
        for key, value in current.items()
    ) and all(key in saved for key in current)


def _order_points(points: np.ndarray) -> np.ndarray:
    """
    将四点排序为：左上、右上、右下、左下。
    不做额外旋转，只保证透视变换时点序稳定。
    """
    pts = np.asarray(points, dtype=np.float32).reshape(4, 2)
    if not np.isfinite(pts).all():
        raise ValueError("box contains non-finite coordinates")

    # 旧实现用 x+y / y-x 的极值挑四角。矩形接近 45° 时会出现并列极值，
    # np.argmin/argmax 可能把同一个点同时选作 TL/TR（或 BR/BL），导致一个
    # 本来有效的菱形框被排成零宽框。绕质心按角度排序可保证每个角点只用一次。
    center = pts.mean(axis=0)
    if np.linalg.norm(pts - center, axis=1).min() <= 1e-6:
        raise ValueError("box has a point at its centroid")
    angles = np.arctan2(pts[:, 1] - center[1], pts[:, 0] - center[0])
    ordered = pts[np.argsort(angles, kind="stable")]

    # 与旧版“最小 x+y 是左上角”的方向尽量保持一致；并列时先取更靠上的点，
    # 解决标准菱形 top/left 两点 x+y 完全相同的情况。
    scores = ordered[:, 0] + ordered[:, 1]
    min_score = float(scores.min())
    candidates = np.flatnonzero(np.isclose(scores, min_score, rtol=0.0, atol=1e-5))
    start = min(candidates, key=lambda i: (float(ordered[i, 1]), float(ordered[i, 0])))
    return np.roll(ordered, -int(start), axis=0).astype(np.float32)


def _is_cropable_box(points) -> bool:
    """四点是否能形成有面积、无重复角点的透视裁剪区域。"""
    box = _as_box_array(points)
    if box is None:
        return False
    try:
        ordered = _order_points(box)
    except (TypeError, ValueError):
        return False

    edges = np.roll(ordered, -1, axis=0) - ordered
    if np.linalg.norm(edges, axis=1).min() <= 1e-3:
        return False
    x = ordered[:, 0].astype(np.float64)
    y = ordered[:, 1].astype(np.float64)
    area = 0.5 * abs(float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))
    return np.isfinite(area) and area > 1e-3


def get_rotate_crop_image(img, points):
    """
    根据四点框裁剪图像，保留原始文字方向。

    注意：这里刻意不调用 ocr/geometry.py 中的 get_rotate_crop_image，因为旧函数在
    dst_img_height / dst_img_width >= 1.5 时会执行 np.rot90(dst_img)，这会把很多
    竖排中文单字/碑文切片旋转成横向扁图，导致 Excel 中显示为一条扁长图片。
    """
    box = _as_box_array(points)
    if box is None or img is None or getattr(img, "size", 0) == 0:
        return None

    try:
        src = _order_points(box)

        width_top = np.linalg.norm(src[0] - src[1])
        width_bottom = np.linalg.norm(src[3] - src[2])
        height_left = np.linalg.norm(src[0] - src[3])
        height_right = np.linalg.norm(src[1] - src[2])

        if not _is_cropable_box(src):
            return None

        # 极窄但仍有面积的框至少生成 2px，避免目标矩形退化为 0；真正的
        # 零面积框已由 _is_cropable_box 排除。
        width = max(2, int(round(max(width_top, width_bottom))))
        height = max(2, int(round(max(height_left, height_right))))

        dst = np.array(
            [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]],
            dtype=np.float32,
        )
        mat = cv2.getPerspectiveTransform(src, dst)
        if not np.isfinite(mat).all():
            return None
        crop = cv2.warpPerspective(img, mat, (width, height), borderMode=cv2.BORDER_REPLICATE)
        return crop if crop is not None and getattr(crop, "size", 0) > 0 else None
    except Exception as e:
        logger.warning("透视裁剪失败：%s", e)
        return None


def _json_path(output_root_dir: str, item: dict) -> str:
    folder, prefix = get_output_dir_for_item(output_root_dir, item)
    return os.path.join(folder, f"{prefix}.json")


def get_json_path_for_item(output_root_dir: str, item: dict) -> str:
    """公开 JSON 路径，供其他模块调用。"""
    return _json_path(output_root_dir, item)


def save_item_json(
    output_root_dir: str,
    item: dict,
    boxes_list: Sequence[np.ndarray],
    image_size_wh: Tuple[int, int],
) -> str:
    """保存当前 item 的标注框 JSON。"""
    folder, _ = get_output_dir_for_item(output_root_dir, item)
    os.makedirs(folder, exist_ok=True)
    jp = _json_path(output_root_dir, item)

    valid_box_arrays = normalize_boxes(boxes_list)
    valid_boxes = [_json_box(box) for box in valid_box_arrays]
    raw_box_context = (
        item.get("box_context")
        if item.get("box_context") is not None
        else item.get("annotation_context")
    )
    box_context = normalize_annotation_context(raw_box_context)

    data = {
        "schema_version": ANNOTATION_SCHEMA_VERSION,
        "type": item.get("type"),
        "source_path": item.get("source_path"),
        "page_index": item.get("page_index"),
        "image_size": {"width": int(image_size_wh[0]), "height": int(image_size_wh[1])},
        "boxes": valid_boxes,
        "annotation_context": box_context,
        "coordinate_system": "base_image_after_image_scale",
        "preserve_crop_orientation": True,
        "note": "boxes are stored in the real processing image coordinate system, not the display/view scale",
    }
    temp_json_path = os.path.join(
        folder,
        f".{os.path.basename(jp)}.tmp-{uuid.uuid4().hex}",
    )
    try:
        with open(temp_json_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
            f.flush()
            os.fsync(f.fileno())
        os.replace(temp_json_path, jp)
    except Exception:
        try:
            if os.path.exists(temp_json_path):
                os.remove(temp_json_path)
        except OSError:
            pass
        raise
    return jp


def load_item_annotation_record_if_exists(
    output_root_dir: str,
    item: dict,
    target_size_wh: Optional[Tuple[int, int]] = None,
) -> Optional[dict]:
    """读取坐标框、schema 和生成上下文，并按目标尺寸同步缩放。

    Unknown JSON keys are intentionally ignored so annotation files written by
    older versions with hidden recognition metadata remain readable.
    """

    json_path = None
    for p in _legacy_json_candidates(output_root_dir, item):
        if os.path.isfile(p):
            json_path = p
            break
    if json_path is None:
        return None

    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        sx = sy = 1.0
        if target_size_wh and "image_size" in data:
            jw = max(1, int(data["image_size"].get("width", 1)))
            jh = max(1, int(data["image_size"].get("height", 1)))
            tw = max(1, int(target_size_wh[0]))
            th = max(1, int(target_size_wh[1]))
            sx, sy = tw / jw, th / jh

        boxes_loaded: List[np.ndarray] = []
        for b in data.get("boxes", []):
            arr = _as_box_array(b)
            if arr is None:
                continue
            arr = arr.copy()
            arr[:, 0] *= sx
            arr[:, 1] *= sy
            boxes_loaded.append(arr.astype(np.float32))

        return {
            "boxes": normalize_boxes(boxes_loaded),
            "schema_version": int(data.get("schema_version", 1) or 1),
            "annotation_context": normalize_annotation_context(
                data.get("annotation_context")
            ),
            "json_path": json_path,
        }
    except Exception as e:
        logger.warning("读取 JSON 失败：%s，错误：%s", json_path, e)
        return None


def load_item_annotation_if_exists(
    output_root_dir: str,
    item: dict,
    target_size_wh: Optional[Tuple[int, int]] = None,
) -> Optional[List[np.ndarray]]:
    """兼容旧调用方：只返回标注框。"""

    record = load_item_annotation_record_if_exists(
        output_root_dir,
        item,
        target_size_wh=target_size_wh,
    )
    return None if record is None else record["boxes"]


def load_item_json_if_exists(
    output_root_dir: str,
    item: dict,
    target_size_wh: Optional[Tuple[int, int]] = None,
) -> Optional[List[np.ndarray]]:
    """兼容旧调用方：只返回标注框，不暴露识别上下文。"""

    annotation = load_item_annotation_if_exists(
        output_root_dir,
        item,
        target_size_wh=target_size_wh,
    )
    return annotation


def _safe_imwrite(path: str, img: np.ndarray) -> bool:
    """支持中文路径的 OpenCV 写图。"""
    ext = os.path.splitext(path)[1].lower() or ".png"
    ok, buffer = cv2.imencode(ext, img)
    if not ok:
        return False
    with open(path, "wb") as f:
        f.write(buffer)
    return True


def _crop_filename(item: dict, name_prefix: str, index: int) -> str:
    return get_crop_filename_for_item(item, name_prefix, index)


def get_crop_filename_for_item(
    item: dict,
    name_prefix: str,
    index: int,
) -> str:
    """返回当前保存流程为指定框生成的唯一规范切片文件名。"""

    parent_folder_name = os.path.basename(os.path.dirname(item.get("source_path", ""))) or "input"
    return f"{parent_folder_name}-{name_prefix}_crop_{index}.png"


def _existing_crop_paths(out_dir: str) -> List[str]:
    if not os.path.isdir(out_dir):
        return []
    patterns = [
        "*_crop_*.png",
        "*_crop_*.jpg",
        "*_crop_*.jpeg",
        "*_crop_*.bmp",
        "*_crop_*.tif",
        "*_crop_*.tiff",
        "char_*.png",
        "char_*.jpg",
        "char_*.jpeg",
        "char_*.bmp",
        "char_*.tif",
        "char_*.tiff",
    ]
    paths = set()
    for pattern in patterns:
        paths.update(glob.glob(os.path.join(out_dir, pattern)))
    return sorted(paths)


def clear_existing_crops(out_dir: str) -> None:
    """保存前清理旧 crop，避免删除/减少框后旧图片残留。"""
    for path in _existing_crop_paths(out_dir):
        try:
            os.remove(path)
        except OSError as e:
            logger.warning("删除旧 crop 失败：%s，错误：%s", path, e)


def normalize_boxes(boxes: Iterable) -> List[np.ndarray]:
    out: List[np.ndarray] = []
    for b in boxes or []:
        arr = _as_box_array(b)
        if arr is not None and _is_cropable_box(arr):
            out.append(arr.astype(np.float32))
    return out


def _split_quad_along_long_axis(box, count: int) -> List[np.ndarray]:
    """把有序四边形沿长轴等分，兼容横排、竖排和旋转文本。"""
    if count < 2:
        return []
    try:
        ordered = _order_points(np.asarray(box, dtype=np.float32))
    except (TypeError, ValueError):
        return []

    width = max(
        np.linalg.norm(ordered[0] - ordered[1]),
        np.linalg.norm(ordered[3] - ordered[2]),
    )
    height = max(
        np.linalg.norm(ordered[0] - ordered[3]),
        np.linalg.norm(ordered[1] - ordered[2]),
    )
    boxes = []
    for index in range(count):
        start = index / float(count)
        end = (index + 1) / float(count)
        if width >= height:
            top_start = ordered[0] + start * (ordered[1] - ordered[0])
            top_end = ordered[0] + end * (ordered[1] - ordered[0])
            bottom_start = ordered[3] + start * (ordered[2] - ordered[3])
            bottom_end = ordered[3] + end * (ordered[2] - ordered[3])
            part = np.asarray(
                [top_start, top_end, bottom_end, bottom_start],
                dtype=np.float32,
            )
        else:
            left_start = ordered[0] + start * (ordered[3] - ordered[0])
            left_end = ordered[0] + end * (ordered[3] - ordered[0])
            right_start = ordered[1] + start * (ordered[2] - ordered[1])
            right_end = ordered[1] + end * (ordered[2] - ordered[1])
            part = np.asarray(
                [left_start, right_start, right_end, left_end],
                dtype=np.float32,
            )
        if _is_cropable_box(part):
            boxes.append(part)
    return boxes


def _fallback_char_boxes_for_underrecognized_cjk(text_block, char_boxes):
    """严重少识别的长中文行/列使用几何等分，避免一个框吞掉整行。"""
    if (text_block.get("result") or {}).get("char_box_strategy") in {"independent-strokes", "stroke-ownership"}:
        # Keep independent extents; never replace them with equal row slices.
        return None
    line_box = _as_box_array(text_block.get("bbox"))
    if line_box is None:
        return None
    ordered = _order_points(line_box)
    width = max(
        np.linalg.norm(ordered[0] - ordered[1]),
        np.linalg.norm(ordered[3] - ordered[2]),
    )
    height = max(
        np.linalg.norm(ordered[0] - ordered[3]),
        np.linalg.norm(ordered[1] - ordered[2]),
    )
    short_side = min(width, height)
    if short_side <= 1e-3:
        return None
    aspect_ratio = max(width, height) / short_side
    if aspect_ratio < 3.0:
        return None

    result = text_block.get("result", {}) or {}
    line_pred = str(result.get("line_pred", ""))
    if not re.search(r"[\u3400-\u9fff]", line_pred):
        return None

    token_count = len(re.findall(r"\S", line_pred))
    char_items = result.get("char_list", []) or []
    scores = []
    for char_item in char_items:
        if not isinstance(char_item, dict):
            continue
        try:
            score = float(char_item.get("score"))
        except (TypeError, ValueError):
            continue
        if np.isfinite(score):
            scores.append(score)
    try:
        result_score = float(result.get("score"))
    except (TypeError, ValueError):
        result_score = float("nan")
    confidence = (
        float(np.mean(scores))
        if scores
        else result_score if np.isfinite(result_score) else None
    )

    # 单个高置信度字符放在留白较大的检测框内是合理结果；没有置信度时
    # 也不冒险凭长宽比批量制造框。只有明确低置信度时才允许有限兜底。
    if token_count <= 1:
        if confidence is None or confidence >= 0.7:
            return None
        count_limit = 4
    else:
        count_limit = max(
            token_count + 2,
            int(np.ceil(token_count * 1.5)),
        )

    # 两个以上模型框可提供字符间距信号，用中心间距进一步收紧上限。
    if len(char_boxes) >= 2:
        long_axis = (
            ordered[1] - ordered[0]
            if width >= height
            else ordered[3] - ordered[0]
        )
        axis_length = float(np.linalg.norm(long_axis))
        if axis_length > 1e-3:
            axis = long_axis / axis_length
            projections = sorted(
                float(np.dot(np.mean(box, axis=0) - ordered[0], axis))
                for box in char_boxes
            )
            gaps = [
                right - left
                for left, right in zip(projections, projections[1:])
                if right - left > 1e-3
            ]
            if gaps:
                spacing_count = max(
                    len(char_boxes),
                    int(round(axis_length / float(np.median(gaps)))),
                )
                count_limit = min(count_limit, spacing_count)

    expected_count = min(
        512,
        max(2, int(round(aspect_ratio))),
        count_limit,
    )
    if expected_count <= len(char_boxes):
        return None
    if len(char_boxes) >= max(2, int(round(expected_count * 0.6))):
        return None

    fallback = _split_quad_along_long_axis(line_box, expected_count)
    if len(fallback) != expected_count:
        return None
    logger.warning(
        "char 模式检测到低置信度长中文行严重少识别，使用受限几何兜底拆分：%d → %d",
        len(char_boxes),
        expected_count,
    )
    result["char_box_fallback"] = {
        "source": "bounded_geometry",
        "model_box_count": len(char_boxes),
        "output_box_count": expected_count,
    }
    return fallback


def extract_boxes_from_ocr_results(all_results, level: str = "char") -> List[np.ndarray]:
    """从 OcrHandle.text_predict 的返回结果中提取指定层级的 boxes。"""
    level = (level or "char").strip().lower()
    boxes: List[np.ndarray] = []

    for text_block in all_results or []:
        if level == "line":
            arr = _as_box_array(text_block.get("bbox"))
            if arr is not None:
                boxes.append(arr.astype(np.float32))
            continue

        result = text_block.get("result", {}) or {}
        candidates = result.get("words", []) if level == "word" else result.get("char_list", [])
        block_boxes = []
        for obj in candidates or []:
            if not isinstance(obj, dict):
                continue
            arr = _as_box_array(obj.get("bbox"))
            if arr is not None:
                block_boxes.append(arr.astype(np.float32))
        if level == "char":
            fallback = _fallback_char_boxes_for_underrecognized_cjk(
                text_block,
                block_boxes,
            )
            if fallback is not None:
                block_boxes = fallback
        boxes.extend(block_boxes)
    return boxes


def save_current_slices_and_json(output_root_dir: str, item: dict, base_image_bgr: np.ndarray, boxes: Sequence[np.ndarray], *, progress_callback=None, crop_callback=None) -> int:
    """
    保存当前页面切片和 JSON。

    注意：base_image_bgr 必须是真实处理图像，不是显示缩放后的图像。
    """
    if base_image_bgr is None or getattr(base_image_bgr, "size", 0) == 0:
        logger.error("保存失败：base_image_bgr 为空。")
        return -1

    h, w = base_image_bgr.shape[:2]
    out_dir, name_prefix = get_output_dir_for_item(output_root_dir, item)
    os.makedirs(out_dir, exist_ok=True)

    valid_boxes = normalize_boxes(boxes)
    transaction_id = uuid.uuid4().hex
    staged_paths = []
    final_paths = []
    backup_paths = []
    committed_paths = []
    old_item_boxes = item.get("boxes")
    try:
        # 全部新切片先写到同目录临时文件。此阶段失败时旧切片和 JSON
        # 完全没有变化。
        # Keep only one full-resolution crop in memory at a time. All staged
        # files still precede the transaction commit, preserving rollback.
        for i, box in enumerate(valid_boxes):
            crop = get_rotate_crop_image(base_image_bgr, box.copy())
            if crop is None or getattr(crop, 'size', 0) == 0:
                raise RuntimeError(f'box {i} 透视裁剪失败：{box.tolist()}')
            final_path = os.path.join(
                out_dir,
                _crop_filename(item, name_prefix, i),
            )
            staged_path = os.path.join(
                out_dir,
                f".crop-stage-{transaction_id}-{i}.png",
            )
            if not _safe_imwrite(staged_path, crop):
                raise RuntimeError(f"切片写入失败：{final_path}")
            staged_paths.append(staged_path)
            final_paths.append(final_path)
            if crop_callback is not None:
                crop_callback(i, crop, box)
            if progress_callback is not None:
                progress_callback(i+1,len(valid_boxes))

        # 先把旧切片移到同目录备份，再原子替换新切片。后续任一步失败
        # 都可在当前进程内恢复上一版输出。
        for i, old_path in enumerate(_existing_crop_paths(out_dir)):
            backup_path = os.path.join(
                out_dir,
                f".crop-backup-{transaction_id}-{i}.bak",
            )
            os.replace(old_path, backup_path)
            backup_paths.append((old_path, backup_path))

        for staged_path, final_path in zip(staged_paths, final_paths):
            os.replace(staged_path, final_path)
            committed_paths.append(final_path)

        save_item_json(
            output_root_dir,
            item,
            valid_boxes,
            (w, h),
        )
        item["boxes"] = [b.copy() for b in valid_boxes]
    except Exception:
        item["boxes"] = old_item_boxes
        for path in committed_paths:
            try:
                if os.path.exists(path):
                    os.remove(path)
            except OSError as rollback_error:
                logger.error("回滚新切片失败：%s，错误：%s", path, rollback_error)
        for old_path, backup_path in reversed(backup_paths):
            try:
                if os.path.exists(backup_path):
                    os.replace(backup_path, old_path)
            except OSError as rollback_error:
                logger.error("恢复旧切片失败：%s，错误：%s", old_path, rollback_error)
        raise
    finally:
        for path in staged_paths:
            try:
                if os.path.exists(path):
                    os.remove(path)
            except OSError:
                pass

    for _old_path, backup_path in backup_paths:
        try:
            if os.path.exists(backup_path):
                os.remove(backup_path)
        except OSError as e:
            logger.warning("清理切片事务备份失败：%s，错误：%s", backup_path, e)

    saved = len(valid_boxes)
    logger.info("操作完成：成功保存 %d / %d 个图片切片。", saved, len(valid_boxes))
    return saved


def save_item_boxes_to_disk(
    output_root_dir: str,
    item: dict,
    base_pil_image,
    ocr_handler=None,
    dbnet_short_side: int = 960,
    level: str = "char",
    pdf_dpi: int = 200,
    poppler_path: Optional[str] = None,
) -> int:
    """
    批量保存单个 item。

    若 item['boxes'] 不存在：
    1. 优先读取 JSON；
    2. 若 JSON 不存在且提供 ocr_handler，则运行 OCR 获取 boxes；
    3. 最后仍为空则保存空 JSON 并返回 0。
    """
    if base_pil_image is None:
        logger.warning("%s base_pil_image 为空，跳过。", item.get('display_name', ''))
        return 0

    base_bgr = cv2.cvtColor(np.array(base_pil_image.convert("RGB")), cv2.COLOR_RGB2BGR)
    w0, h0 = base_pil_image.size

    boxes = normalize_boxes(item.get("boxes", []))
    authoritative_empty_boxes = (
        "boxes" in item
        and bool(item.get("visited"))
        and bool(item.get("dirty"))
    )
    if not boxes and not authoritative_empty_boxes:
        record = load_item_annotation_record_if_exists(
            output_root_dir,
            item,
            target_size_wh=(w0, h0),
        )
        current_context = item.get("annotation_context", {})
        if record is not None and annotation_context_matches(
            record.get("annotation_context"),
            current_context,
        ):
            boxes = normalize_boxes(record.get("boxes"))
            item["box_context"] = normalize_annotation_context(
                record.get("annotation_context")
            )
        elif ocr_handler is not None:
            try:
                all_results = ocr_handler.text_predict(base_pil_image.convert("RGB"), dbnet_short_side)
                boxes = extract_boxes_from_ocr_results(all_results, level)
                item["box_context"] = normalize_annotation_context(
                    current_context
                )
            except Exception as e:
                display_name = item.get("display_name", "")
                raise RuntimeError(
                    f"{display_name} OCR 失败，无法完整保存：{e}"
                ) from e

    item["boxes"] = [b.copy() for b in boxes]
    item["view_scale"] = item.get("view_scale", item.get("nav_scale", None))
    item["nav_scale"] = 1.0  # 兼容旧字段；不再用于真实裁剪。
    return save_current_slices_and_json(output_root_dir, item, base_bgr, boxes)
