# -*- coding: utf-8 -*-
"""
excel_utils.py

Excel 导出：中文横/竖排对齐、防重叠、避免扁图、缩略图控体积。

要点：
1. 单元格行列不按 center_x // max_w、center_y // max_h 直接整除映射——多个字符会落入同一单元格；
2. 竖排模式下同时按“列”和“行”全局聚类，尽量保持中文竖排的横向行对齐；
3. 横排模式下同时按“行”和“列”全局聚类，尽量保持普通中文横排版式；
4. 若旧切片已经被错误旋转成横向扁图，会结合 JSON 中 bbox 的宽高关系在 Excel 缩略图阶段自动纠正显示方向；
5. 单元格统一设置为正方形，图片等比例缩放并居中，不拉伸、不压扁；
6. 插入缩略图而不是原始大图，降低一两千张图时的 xlsx 体积。
"""

import glob
import io
import json
import logging
import math
import os
import re
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image, ImageOps

logger = logging.getLogger(__name__)

try:
    import xlsxwriter
    HAS_XLSX = True
except ImportError:
    HAS_XLSX = False


# =========================
# 可根据需要调整的默认参数
# =========================

# 古籍/碑文通常为竖排，从右到左读；如果是现代横排，可改为 horizontal_ltr。
DEFAULT_READING_MODE = "vertical_rtl"

# aligned：按全局行列坐标对齐，适合碑文/古籍；compact：每列/每行压缩排布，空格更少但对齐弱。
DEFAULT_LAYOUT_POLICY = "aligned"

# Excel 中每个字符单元格尺寸。越大越清晰，但 xlsx 也会更大。建议 72~96。
DEFAULT_CELL_SIZE_PX = 84
DEFAULT_PADDING_PX = 5

# 缩略图格式。PNG 清晰；若 Excel 过大可改为 JPEG。
DEFAULT_THUMB_FORMAT = "PNG"
DEFAULT_JPEG_QUALITY = 86

DEFAULT_ADD_INFO_SHEET = True

# 聚类阈值系数。值越大，越容易把相近 y/x 位置归为同一行/列。
DEFAULT_X_CLUSTER_TOLERANCE = 0.85
DEFAULT_Y_CLUSTER_TOLERANCE = 0.85

# 如果旧切片方向与 bbox 方向明显相反，在 Excel 缩略图中自动纠正。
AUTO_FIX_THUMB_ORIENTATION = True
ORIENTATION_FIX_RATIO = 1.35

_INVALID_SHEET_CHARS = '[]:*?/\\'
_TRANS_TABLE = str.maketrans({ch: '_' for ch in _INVALID_SHEET_CHARS})


# =========================
# 路径与文件读取
# =========================

def get_item_folder(output_root_dir, item):
    """获取该 item 对应的输出文件夹路径。保留原函数名，兼容 annotator_core.py。"""
    folder, _ = get_item_folder_and_prefix(output_root_dir, item)
    return folder


def get_item_folder_and_prefix(output_root_dir, item) -> Tuple[str, str]:
    """返回 item 输出文件夹和 JSON/切片常用前缀。逻辑与 storage_utils.py 保持一致。"""
    if item["type"] == "image":
        stem = os.path.splitext(os.path.basename(item["source_path"]))[0]
        return os.path.join(output_root_dir, stem), stem

    pdf_stem = os.path.splitext(os.path.basename(item["source_path"]))[0]
    page_folder = f"page_{item['page_index'] + 1:03d}"
    prefix = f"{pdf_stem}_p{item['page_index'] + 1:03d}"
    return os.path.join(output_root_dir, pdf_stem, page_folder), prefix


def get_json_candidates(output_root_dir, item) -> List[str]:
    """给出可能的 JSON 路径，兼容新版与旧版命名方式。"""
    folder, prefix = get_item_folder_and_prefix(output_root_dir, item)
    candidates = [
        os.path.join(folder, f"{prefix}.json"),
        os.path.join(folder, f"{os.path.basename(folder)}.json"),
    ]
    out, seen = [], set()
    for p in candidates:
        if p not in seen:
            out.append(p)
            seen.add(p)
    return out


def load_boxes_from_json(output_root_dir, item) -> Tuple[List, Optional[str]]:
    for json_path in get_json_candidates(output_root_dir, item):
        if not os.path.isfile(json_path):
            continue
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return data.get("boxes", []), json_path
        except Exception as e:
            logger.warning("[Excel] 读取 JSON 失败: %s，错误: %s", json_path, e)
            return [], json_path
    return [], None


def parse_crop_index(path: str) -> Optional[int]:
    name = os.path.basename(path)
    m = re.search(r"_crop_(\d+)\.(png|jpg|jpeg|bmp|tif|tiff)$", name, flags=re.IGNORECASE)
    if m:
        return int(m.group(1))
    m = re.search(r"char_(\d+)\.(png|jpg|jpeg|bmp|tif|tiff)$", name, flags=re.IGNORECASE)
    if m:
        return max(0, int(m.group(1)) - 1)
    return None


def list_saved_crops_for_item(folder):
    """在指定文件夹中查找所有 crop/char 图片，并按编号排序。"""
    if not os.path.isdir(folder):
        return []

    patterns = [
        os.path.join(folder, "*_crop_*.png"), os.path.join(folder, "*_crop_*.jpg"),
        os.path.join(folder, "*_crop_*.jpeg"), os.path.join(folder, "*_crop_*.bmp"),
        os.path.join(folder, "*_crop_*.tif"), os.path.join(folder, "*_crop_*.tiff"),
        os.path.join(folder, "char_*.png"), os.path.join(folder, "char_*.jpg"),
        os.path.join(folder, "char_*.jpeg"),
    ]
    files = []
    for pattern in patterns:
        files.extend(glob.glob(pattern))

    def _sort_key(path):
        crop_index = parse_crop_index(path)
        return (
            crop_index is None,
            crop_index if crop_index is not None else 10**9,
            path,
        )

    return sorted(set(files), key=_sort_key)


# =========================
# 坐标与排序逻辑
# =========================

def _box_to_np(box) -> Optional[np.ndarray]:
    try:
        arr = np.array(box, dtype=np.float32)
        if arr.shape == (4, 2):
            return arr
        if arr.ndim == 1 and arr.size == 8:
            return arr.reshape(4, 2)
    except (ValueError, TypeError):
        # 输入不是合法的 box（None / 字符串 / 形状错），按"无框"处理
        return None
    return None


def _box_features(box) -> Optional[Dict[str, float]]:
    arr = _box_to_np(box)
    if arr is None:
        return None
    xs = arr[:, 0]
    ys = arr[:, 1]
    w = float(max(1.0, np.max(xs) - np.min(xs)))
    h = float(max(1.0, np.max(ys) - np.min(ys)))
    return {
        "cx": float(np.mean(xs)), "cy": float(np.mean(ys)), "w": w, "h": h,
        "x_min": float(np.min(xs)), "x_max": float(np.max(xs)),
        "y_min": float(np.min(ys)), "y_max": float(np.max(ys)),
    }


def _cluster_axis(records: List[Dict], axis_key: str, span_key: str, reverse: bool, tolerance: float) -> List[Dict]:
    """
    按 x 或 y 坐标聚类，返回若干组：[{center, records}]。
    """
    if not records:
        return []

    spans = [max(1.0, float(r.get(span_key, 1.0))) for r in records]
    median_span = float(np.median(spans)) if spans else 12.0
    threshold = max(8.0, median_span * float(tolerance))

    ordered = sorted(records, key=lambda r: float(r[axis_key]), reverse=reverse)
    groups: List[Dict] = []

    for rec in ordered:
        value = float(rec[axis_key])
        best_i, best_d = -1, None
        for i, group in enumerate(groups):
            d = abs(value - group["center"])
            if d <= threshold and (best_d is None or d < best_d):
                best_i, best_d = i, d
        if best_i >= 0:
            groups[best_i]["records"].append(rec)
            groups[best_i]["center"] = float(np.mean([float(r[axis_key]) for r in groups[best_i]["records"]]))
        else:
            groups.append({"center": value, "records": [rec]})

    groups.sort(key=lambda g: g["center"], reverse=reverse)
    return groups


def _assign_by_aligned_grid(records: List[Dict], reading_mode: str) -> List[Dict]:
    """
    全局行列对齐排布，单元格保持原图的空间方向。

    reading_mode 只决定 order_index，不镜像 Excel 中可见的原图版面。
    """
    mode = (reading_mode or DEFAULT_READING_MODE).strip().lower()

    if mode not in (
        "vertical_rtl",
        "vertical_ltr",
        "horizontal_ltr",
        "horizontal_rtl",
    ):
        return []

    x_groups = _cluster_axis(
        records,
        "cx",
        "w",
        reverse=False,
        tolerance=DEFAULT_X_CLUSTER_TOLERANCE,
    )
    y_groups = _cluster_axis(
        records,
        "cy",
        "h",
        reverse=False,
        tolerance=DEFAULT_Y_CLUSTER_TOLERANCE,
    )

    x_id = {}
    for col_idx, group in enumerate(x_groups):
        for rec in group["records"]:
            x_id[id(rec)] = col_idx

    y_id = {}
    for row_idx, group in enumerate(y_groups):
        for rec in group["records"]:
            y_id[id(rec)] = row_idx

    out: List[Dict] = []
    occupied = set()

    # 阅读顺序仅用于 order_index；实际 row/col 用全局聚类结果。
    if mode == "vertical_rtl":
        ordered = sorted(
            records,
            key=lambda r: (
                -x_id[id(r)],
                y_id[id(r)],
                r.get("crop_index", 10**9),
            ),
        )
    elif mode == "vertical_ltr":
        ordered = sorted(
            records,
            key=lambda r: (
                x_id[id(r)],
                y_id[id(r)],
                r.get("crop_index", 10**9),
            ),
        )
    elif mode == "horizontal_rtl":
        ordered = sorted(
            records,
            key=lambda r: (
                y_id[id(r)],
                -x_id[id(r)],
                r.get("crop_index", 10**9),
            ),
        )
    else:
        ordered = sorted(
            records,
            key=lambda r: (
                y_id[id(r)],
                x_id[id(r)],
                r.get("crop_index", 10**9),
            ),
        )

    for order, rec in enumerate(ordered):
        row = int(y_id.get(id(rec), 0))
        col = int(x_id.get(id(rec), 0))

        # 避免同一行列中因重复检测或聚类误差导致重叠。
        if (row, col) in occupied:
            base_row = row
            while (row, col) in occupied:
                row += 1
            rec["excel_collision_from_row"] = base_row
        occupied.add((row, col))

        rec["excel_row"] = row
        rec["excel_col"] = col
        rec["order_index"] = order
        out.append(rec)
    return out


def _assign_by_compact(records: List[Dict], reading_mode: str) -> List[Dict]:
    """压缩排布：适合不需要保留空格的情况。"""
    mode = (reading_mode or DEFAULT_READING_MODE).strip().lower()
    out: List[Dict] = []

    if mode in ("vertical_rtl", "vertical_ltr"):
        columns = _cluster_axis(records, "cx", "w", reverse=(mode == "vertical_rtl"), tolerance=DEFAULT_X_CLUSTER_TOLERANCE)
        order = 0
        for col_idx, group in enumerate(columns):
            col_records = sorted(group["records"], key=lambda r: (r["cy"], r["cx"], r.get("crop_index", 10**9)))
            for row_idx, rec in enumerate(col_records):
                rec["excel_row"] = row_idx
                rec["excel_col"] = col_idx
                rec["order_index"] = order
                order += 1
                out.append(rec)
        return out

    if mode in ("horizontal_ltr", "horizontal_rtl"):
        rows = _cluster_axis(records, "cy", "h", reverse=False, tolerance=DEFAULT_Y_CLUSTER_TOLERANCE)
        order = 0
        for row_idx, group in enumerate(rows):
            row_records = sorted(group["records"], key=lambda r: r["cx"], reverse=(mode == "horizontal_rtl"))
            for col_idx, rec in enumerate(row_records):
                rec["excel_row"] = row_idx
                rec["excel_col"] = col_idx
                rec["order_index"] = order
                order += 1
                out.append(rec)
        return out

    return []


def assign_excel_positions(records: List[Dict], reading_mode: str = DEFAULT_READING_MODE,
                           layout_policy: str = DEFAULT_LAYOUT_POLICY) -> List[Dict]:
    """
    根据中文横/竖排规则分配 Excel 行列。核心保证：每条记录唯一单元格，不重叠。
    """
    if not records:
        return []

    mode = (reading_mode or DEFAULT_READING_MODE).strip().lower()
    policy = (layout_policy or DEFAULT_LAYOUT_POLICY).strip().lower()

    has_position = all(r.get("has_box", False) for r in records)
    if mode == "sequential" or not has_position:
        columns = 20
        out = []
        for i, rec in enumerate(sorted(records, key=lambda r: (r.get("crop_index", 10**9), r.get("path", "")))):
            rec["excel_row"] = i // columns
            rec["excel_col"] = i % columns
            rec["order_index"] = i
            out.append(rec)
        return out

    if policy == "compact":
        out = _assign_by_compact(records, mode)
    else:
        out = _assign_by_aligned_grid(records, mode)

    if out:
        return out

    logger.warning("[Excel] 未识别 reading_mode='%s'，已按 vertical_rtl 处理。", reading_mode)
    return _assign_by_aligned_grid(records, "vertical_rtl")


# =========================
# 图片缩略图与 Excel 插入
# =========================

def px_to_col_width(px):
    return max(1, px / 7.0)


def sanitize_sheet_name(name: str) -> str:
    s = (name or "Sheet").translate(_TRANS_TABLE).strip()
    if not s:
        s = "Sheet"
    return s[:31]


def _maybe_fix_orientation(im: Image.Image, rec: Dict) -> Tuple[Image.Image, str]:
    """
    若 bbox 是竖向而切片图片却是横向扁图，说明旧裁剪中可能发生了自动 np.rot90；
    在 Excel 缩略图阶段反向旋转回来，避免显示为一条扁图。
    """
    if not AUTO_FIX_THUMB_ORIENTATION or not rec.get("has_box", False):
        return im, "none"

    try:
        box_w = float(rec.get("w") or 1.0)
        box_h = float(rec.get("h") or 1.0)
        img_w, img_h = im.size

        # 原 bbox 明显竖向，但实际切片明显横向：反向旋转旧函数的 np.rot90。
        if box_h > box_w * ORIENTATION_FIX_RATIO and img_w > img_h * ORIENTATION_FIX_RATIO:
            return im.transpose(Image.Transpose.ROTATE_270), "rotate_90_cw"

        # 原 bbox 明显横向，但实际切片明显竖向：也做一次方向修正，兼容横排文本。
        if box_w > box_h * ORIENTATION_FIX_RATIO and img_h > img_w * ORIENTATION_FIX_RATIO:
            return im.transpose(Image.Transpose.ROTATE_90), "rotate_90_ccw"
    except (ValueError, TypeError, AttributeError) as e:
        logger.debug("方向纠偏失败，跳过：%s", e)

    return im, "none"


def _make_thumbnail_bytes(path: str, rec: Dict, max_side_px: int,
                          thumb_format: str = DEFAULT_THUMB_FORMAT,
                          jpeg_quality: int = DEFAULT_JPEG_QUALITY) -> Tuple[io.BytesIO, int, int, int, int, str, str]:
    """
    读取原切片，生成等比例缩略图 bytes。
    返回：bytes_io, thumb_w, thumb_h, original_w, original_h, image_ext, orientation_fix
    """
    with open(path, "rb") as f:
        raw = f.read()

    with Image.open(io.BytesIO(raw)) as im:
        im = ImageOps.exif_transpose(im)
        return _thumbnail_from_image(im, rec, max_side_px, thumb_format, jpeg_quality)


def _thumbnail_from_image(im, rec, max_side_px, thumb_format, jpeg_quality):
    original_w, original_h = im.size
    im, orientation_fix = _maybe_fix_orientation(im, rec)

    max_side_px = max(16, int(max_side_px))
    im.thumbnail((max_side_px, max_side_px), Image.Resampling.LANCZOS)

    fmt = (thumb_format or DEFAULT_THUMB_FORMAT).strip().upper()
    bio = io.BytesIO()

    if fmt in ("JPG", "JPEG"):
        if im.mode in ("RGBA", "LA"):
            bg = Image.new("RGB", im.size, (255, 255, 255))
            alpha = im.getchannel("A") if im.mode == "RGBA" else im.getchannel(1)
            bg.paste(im.convert("RGB"), mask=alpha)
            im = bg
        else:
            im = im.convert("RGB")
        im.save(bio, format="JPEG", quality=int(jpeg_quality), optimize=True)
        ext = "jpg"
    else:
        if im.mode not in ("RGB", "RGBA", "L"):
            im = im.convert("RGB")
        im.save(bio, format="PNG", optimize=True)
        ext = "png"

    bio.seek(0)
    thumb_w, thumb_h = im.size
    return bio, thumb_w, thumb_h, original_w, original_h, ext, orientation_fix


def make_crop_thumbnail(crop_bgr, box):
    """Keep only a bounded Excel thumbnail while an original crop is in memory."""
    im = Image.fromarray(crop_bgr[:, :, ::-1])
    rec = dict(_box_features(box) or {}, has_box=True)
    return _thumbnail_from_image(
        im, rec, DEFAULT_CELL_SIZE_PX - 2 * DEFAULT_PADDING_PX,
        DEFAULT_THUMB_FORMAT, DEFAULT_JPEG_QUALITY)


def _safe_set_cell_size(ws, rows: int, cols: int, cell_px: int):
    rows = max(1, int(rows))
    cols = max(1, int(cols))
    cell_px = max(24, int(cell_px))

    set_col_pixels = getattr(ws, "set_column_pixels", None)
    set_row_pixels = getattr(ws, "set_row_pixels", None)

    if set_col_pixels:
        ws.set_column_pixels(0, cols - 1, cell_px)
    else:
        ws.set_column(0, cols - 1, px_to_col_width(cell_px))

    try:
        ws.set_default_row(cell_px * 0.75)
    except AttributeError:
        # 旧版 xlsxwriter 没有 set_default_row，跳过即可
        pass

    for r in range(rows):
        if set_row_pixels:
            ws.set_row_pixels(r, cell_px)
        else:
            ws.set_row(r, cell_px * 0.75)


def _build_records(output_root_dir, item) -> Tuple[List[Dict], Optional[str], str]:
    folder, _ = get_item_folder_and_prefix(output_root_dir, item)
    boxes_from_json, json_path = load_boxes_from_json(output_root_dir, item)
    crops_paths = list_saved_crops_for_item(folder)

    if not crops_paths:
        return [], json_path, folder

    records: List[Dict] = []
    for seq_idx, path in enumerate(crops_paths):
        crop_idx = parse_crop_index(path)
        if crop_idx is None:
            crop_idx = seq_idx

        box = None
        if 0 <= crop_idx < len(boxes_from_json):
            box = boxes_from_json[crop_idx]
        elif seq_idx < len(boxes_from_json):
            box = boxes_from_json[seq_idx]

        feat = _box_features(box) if box is not None else None
        rec = {
            "path": path,
            "crop_index": crop_idx,
            "box": box,
            "has_box": feat is not None,
            "cx": feat["cx"] if feat else float(seq_idx),
            "cy": feat["cy"] if feat else 0.0,
            "w": feat["w"] if feat else 1.0,
            "h": feat["h"] if feat else 1.0,
            "x_min": feat["x_min"] if feat else None,
            "x_max": feat["x_max"] if feat else None,
            "y_min": feat["y_min"] if feat else None,
            "y_max": feat["y_max"] if feat else None,
        }
        records.append(rec)

    if boxes_from_json and len(crops_paths) != len(boxes_from_json):
        logger.warning(
            "[Excel] 提示: %s 切片数(%d) 与 JSON 框数(%d) 不一致，已按 crop 编号尽量对齐生成。",
            item.get('display_name', ''), len(crops_paths), len(boxes_from_json),
        )
    elif not boxes_from_json:
        logger.info("[Excel] 提示: 未找到有效 JSON 坐标，将按文件编号顺序排布: %s", folder)

    return records, json_path, folder


# =========================
# 主入口：保持原调用方式兼容
# =========================

def build_excel_for_single_item(
    output_root_dir,
    item,
    workbook_name="crops.xlsx",
    reading_mode: str = DEFAULT_READING_MODE,
    layout_policy: str = DEFAULT_LAYOUT_POLICY,
    cell_size_px: int = DEFAULT_CELL_SIZE_PX,
    padding_px: int = DEFAULT_PADDING_PX,
    thumb_format: str = DEFAULT_THUMB_FORMAT,
    jpeg_quality: int = DEFAULT_JPEG_QUALITY,
    add_info_sheet: bool = DEFAULT_ADD_INFO_SHEET,
    *, cached_thumbnails=None,
):
    """
    为单个 Item 生成独立 Excel，保存在该 Item 的输出文件夹下。

    reading_mode 可选：
    - vertical_rtl：竖排，正文阅读顺序为右起左行。
    - vertical_ltr：竖排，左起右行。
    - horizontal_ltr：横排，从左到右。
    - horizontal_rtl：横排，从右到左。
    - sequential：不看坐标，只按 crop 编号顺序网格排布。

    layout_policy：
    - aligned：保留原图上下左右位置，推荐与古籍/碑文原图并排核对；
    - compact：压缩空白，适合只关心阅读顺序。
    """
    if not HAS_XLSX:
        logger.warning("[Excel] 警告：未安装 xlsxwriter，跳过生成。请执行 pip install xlsxwriter")
        return

    folder, _ = get_item_folder_and_prefix(output_root_dir, item)
    if not os.path.isdir(folder):
        return

    records, json_path, folder = _build_records(output_root_dir, item)
    if not records:
        return

    laid_out = assign_excel_positions(records, reading_mode=reading_mode, layout_policy=layout_policy)
    if not laid_out:
        return

    max_row = max(int(r["excel_row"]) for r in laid_out)
    max_col = max(int(r["excel_col"]) for r in laid_out)
    row_count = max_row + 1
    col_count = max_col + 1

    if row_count > 1_048_576 or col_count > 16_384:
        logger.warning("[Excel] 警告: 排布超出 Excel 限制 row=%d, col=%d，已跳过。", row_count, col_count)
        return

    excel_path = os.path.join(folder, workbook_name)
    cell_size_px = max(24, int(cell_size_px))
    padding_px = max(0, int(padding_px))
    max_thumb_side = max(16, cell_size_px - padding_px * 2)

    try:
        # 不启用 constant_memory，避免大量图片/行高设置在部分 Excel/WPS 环境中显示异常。
        # Thumbnails are already bounded in memory; avoid one temporary file
        # per embedded image (hundreds of slow opens on Windows).
        wb = xlsxwriter.Workbook(excel_path, {"in_memory": True})
        ws = wb.add_worksheet("Crops")
        ws.hide_gridlines(0)
        _safe_set_cell_size(ws, row_count, col_count, cell_size_px)

        occupied = set()
        orientation_fixed_count = 0
        skipped = 0

        for rec in sorted(laid_out, key=lambda r: (r.get("excel_row", 0), r.get("excel_col", 0), r.get("order_index", 0))):
            row = int(rec["excel_row"])
            col = int(rec["excel_col"])

            if (row, col) in occupied:
                row = row_count + skipped
                skipped += 1
                _safe_set_cell_size(ws, row + 1, max(col_count, col + 1), cell_size_px)
            occupied.add((row, col))

            try:
                thumbnail = (cached_thumbnails or {}).get(rec["crop_index"])
                if thumbnail is None:
                    thumbnail = _make_thumbnail_bytes(
                        rec["path"], rec, max_thumb_side,
                        thumb_format=thumb_format, jpeg_quality=jpeg_quality)
                image_data, tw, th, ow, oh, ext, orientation_fix = thumbnail
            except Exception as e:
                logger.warning("[Excel] 图片缩略图生成失败，已跳过: %s，错误: %s", rec['path'], e)
                continue

            if orientation_fix != "none":
                orientation_fixed_count += 1
                rec["orientation_fix"] = orientation_fix
            else:
                rec["orientation_fix"] = "none"

            rec["original_w"] = ow
            rec["original_h"] = oh
            rec["thumb_w"] = tw
            rec["thumb_h"] = th

            x_offset = max(0, (cell_size_px - tw) // 2)
            y_offset = max(0, (cell_size_px - th) // 2)
            display_name = f"crop_{int(rec.get('crop_index', 0)):05d}.{ext}"

            ws.insert_image(row, col, display_name, {
                "image_data": image_data,
                "x_offset": x_offset,
                "y_offset": y_offset,
                "object_position": 2,  # move but don't size with cells，避免被行高/列宽变化拉伸。
                "description": os.path.basename(rec["path"]),
            })

        if add_info_sheet:
            info = wb.add_worksheet("Info")
            headers = [
                "order", "excel_row", "excel_col", "crop_index", "file",
                "cx", "cy", "w", "h", "x_min", "y_min", "x_max", "y_max",
                "original_w", "original_h", "thumb_w", "thumb_h", "orientation_fix"
            ]
            header_fmt = wb.add_format({"bold": True, "bg_color": "#D9EAF7", "border": 1})
            normal_fmt = wb.add_format({"border": 1})
            for c, h in enumerate(headers):
                info.write(0, c, h, header_fmt)

            for r, rec in enumerate(sorted(laid_out, key=lambda x: x.get("order_index", 0)), start=1):
                values = [
                    rec.get("order_index"), rec.get("excel_row"), rec.get("excel_col"), rec.get("crop_index"),
                    os.path.basename(rec.get("path", "")), rec.get("cx"), rec.get("cy"), rec.get("w"), rec.get("h"),
                    rec.get("x_min"), rec.get("y_min"), rec.get("x_max"), rec.get("y_max"),
                    rec.get("original_w"), rec.get("original_h"), rec.get("thumb_w"), rec.get("thumb_h"),
                    rec.get("orientation_fix", "none"),
                ]
                for c, v in enumerate(values):
                    info.write(r, c, v, normal_fmt)

            info.write(len(laid_out) + 2, 0, "reading_mode", header_fmt)
            info.write(len(laid_out) + 2, 1, reading_mode)
            info.write(len(laid_out) + 3, 0, "layout_policy", header_fmt)
            info.write(len(laid_out) + 3, 1, layout_policy)
            info.write(len(laid_out) + 4, 0, "cell_size_px", header_fmt)
            info.write(len(laid_out) + 4, 1, cell_size_px)
            info.write(len(laid_out) + 5, 0, "json_path", header_fmt)
            info.write(len(laid_out) + 5, 1, json_path or "")
            info.write(len(laid_out) + 6, 0, "orientation_fixed_count", header_fmt)
            info.write(len(laid_out) + 6, 1, orientation_fixed_count)

            info.set_column(0, 3, 12)
            info.set_column(4, 4, 42)
            info.set_column(5, 17, 12)

        wb.close()
        logger.info(
            "Excel 已保存: %s | 模式: %s/%s | 字符数: %d | 布局: %d 行 x %d 列 | 单元格: %dpx | 方向修正: %d",
            excel_path, reading_mode, layout_policy, len(laid_out),
            row_count, col_count, cell_size_px, orientation_fixed_count,
        )

    except Exception as e:
        logger.exception("[Excel] 生成出错: %s", e)
        try:
            wb.close()
        except Exception:
            pass
