"""PDF 渲染工具：PyMuPDF → pdf2image → Pillow 三级 fallback。

任一级失败会记录 warning 并落到下一级；三级全失败则抛 RuntimeError。
"""

import io
import logging
import re
from typing import Optional

from PIL import Image

logger = logging.getLogger(__name__)


def _extract_full_page_scan(doc, page):
    """仅提取可证明等同整页外观的单图；复杂 PDF 保守回退渲染。"""
    if page.first_annot is not None or page.first_widget is not None:
        return None
    # Form XObjects、裁剪路径、透明度等可能改变图片实际可见内容。
    if page.get_xobjects():
        return None
    if re.search(rb"(?:^|[\s\[\]()<>/%])(?:W\*?|gs)(?=$|[\s\[\]()<>/%])", page.read_contents()):
        return None
    if [entry[0] for entry in page.get_bboxlog()] != ["fill-image"]:
        return None
    images = page.get_image_info(xrefs=True)
    if len(images) != 1 or not images[0].get("xref"):
        return None
    info = images[0]
    a, b, c, d, _, _ = info["transform"]
    # 图片自身旋转、镜像、斜切时交给渲染器；页面旋转可无损转置。
    if a <= 0 or d <= 0 or abs(b) > 1e-6 or abs(c) > 1e-6:
        return None
    bounds = page.rect * page.derotation_matrix
    if any(abs(x - y) > 1e-3 for x, y in zip(info["bbox"], bounds)):
        return None
    xref = info["xref"]
    for key in ("SMask", "Mask", "Decode"):
        if doc.xref_get_key(xref, key)[0] != "null":
            return None
    if doc.xref_get_key(xref, "ImageMask")[1] == "true":
        return None
    data = doc.extract_image(xref)
    if not data or data.get("smask", 0):
        return None
    with Image.open(io.BytesIO(data["image"])) as source:
        # 自定义颜色空间/透明度交给 PDF 渲染，避免改变页面外观。
        color_space = doc.xref_get_key(xref, "ColorSpace")[1]
        if source.mode not in ("1", "L", "RGB") or color_space not in (
            "/DeviceGray", "/DeviceRGB",
        ):
            return None
        image = source.convert("RGB")
    rotation = {90: Image.Transpose.ROTATE_270,
                180: Image.Transpose.ROTATE_180,
                270: Image.Transpose.ROTATE_90}.get(page.rotation)
    return image.transpose(rotation) if rotation is not None else image


def render_pdf_page_to_pil(pdf_path, page_index, dpi=200, poppler_path: Optional[str] = None):
    """整页扫描图优先提取原始像素，其余页面按 DPI 渲染为 RGB。"""
    # 1) PyMuPDF
    try:
        import fitz
        with fitz.open(pdf_path) as doc:
            if page_index < 0 or page_index >= doc.page_count:
                raise IndexError(f"页索引越界: {page_index} / {doc.page_count}")
            page = doc.load_page(page_index)
            try:
                original = _extract_full_page_scan(doc, page)
                if original is not None:
                    return original
            except Exception as error:
                logger.debug("扫描页原图提取不可用，按 DPI 渲染：%s", error)
        # 图像检查可能建立 MuPDF 解码缓存；重新打开以保持标准渲染结果。
        with fitz.open(pdf_path) as doc:
            page = doc.load_page(page_index)
            zoom = dpi / 72.0
            pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), alpha=False)
            return Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    except ImportError:
        logger.warning("PyMuPDF (fitz) 未安装，尝试 pdf2image")
    except Exception as e:
        logger.warning("PyMuPDF 渲染 PDF 失败：%s，尝试 pdf2image", e)

    # 2) pdf2image
    try:
        from pdf2image import convert_from_path
        images = convert_from_path(pdf_path, dpi=dpi, first_page=page_index+1, last_page=page_index+1, poppler_path=poppler_path)
        if images:
            return images[0].convert('RGB')
    except ImportError:
        logger.warning("pdf2image 未安装，尝试 Pillow 直接解码")
    except Exception as e:
        logger.warning("pdf2image 渲染 PDF 失败（请确认系统已安装 poppler）：%s，尝试 Pillow", e)

    # 3) Pillow（系统需支持PDF解码）
    try:
        with Image.open(pdf_path) as im:
            im.seek(page_index)
            return im.convert('RGB').copy()
    except Exception as e:
        logger.error("Pillow 也无法解码 PDF：%s", e)
        raise RuntimeError(
            f"无法渲染 PDF 页面（PyMuPDF / pdf2image / Pillow 三级 fallback 全部失败）：{e}"
        )


def get_pdf_page_count(pdf_path, poppler_path: Optional[str] = None) -> int:
    """获取 PDF 总页数。"""
    # 1) PyMuPDF
    try:
        import fitz
        with fitz.open(pdf_path) as doc:
            return doc.page_count
    except ImportError:
        logger.warning("PyMuPDF (fitz) 未安装，尝试 pdf2image")
    except Exception as e:
        logger.warning("PyMuPDF 读取页数失败：%s，尝试 pdf2image", e)

    # 2) pdf2image
    try:
        from pdf2image import pdfinfo_from_path
        info = pdfinfo_from_path(pdf_path, poppler_path=poppler_path)
        pages = info.get("Pages") or info.get("pages") or info.get("PAGE_COUNT")
        if isinstance(pages, str):
            pages = int(pages)
        return int(pages)
    except ImportError:
        logger.warning("pdf2image 未安装，尝试 Pillow 直接解码")
    except Exception as e:
        logger.warning("pdf2image 读取页数失败（请确认系统已安装 poppler）：%s，尝试 Pillow", e)

    # 3) Pillow
    cnt = 0
    try:
        with Image.open(pdf_path) as im:
            if hasattr(im, "n_frames"):
                return int(im.n_frames)
            while True:
                im.seek(cnt)
                cnt += 1
    except EOFError:
        return cnt
    except Exception as e:
        logger.error("Pillow 也无法读取 PDF 页数：%s", e)
        raise RuntimeError(
            "无法获取 PDF 页数（PyMuPDF / pdf2image / Pillow 三级 fallback 全部失败），"
            "请安装 PyMuPDF 或 pdf2image+poppler。"
        )
