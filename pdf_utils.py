"""PDF 渲染工具：PyMuPDF → pdf2image → Pillow 三级 fallback。

任一级失败会记录 warning 并落到下一级；三级全失败则抛 RuntimeError。
"""

import logging
from typing import Optional

from PIL import Image

logger = logging.getLogger(__name__)


def render_pdf_page_to_pil(pdf_path, page_index, dpi=200, poppler_path: Optional[str] = None):
    """把指定页渲染为 PIL RGB 图像。"""
    # 1) PyMuPDF
    try:
        import fitz
        doc = fitz.open(pdf_path)
        if page_index < 0 or page_index >= doc.page_count:
            doc.close()
            raise IndexError(f"页索引越界: {page_index} / {doc.page_count}")
        page = doc.load_page(page_index)
        zoom = dpi / 72.0
        mat = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=mat, alpha=False)
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        doc.close()
        return img.convert('RGB')
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
