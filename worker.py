# -*- coding: utf-8 -*-
"""worker.py — 长任务后台线程。

把原图加载、OCR 预标注、当前页和批量保存等耗时调用
从主线程移到 QThread，避免 Qt 事件循环被阻塞导致界面"未响应"。

约定（重要）：
- Worker 内**绝不**碰 QWidget / QImage / QMessageBox 等 GUI 对象；
- 所有结果通过 signal 投递回主线程（Qt 自动跨线程 queue）；
- 主线程在 slot 里更新 GUI 与 core 状态；
- 取消用 `cancel()` 方法置标志位，Worker 在循环点检查。
"""

import copy
import logging
import traceback
from typing import Any, Callable, List, Optional

import cv2
import numpy as np
from PIL import Image
from PySide6.QtCore import QThread, Signal

from cutbatch import OcrHandle
from excel_utils import build_excel_for_single_item, make_crop_thumbnail
from pdf_utils import get_pdf_page_count, render_pdf_page_to_pil
from storage_utils import (
    extract_boxes_from_ocr_results,
    save_item_boxes_to_disk,
    save_current_slices_and_json,
    load_item_annotation_record_if_exists,
)

logger = logging.getLogger(__name__)


class _BaseWorker(QThread):
    """所有 worker 的公共基类：提供 cancel 与错误投递。"""

    error = Signal(str)         # 出错文本（含 traceback）
    finished_ok = Signal()      # 正常结束（不含错误）

    def __init__(self, parent=None):
        super().__init__(parent)
        self._cancelled = False

    def cancel(self):
        """请求取消。Worker 在下一个检查点退出。"""
        self._cancelled = True

    @property
    def is_cancelled(self) -> bool:
        return self._cancelled

    def _emit_error(self, exc, where: str):
        tb = traceback.format_exc()
        logger.error("[worker %s] 异常：%s\n%s", where, exc, tb)
        self.error.emit(f"{where}：{exc}")


class ModelLoadWorker(_BaseWorker):
    """加载 ONNX 模型（dbnet + crnn）。首次启动约 2-5 秒。

    成功后通过 ocr_ready 投递 OcrHandle。模型加载和后续推理都由
    后台 worker 执行，主线程只编排任务并合并结果。
    """
    ocr_ready = Signal(object)   # OcrHandle 实例

    def run(self):
        try:
            handle = OcrHandle()
            if self.is_cancelled:
                return
            self.ocr_ready.emit(handle)
            self.finished_ok.emit()
        except Exception as e:
            self._emit_error(e, "模型加载")


def _load_item_pil(item, pdf_dpi, poppler_path, image_scale) -> Image.Image:
    """worker 线程里加载/渲染一页（复制自 annotator_core，避免循环依赖）。"""
    if item.get("type") == "image":
        pil_rgb = Image.open(item["source_path"]).convert("RGB")
    else:
        pil_rgb = render_pdf_page_to_pil(
            item["source_path"], item["page_index"],
            dpi=pdf_dpi, poppler_path=poppler_path,
        )
    # image_scale now affects only the OCR proxy, never saved pixels.
    return pil_rgb.convert("RGB")


def _predict_ocr(handler, image, size, cancel_check):
    # Keep lightweight/custom backends compatible with the two-argument API.
    if isinstance(handler, OcrHandle):
        return handler.text_predict(image, size, cancel_check=cancel_check)
    return handler.text_predict(image, size)


class PageLoadWorker(_BaseWorker):
    """Decode/render an original page and read its annotation off the UI thread."""
    loaded = Signal(object)

    def __init__(self, item, output_dir, pdf_dpi, poppler_path, context, parent=None):
        super().__init__(parent)
        self._item = copy.deepcopy(item)
        self._output_dir = output_dir
        self._pdf_dpi = pdf_dpi
        self._poppler = poppler_path
        self._context = dict(context)

    def run(self):
        try:
            if self.is_cancelled:
                return
            pil = _load_item_pil(self._item, self._pdf_dpi, self._poppler, 1.0)
            if self.is_cancelled:
                return
            record = None if self._item.get('visited') else load_item_annotation_record_if_exists(
                self._output_dir, self._item, target_size_wh=pil.size)
            bgr = cv2.cvtColor(np.asarray(pil), cv2.COLOR_RGB2BGR)
            if not self.is_cancelled:
                self.loaded.emit({'pil': pil, 'bgr': bgr, 'record': record, 'context': self._context})
                self.finished_ok.emit()
        except Exception as error:
            if not self.is_cancelled:
                self._emit_error(error, '页面加载')


class SaveCurrentWorker(_BaseWorker):
    """Save an immutable page snapshot; the original image buffer is read-only."""
    progress = Signal(int, int, str)
    page_saved = Signal(object)
    stopped = Signal(str)

    def __init__(self, output_dir, item, base_image_bgr, parent=None):
        super().__init__(parent)
        self._output_dir = output_dir
        self._item = copy.deepcopy(item)
        self._image = base_image_bgr
        self._single_page = True

    def run(self):
        if self.is_cancelled:
            self.stopped.emit('已取消保存当前页')
            return
        try:
            thumbnails = {}
            def remember_thumbnail(index, crop, box):
                thumbnails[index] = make_crop_thumbnail(crop, box)
            count = save_current_slices_and_json(
                self._output_dir, self._item, self._image, self._item.get('boxes', []),
                progress_callback=lambda done,total: self.progress.emit(done,total,'正在写入原图切片'),
                crop_callback=remember_thumbnail)
            if count < 0:
                raise RuntimeError('原图为空，无法保存')
            self.progress.emit(0,1,'正在生成 Excel')
            build_excel_for_single_item(self._output_dir, self._item, cached_thumbnails=thumbnails)
            self._item.update(ever_saved=True, dirty=False, nav_scale=1.0)
            self.page_saved.emit(copy.deepcopy(self._item))
            if self.is_cancelled:
                self.stopped.emit('当前页已保存，后续翻页已取消')
            else:
                self.progress.emit(1,1,'当前页保存完成')
                self.finished_ok.emit()
        except Exception as error:
            self._emit_error(error, '保存当前页')
        finally:
            self._image = None


class OcrPrefetchWorker(_BaseWorker):
    """对当前页跑 OCR 预标注，产出 boxes。

    输入：pil_rgb（原始像素）、short_size、level
    输出：单个 dict；boxes 每项为 (4,2) float32
    """
    progress = Signal(str)
    ocr_done = Signal(object)

    def __init__(self, pil_rgb: Image.Image, short_size: int, level: str,
                 ocr_handler: OcrHandle, context=None, parent=None):
        super().__init__(parent)
        self._pil = pil_rgb
        self._short_size = short_size
        self._level = level
        self._ocr = ocr_handler
        self._context = copy.deepcopy(context or {})

    def run(self):
        try:
            self.progress.emit("正在执行 OCR 预标注...")
            results = _predict_ocr(self._ocr, self._pil, self._short_size, lambda: self.is_cancelled)
            if self.is_cancelled:
                return
            boxes = extract_boxes_from_ocr_results(results, self._level)
            if self.is_cancelled:
                return
            self.ocr_done.emit(
                {
                    "boxes": boxes,
                    "ocr_geometry": results[0].get("ocr_geometry") if results else None,
                    "context": copy.deepcopy(self._context),
                }
            )
            self.finished_ok.emit()
        except Exception as e:
            self._emit_error(e, "OCR 预标注")


class OcrBatchPrefetchWorker(_BaseWorker):
    """空闲时在后台顺序预标注尚未访问的页面（无感预取）。

    与 OcrPrefetchWorker 的区别：
    - 自己加载/渲染图片（不依赖主线程先备好 PIL）；
    - 一次处理一批页面，逐页通过 page_ocr_done 投递结果；
    - 单页失败只记日志并跳过（page_ocr_failed 告知 core 标记该页），
      不打断批次，也不弹错误框——后台任务保持无感；
    - 页面归属、请求失效与编辑保护全部由 core._on_ocr_done 按
      context（item_identity / request_id / edit_revision）判定。
    """

    page_ocr_done = Signal(object)     # {"boxes": [...], "context": {...}}
    page_ocr_failed = Signal(object)   # {"context": {...}}

    def __init__(self, jobs: List[dict], ocr_handler: OcrHandle,
                 dbnet_short_side: int, level: str,
                 pdf_dpi: int, poppler_path: Optional[str],
                 image_scale: float, parent=None):
        super().__init__(parent)
        self._jobs = list(jobs)
        self._ocr = ocr_handler
        self._short_size = dbnet_short_side
        self._level = level
        self._pdf_dpi = pdf_dpi
        self._poppler = poppler_path
        self._image_scale = image_scale

    def run(self):
        try:
            for job in self._jobs:
                if self.is_cancelled:
                    return
                name = job.get("display_name", "")
                try:
                    if job.get('type') == 'image' and isinstance(self._ocr, OcrHandle):
                        results = self._ocr.text_predict_file(
                            job['source_path'], self._short_size, cancel_check=lambda: self.is_cancelled)
                    else:
                        pil_rgb = _load_item_pil(
                            job, self._pdf_dpi, self._poppler, self._image_scale)
                        if self.is_cancelled:
                            return
                        results = _predict_ocr(
                            self._ocr, pil_rgb, self._short_size, lambda: self.is_cancelled)
                    if self.is_cancelled:
                        return
                    boxes = extract_boxes_from_ocr_results(
                        results, self._level
                    )
                    if self.is_cancelled:
                        return
                    self.page_ocr_done.emit(
                        {
                            "boxes": boxes,
                            "ocr_geometry": results[0].get("ocr_geometry") if results else None,
                            "context": copy.deepcopy(
                                job.get("context") or {}
                            ),
                        }
                    )
                except Exception as e:
                    logger.warning(
                        "后台预标注失败，跳过 %s：%s", name, e
                    )
                    self.page_ocr_failed.emit(
                        {"context": copy.deepcopy(job.get("context") or {})}
                    )
                    continue
            self.finished_ok.emit()
        except Exception as e:
            self._emit_error(e, "后台预标注")


class SaveAllWorker(_BaseWorker):
    """批量保存所有 item：切片 + JSON + Excel。

    关键：这个 worker 自己加载/渲染图片、自己调 OCR 兜底、自己写 Excel，
    完全不依赖主线程的 core 状态。主线程只需把 items 的浅拷贝传进来。
    """
    progress = Signal(int, int, str)   # (done, total, display_name)
    page_saved = Signal(object)        # item dict（主线程用它刷新 ever_saved/dirty）
    stopped = Signal(str)              # 取消/部分失败（任务结束但不是成功）

    def __init__(self, output_dir: str, items: List[dict],
                 ocr_handler: OcrHandle, dbnet_short_side: int, level: str,
                 pdf_dpi: int, poppler_path: Optional[str],
                 image_scale: float, parent=None):
        super().__init__(parent)
        self._output_dir = output_dir
        self._items = items
        self._ocr = ocr_handler
        self._short_size = dbnet_short_side
        self._level = level
        self._pdf_dpi = pdf_dpi
        self._poppler = poppler_path
        self._image_scale = image_scale

    def run(self):
        total = len(self._items)
        saved = 0
        for i, it in enumerate(self._items):
            if self.is_cancelled:
                break
            name = it.get("display_name", "")
            self.progress.emit(i, total, name)

            try:
                pil_rgb = _load_item_pil(
                    it, self._pdf_dpi, self._poppler, self._image_scale
                )
            except Exception as e:
                logger.warning("跳过：%s - %s", name, e)
                continue

            try:
                saved_crop_count = save_item_boxes_to_disk(
                    self._output_dir, it, pil_rgb, self._ocr,
                    self._short_size, self._level,
                    pdf_dpi=self._pdf_dpi, poppler_path=self._poppler,
                )
                expected_crop_count = len(it.get("boxes", []))
                if saved_crop_count < expected_crop_count:
                    raise RuntimeError(
                        "切片写入不完整："
                        f"{saved_crop_count} / {expected_crop_count}"
                    )
                it["ever_saved"] = True
                it["dirty"] = False
                it["nav_scale"] = 1.0
                # Excel 在 worker 线程里跑也安全（纯 IO + xlsxwriter）
                try:
                    build_excel_for_single_item(self._output_dir, it)
                except Exception as ee:
                    logger.warning("[Excel] %s：%s", name, ee)
                self.page_saved.emit(copy.deepcopy(it))
                saved += 1
            except Exception as e:
                logger.warning("保存失败：%s，错误：%s", name, e)

        if self.is_cancelled:
            message = f"批量保存已取消：已保存 {saved} / {total}"
            self.progress.emit(saved, total, message)
            self.stopped.emit(message)
            return
        if saved != total:
            message = f"批量保存未完成：成功 {saved} / {total}"
            logger.warning(message)
            self.progress.emit(saved, total, message)
            self.stopped.emit(message)
            return
        self.progress.emit(saved, total, "完成")
        self.finished_ok.emit()
