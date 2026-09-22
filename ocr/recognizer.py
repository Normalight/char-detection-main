# -*- coding: utf-8 -*-
"""PP-OCRv6 ONNX 文本行识别与单字定位。

输入 RGB 行图及原图四点框；内部按 BGR、高度 48 预处理。
CTC 解码返回 raw、line_pred、char_list、words，字框回映到原图坐标。
字典读取同级 inference.yml；超字典 token 作为词间空格并保留字框。
单字位置由 CTC 时间列估计，再按所选策略细化。
"""

import os
import logging
from typing import List, Tuple

import cv2
import numpy as np
import onnxruntime as ort
import yaml
from PIL import Image

# OCR 共用坐标回映和点序。
from ocr.geometry import reverse_rotate_crop_image, order_points
from .stroke_ownership import stroke_owned_boxes
from .char_boxes import (
    independent_stroke_boxes,
    adaptive_midpoint_boundaries,
    midpoint_boundaries,
    refine_boundaries_by_polarity,
    refine_boundaries_by_projection,
    valid_intervals_from_boundaries,
)

logger = logging.getLogger(__name__)


# v6 预处理常量。
IMG_H = 48                  # v6 固定高度
MAX_W_DEFAULT = 3200        # 推理时不截断到训练上限 320，按真实比例算（实测必须如此）


def _build_session(model_path: str) -> ort.InferenceSession:
    opts = ort.SessionOptions()
    # 不显式设 intra_op_num_threads。实测在 macOS 10 核 CPU 上，
    # 固定 W=320 的合成输入下 thread=4 比 thread=0(默认) 快 1.6×；
    # 但真实多宽度行混合场景下两者波动重叠（1700-2100ms / 31 行），
    # ORT 的自动调度反而更稳。保留默认行为，需要时用环境变量覆盖：
    #   OMP_NUM_THREADS=4 python ...
    opts.inter_op_num_threads = 1
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(model_path, sess_options=opts,
                                 providers=["CPUExecutionProvider"])


class PPOcrV6RecHandle:
    """PP-OCRv6 文本行识别模型包装。

    参数：
        model_path: inference.onnx 路径（同级应有 inference.yml）
        yml_path:   可选，显式指定 yml 路径；默认用 model_path 同目录的 inference.yml
        char_box_strategy:
            stroke-ownership 按笔画归属生成框，仅保留 1 像素余量；
            independent-strokes 旧独立框及比例留白策略，用于对照；
            ctc-midpoint 只使用 CTC 中心；
            projection-refined 固定按暗字亮底投影微调；
            polarity-aware 亮底投影微调、暗底保留 CTC 中点，并保全首尾。
    """

    def __init__(
        self,
        model_path: str,
        yml_path: str = None,
        char_box_strategy: str = "ctc-midpoint",
    ):
        supported_strategies = {
            "stroke-ownership",
            "independent-strokes",
            "ctc-midpoint",
            "projection-refined",
            "polarity-aware",
        }
        if char_box_strategy not in supported_strategies:
            raise ValueError(
                "unsupported char_box_strategy: "
                f"{char_box_strategy}; expected one of "
                f"{sorted(supported_strategies)}"
            )
        self.char_box_strategy = char_box_strategy
        self.sess = _build_session(model_path)
        self.input_name = self.sess.get_inputs()[0].name
        self.output_name = self.sess.get_outputs()[0].name
        self.vocab_size = self.sess.get_outputs()[0].shape[-1]
        if isinstance(self.vocab_size, str):
            # 动态维度，从 yml 推断
            self.vocab_size = None

        if yml_path is None:
            yml_path = os.path.join(os.path.dirname(model_path), "inference.yml")
        # PaddleOCR ships ``inference.yml`` as UTF-8.  Always specify the
        # encoding because Windows otherwise uses the active ANSI code page
        # (commonly GBK), which cannot decode parts of the official config.
        with open(yml_path, encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        # 字典：CTCLabelDecode 的 character 是 dict（list），
        # 推理时 character = ["<blank>"] + dict，idx 0 = blank，idx 1..N 对应 dict[0..N-1]。
        char_dict = cfg["PostProcess"]["character_dict"]
        self.character = ["<blank>"] + list(char_dict)
        if self.vocab_size is None:
            self.vocab_size = len(self.character)
        # ONNX 输出可能比字典多 1 维（padding），decode 时用 min 防越界。
        self.decode_limit = min(self.vocab_size, len(self.character))

    # ---- 预处理 ----

    def _preprocess(self, crop_im: Image.Image) -> Tuple[np.ndarray, float, int]:
        """PIL RGB crop → (1, 3, 48, W) float32。

        返回 (tensor, scale, resized_w)。
        scale = resized_w / crop_w = 48 / crop_h（保持长宽比）。

        优化（实测 1.45→0.83 ms/行，加速 1.75×）：
          - 不再走 PIL.convert("RGB") + 双重通道反转 + copy，
            直接 np.array(crop_im) 假定 RGB，
            用 [::-1] view 得到 BGR，不复制。
          - 归一化合并成单次乘加：x * (1/127.5) - 1.0
            数学上等价于 (x/255 - 0.5)/0.5 = (x - 127.5)/127.5。
          - transpose 后立刻 ascontiguousarray，让 ONNX 拿到连续内存。
        """
        # PIL → ndarray；通道顺序：crop_im 是 RGB（由 ocr.pipeline.recognize_rows
        # 的 Image.fromarray(crop_bgr[:,:,::-1]) 构造而来），反转得 BGR。
        rgb = np.asarray(crop_im, dtype=np.uint8)
        if rgb.ndim != 3 or rgb.shape[2] != 3:
            # 兜底：非 RGB 的 PIL（灰度/RGBA）走原路径
            rgb = np.array(crop_im.convert("RGB"))
        bgr = rgb[:, :, ::-1]   # view，不复制
        h, w = bgr.shape[:2]
        if h == 0 or w == 0:
            raise ValueError(f"crop 尺寸为 0: {bgr.shape}")

        # 高 resize 到 48，宽按比例（不截断到训练上限 320，实测会变形）
        new_w = max(1, min(int(round(IMG_H * w / float(h))), MAX_W_DEFAULT))
        resized = cv2.resize(bgr, (new_w, IMG_H))

        # 归一化 (x/255 - 0.5)/0.5 = (x - 127.5)/127.5
        # 合并为单次乘加：astype + *= 1/127.5 + -= 1.0
        arr = resized.astype(np.float32)
        arr *= (1.0 / 127.5)
        arr -= 1.0

        # HWC → CHW → NCHW，连续内存
        tensor = np.ascontiguousarray(arr.transpose(2, 0, 1)[None])
        scale = IMG_H / float(h)   # = new_w / w
        return tensor, scale, new_w

    # ---- 推理 ----

    def predict_rbg(self, crop_im: Image.Image, bbox, *, defer_stroke_refinement=False) -> dict:
        """单行识别与单字位置反推。"""
        tensor, scale, resized_w = self._preprocess(crop_im)
        preds = self.sess.run([self.output_name], {self.input_name: tensor})[0]
        return self._postprocess(preds, scale, resized_w, crop_im, bbox, defer_stroke_refinement=defer_stroke_refinement)

    def predict_rbg_batch(self, items: List[Tuple[Image.Image, np.ndarray]], *, defer_stroke_refinement=False) -> List[dict]:
        """批量识别。

        v6 rec 的 ONNX 名义上是 batch-safe（动态 batch 维度），但实测：
          - 真批量（pad 到同宽一次推理）会让模型在 pad 区产生额外时间步，
            导致行末识别结果偶尔变化（有时更准、有时多/少字）——正确性不可接受。
          - thread=4 + 真 batch 在合成固定宽度输入下能加速 1.5×，
            但真实多宽度混合场景下与逐行相比波动重叠，收益不稳定。

        因此当前实现按输入顺序逐行推理。
        未来若要重新启用真批量，需先解决 pad 区对 CTC 解码的干扰
        （例如用复制边缘像素代替 0 填充，或按真实宽度切片只取有效 T）。
        """
        return [self.predict_rbg(crop, bbox, defer_stroke_refinement=defer_stroke_refinement) for crop, bbox in items]

    # ---- 后处理：CTC decode + 字级位置反推 ----

    def _decode_ctc(self, preds: np.ndarray) -> Tuple[str, List[dict]]:
        """CTC greedy decode。

        preds: 单样本 (T, vocab) 或带 batch (1, T, vocab) → 返回 (line_pred, char_list)
        char_list[i] = {'char': str, 'idx': int}
        idx 是该字符在 T 序列里第一次出现的时间步（用于位置反推）。
        """
        # 兼容 (T, vocab) 和 (1, T, vocab) 两种 shape
        if preds.ndim == 3 and preds.shape[0] == 1:
            preds = preds[0]
        idxs = preds.argmax(axis=1)   # (T,)
        chars = []
        char_list = []
        last = 0   # CTC blank reset
        for t, idx in enumerate(idxs):
            idx = int(idx)
            if idx == 0:        # blank
                last = 0
                continue
            if idx == last:     # 重复折叠
                continue
            is_filler = not (0 < idx < self.decode_limit)
            if is_filler:
                # v6 字典不含空格：超字典 token 是 CTC 对词间空隙的输出。
                # 还原为空格并保留字框，相邻多个空隙折叠为一个。
                if chars and chars[-1] == " ":
                    last = idx
                    continue
                ch = " "
            else:
                ch = self.character[idx]
            chars.append(ch)
            char_list.append(
                {
                    "char": ch,
                    "idx": t,
                    "score": float(preds[t, idx]),
                    "filler": is_filler,
                }
            )
            last = idx
        line_pred = "".join(chars)
        return line_pred, char_list

    def _postprocess(self, preds: np.ndarray, scale: float, resized_w: int,
                     crop_im: Image.Image, bbox, *, defer_stroke_refinement=False) -> dict:
        """CTC 解码与原图单字坐标还原。"""
        line_pred, char_list = self._decode_ctc(preds)
        raw = line_pred   # v6 没有 raw 概念，用 line_pred 占位以保持 schema

        crop_w, crop_h = crop_im.size
        # Official PaddleOCR return_word_box maps CTC columns using the actual
        # decoded column count and places a token at the centre of its column.
        # This is more exact than assuming a fixed backbone downsample ratio
        # after input-width rounding.
        time_steps = preds.shape[-2]
        cell_width = crop_w / float(time_steps)
        centers = [
            (char_item["idx"] + 0.5) * cell_width
            for char_item in char_list
        ]
        if self.char_box_strategy == "stroke-ownership" and defer_stroke_refinement:
            # The full-image pipeline refines exactly once, after page geometry
            # is known. These provisional boxes only carry CTC anchor geometry.
            local_boxes = [(i,(max(0,c-crop_h/2),0,min(crop_w,c+crop_h/2),crop_h))
                           for i,c in enumerate(centers)]
        elif self.char_box_strategy == "stroke-ownership":
            local_boxes = list(enumerate(stroke_owned_boxes(
                np.asarray(crop_im.convert("RGB")), centers,
                [item["char"] for item in char_list],
            )))
        elif self.char_box_strategy == "independent-strokes":
            local_boxes = list(enumerate(independent_stroke_boxes(
                np.asarray(crop_im.convert("RGB")), centers,
                [item["char"] for item in char_list], allow_outer_padding=True,
            )))
        elif self.char_box_strategy in {
            "projection-refined",
            "polarity-aware",
        }:
            crop_rgb = np.asarray(crop_im.convert("RGB"))
            if self.char_box_strategy == "polarity-aware":
                # Keep the full detector-row extent for the first and last
                # characters. Adaptive outer trimming can clip wide serifs or
                # calligraphic strokes even when all internal cuts are sound.
                boundaries = midpoint_boundaries(centers, crop_w)
                boundaries = refine_boundaries_by_polarity(
                    crop_rgb,
                    centers,
                    boundaries,
                )
            else:
                boundaries = adaptive_midpoint_boundaries(centers, crop_w)
                boundaries = refine_boundaries_by_projection(
                    crop_rgb,
                    centers,
                    boundaries,
                )
        else:
            boundaries = midpoint_boundaries(centers, crop_w)
        if self.char_box_strategy not in {"independent-strokes", "stroke-ownership"}:
            local_boxes = [
                (index, (left, 0, right, crop_h))
                for index, (left, right) in valid_intervals_from_boundaries(boundaries)
            ]
        retained_chars = []
        for source_index, (left, top, right, bottom) in local_boxes:
            char_item = char_list[source_index]
            center = centers[source_index]
            line = [[center, 0], [center, crop_h]]
            line = reverse_rotate_crop_image(None, None, bbox, line, (0, 0))
            char_item["line"] = line

            rect = [[left, top], [left, bottom], [right, bottom], [right, top]]
            rect = reverse_rotate_crop_image(None, None, bbox, rect, (0, 0))
            rect = order_points(rect)
            char_item["bbox"] = rect
            retained_chars.append(char_item)

        char_list = retained_chars
        line_pred = "".join(item["char"] for item in char_list)

        # words 按中文字符、英文和符号分组。
        import re
        words = []

        def group_char_to_word(char_items):
            bboxs = [x["bbox"] for x in char_items]
            points = np.array(bboxs, dtype=np.intp).reshape((-1, 1, 2))
            if len(points) == 0:
                return None
            rect = cv2.minAreaRect(points)
            bbox = cv2.boxPoints(rect)
            bbox = order_points(bbox)
            bbox = np.array(bbox, dtype=np.intp)
            return {"value": "".join(x["char"] for x in char_items), "bbox": bbox}

        matches = list(re.finditer(r"[\u4e00-\u9fa5 ,.?:/\\()\[\]]", line_pred))
        prev_idx = 0
        for match in matches:
            left, right = match.span()
            if prev_idx != left:
                w = group_char_to_word(char_list[prev_idx:left])
                if w:
                    words.append(w)
            w = group_char_to_word(char_list[left:right])
            if w:
                words.append(w)
            prev_idx = right
        if prev_idx != len(char_list):
            w = group_char_to_word(char_list[prev_idx:])
            if w:
                words.append(w)

        return {
            "char_box_strategy": self.char_box_strategy,
            "raw": raw,
            "line_pred": line_pred,
            "score": (
                float(np.mean([item["score"] for item in char_list]))
                if char_list
                else 0.0
            ),
            "char_list": char_list,
            "words": words,
        }
