# -*- coding: utf-8 -*-
"""生僻字开放集复核的候选协议和纯逻辑。

本模块不依赖 Qt，也不要求模型资产存在。模型适配器只需返回包含 ``char`` 和
``score`` 的候选序列，归并流程即可用同一套规则生成可审计的人工复核建议。
"""

from dataclasses import dataclass
import hashlib
import json
import math
import os
from typing import Callable, Iterable, Mapping, Optional, Tuple
import unicodedata

import numpy as np
from PIL import Image

from glyph_preprocess import pad_glyph_image


REVIEW_COLOR_HIGH = "high"
REVIEW_COLOR_MEDIUM = "medium"
REVIEW_COLOR_LOW = "low"

_REVIEWABLE_STATUSES = frozenset(
    {
        "low_confidence",
        "missing_confidence",
        "review",
        "unrecognized",
    }
)
_INVALID_PATH_CHARS = frozenset('/\\:*?"<>|')
_PLACEHOLDER_CHARS = frozenset({"?", "."})


@dataclass(frozen=True)
class GlyphCandidate:
    """一个经过规范化、按相似度降序排列的单字候选。"""

    char: str
    score: float
    rank: int

    def to_dict(self) -> dict:
        return {
            "char": self.char,
            "score": self.score,
            "rank": self.rank,
        }


@dataclass(frozen=True)
class GlyphSuggestion:
    """待复核切片的展示归组建议，不代表自动确认。"""

    text: str
    score: Optional[float]
    margin: Optional[float]
    source: str
    review_required: bool = True


def normalize_candidate_char(value: object) -> str:
    """返回可安全展示/建目录的单字符，否则返回空串。"""

    raw = "" if value is None else str(value)
    normalized = unicodedata.normalize("NFC", "".join(raw.split()))
    if len(normalized) != 1:
        return ""
    char = normalized[0]
    if (
        char in _PLACEHOLDER_CHARS
        or char in _INVALID_PATH_CHARS
        or unicodedata.category(char).startswith("C")
    ):
        return ""
    return char


def is_han_character(value: object) -> bool:
    """判断单字符是否位于 Unicode 汉字或兼容汉字区。"""

    char = normalize_candidate_char(value)
    if not char:
        return False
    codepoint = ord(char)
    return (
        0x3400 <= codepoint <= 0x4DBF
        or 0x4E00 <= codepoint <= 0x9FFF
        or 0xF900 <= codepoint <= 0xFAFF
        or 0x20000 <= codepoint <= 0x2EE5F
        or 0x2F800 <= codepoint <= 0x2FA1F
        or 0x30000 <= codepoint <= 0x323AF
    )


def normalize_candidates(
    candidates: Optional[Iterable[Mapping]],
    limit: int = 5,
) -> Tuple[GlyphCandidate, ...]:
    """过滤、去重并按分数返回稳定 Top-K。

    候选分数是模型内排序用的归一化相似度，不解释为统计概率。输入超出
    ``[0, 1]`` 时截断，非数值和非有限值直接忽略。
    """

    if limit <= 0:
        return ()
    best_by_char = {}
    for item in candidates or ():
        if not isinstance(item, Mapping):
            continue
        char = normalize_candidate_char(item.get("char"))
        if not char:
            continue
        try:
            score = float(item.get("score"))
        except (TypeError, ValueError):
            continue
        if not math.isfinite(score):
            continue
        score = min(1.0, max(0.0, score))
        previous = best_by_char.get(char)
        if previous is None or score > previous:
            best_by_char[char] = score

    ranked = sorted(
        best_by_char.items(),
        key=lambda pair: (-pair[1], pair[0]),
    )[:limit]
    return tuple(
        GlyphCandidate(char=char, score=score, rank=rank)
        for rank, (char, score) in enumerate(ranked, start=1)
    )


def choose_glyph_suggestion(
    *,
    status: str,
    primary_text: object,
    primary_score: Optional[float],
    review_candidates: Optional[Iterable[Mapping]],
) -> Optional[GlyphSuggestion]:
    """为不确定记录选择展示位置，始终保留人工复核标记。"""

    if status not in _REVIEWABLE_STATUSES:
        return None

    normalized = tuple(
        candidate
        for candidate in normalize_candidates(review_candidates)
        if is_han_character(candidate.char)
    )
    if normalized:
        first = normalized[0]
        margin = None
        if len(normalized) > 1:
            margin = max(0.0, first.score - normalized[1].score)
        return GlyphSuggestion(
            text=first.char,
            score=first.score,
            margin=margin,
            source="open_set",
        )

    primary_char = normalize_candidate_char(primary_text)
    if not is_han_character(primary_char):
        return None
    score = None
    try:
        if primary_score is not None:
            score_value = float(primary_score)
            if math.isfinite(score_value):
                score = min(1.0, max(0.0, score_value))
    except (TypeError, ValueError):
        score = None
    return GlyphSuggestion(
        text=primary_char,
        score=score,
        margin=None,
        source="primary",
    )


def review_color_level(
    *,
    score: Optional[float],
    margin: Optional[float],
    source: str,
) -> str:
    """按开放集相似度与候选间距返回 Excel 颜色等级。"""

    if source != "open_set" or score is None:
        return REVIEW_COLOR_LOW
    if score >= 0.75 and margin is not None and margin >= 0.08:
        return REVIEW_COLOR_HIGH
    if score >= 0.55:
        return REVIEW_COLOR_MEDIUM
    return REVIEW_COLOR_LOW


class OnnxGlyphReviewer:
    """ONNX 单字图像编码器 + 字符原型矩阵的开放集复核器。

    构造阶段只校验轻量资产，首次 ``review`` 才创建 ONNX session。字符原型使用
    ``numpy.load(..., mmap_mode="r")``，并分块计算 cosine Top-K。
    """

    def __init__(
        self,
        asset_dir: str,
        *,
        session_factory: Optional[Callable[[str], object]] = None,
        prototype_block_size: int = 4096,
    ):
        self.asset_dir = os.path.abspath(asset_dir)
        if prototype_block_size <= 0:
            raise ValueError("prototype_block_size 必须大于 0")
        self.prototype_block_size = int(prototype_block_size)
        self._session_factory = session_factory or self._default_session_factory
        self._session = None
        self._session_error = ""
        self._input_name = ""
        self._output_name = ""

        manifest_path = os.path.join(self.asset_dir, "manifest.json")
        if not os.path.isfile(manifest_path):
            raise FileNotFoundError(f"缺少模型清单：{manifest_path}")
        with open(manifest_path, encoding="utf-8") as manifest_file:
            self._manifest = json.load(manifest_file)
        self._validate_manifest()

        self.model_path = self._asset_path(self._manifest["model"])
        character_path = self._asset_path(self._manifest["characters"])
        prototype_path = self._asset_path(self._manifest["prototypes"])
        for path in (self.model_path, character_path, prototype_path):
            if not os.path.isfile(path):
                raise FileNotFoundError(f"缺少生僻字模型资产：{path}")
        self._validate_asset_digest(
            self.model_path,
            self._manifest["model_sha256"],
            "model",
        )
        self._validate_asset_digest(
            character_path,
            self._manifest["characters_sha256"],
            "characters",
        )
        self._validate_asset_digest(
            prototype_path,
            self._manifest["prototypes_sha256"],
            "prototypes",
        )

        with open(character_path, encoding="utf-8") as character_file:
            raw_characters = [
                line.rstrip("\r\n")
                for line in character_file
                if line.rstrip("\r\n")
            ]
        self.characters = tuple(
            normalize_candidate_char(character)
            for character in raw_characters
        )
        if (
            not self.characters
            or any(not character for character in self.characters)
            or len(set(self.characters)) != len(self.characters)
        ):
            raise ValueError("characters.txt 必须包含唯一、合法的单字符")

        self.prototypes = np.load(prototype_path, mmap_mode="r")
        if self.prototypes.ndim != 2:
            raise ValueError("prototypes.npy 必须是二维矩阵")
        if self.prototypes.shape[0] != len(self.characters):
            raise ValueError(
                "原型字符数与字符表字符数不一致："
                f"{self.prototypes.shape[0]} != {len(self.characters)}"
            )
        if self.prototypes.shape[1] <= 0:
            raise ValueError("原型 embedding 维度必须大于 0")
        self._prototype_norms = self._validate_prototypes()

        input_config = self._manifest["input"]
        self.input_height = self._positive_int(
            input_config.get("height"),
            "input.height",
        )
        self.input_width = self._positive_int(
            input_config.get("width"),
            "input.width",
        )
        self.input_color = str(input_config.get("color", "RGB")).upper()
        self.input_layout = str(input_config.get("layout", "NCHW")).upper()
        if self.input_color not in {"RGB", "BGR"}:
            raise ValueError("input.color 只支持 RGB 或 BGR")
        if self.input_layout != "NCHW":
            raise ValueError("input.layout 当前只支持 NCHW")
        self.input_mean = self._channel_values(
            input_config.get("mean"),
            "input.mean",
        )
        self.input_std = self._channel_values(
            input_config.get("std"),
            "input.std",
        )
        if np.any(self.input_std <= 0):
            raise ValueError("input.std 必须全部大于 0")

        self.metadata = {
            "model_id": self._manifest["model_id"],
            "prototype_version": self._manifest["prototype_version"],
            "source": self._manifest["source"],
            "license": self._manifest["license"],
            "character_count": len(self.characters),
            "embedding_size": int(self.prototypes.shape[1]),
        }

    @staticmethod
    def _default_session_factory(model_path: str):
        import onnxruntime as ort

        options = ort.SessionOptions()
        options.inter_op_num_threads = 1
        options.graph_optimization_level = (
            ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        )
        return ort.InferenceSession(
            model_path,
            sess_options=options,
            providers=["CPUExecutionProvider"],
        )

    def _validate_manifest(self) -> None:
        required = (
            "model_id",
            "prototype_version",
            "source",
            "license",
            "model",
            "model_sha256",
            "characters",
            "characters_sha256",
            "prototypes",
            "prototypes_sha256",
            "input",
        )
        for field in required:
            value = self._manifest.get(field)
            if value is None or value == "":
                raise ValueError(f"manifest 缺少必填字段：{field}")
        if self._manifest.get("version") != 1:
            raise ValueError("manifest.version 当前只支持 1")
        if not isinstance(self._manifest["input"], Mapping):
            raise ValueError("manifest.input 必须是对象")

    def _asset_path(self, relative_path: object) -> str:
        value = str(relative_path)
        if os.path.isabs(value):
            raise ValueError("模型资产路径必须相对于 asset_dir")
        resolved = os.path.abspath(os.path.join(self.asset_dir, value))
        if os.path.commonpath([self.asset_dir, resolved]) != self.asset_dir:
            raise ValueError(f"模型资产路径越界：{value}")
        return resolved

    @staticmethod
    def _validate_asset_digest(
        path: str,
        expected_value: object,
        label: str,
    ) -> None:
        expected = str(expected_value).lower()
        digest = hashlib.sha256()
        with open(path, "rb") as asset_file:
            for block in iter(lambda: asset_file.read(1024 * 1024), b""):
                digest.update(block)
        if digest.hexdigest() != expected:
            raise ValueError(f"{label} SHA-256 与 manifest 不一致")

    def _validate_prototypes(self) -> np.ndarray:
        norms = np.empty(self.prototypes.shape[0], dtype=np.float32)
        count = self.prototypes.shape[0]
        for start in range(0, count, self.prototype_block_size):
            end = min(count, start + self.prototype_block_size)
            block = np.asarray(
                self.prototypes[start:end],
                dtype=np.float32,
            )
            if not np.isfinite(block).all():
                raise ValueError("prototypes.npy 含非有限值")
            block_norms = np.linalg.norm(block, axis=1)
            if np.any(block_norms <= 1e-12):
                raise ValueError("prototypes.npy 存在范数为 0 的原型行")
            norms[start:end] = block_norms
        return norms

    @staticmethod
    def _positive_int(value: object, field_name: str) -> int:
        try:
            result = int(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{field_name} 必须是正整数") from exc
        if result <= 0:
            raise ValueError(f"{field_name} 必须是正整数")
        return result

    @staticmethod
    def _channel_values(value: object, field_name: str) -> np.ndarray:
        try:
            result = np.asarray(value, dtype=np.float32)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{field_name} 必须包含 3 个数值") from exc
        if result.shape != (3,) or not np.isfinite(result).all():
            raise ValueError(f"{field_name} 必须包含 3 个有限数值")
        return result.reshape((1, 1, 3))

    def _ensure_session(self):
        if self._session_error:
            raise RuntimeError(self._session_error)
        if self._session is None:
            try:
                session = self._session_factory(self.model_path)
                inputs = session.get_inputs()
                outputs = session.get_outputs()
                if not inputs or not outputs:
                    raise ValueError("ONNX 模型没有有效输入或输出")
                self._input_name = str(
                    self._manifest.get("input_name") or inputs[0].name
                )
                self._output_name = str(
                    self._manifest.get("output_name") or outputs[0].name
                )
                self._session = session
            except Exception as exc:
                self._session = None
                self._session_error = str(exc)
                raise
        return self._session

    def _preprocess(self, crop_rgb) -> np.ndarray:
        if isinstance(crop_rgb, Image.Image):
            image = crop_rgb.convert("RGB")
        else:
            array = np.asarray(crop_rgb)
            if array.ndim != 3 or array.shape[2] not in (3, 4):
                raise ValueError("crop_rgb 必须是 PIL RGB 或 HxWx3/4 数组")
            image = Image.fromarray(array.astype(np.uint8)).convert("RGB")
        if image.width <= 0 or image.height <= 0:
            raise ValueError("crop_rgb 尺寸无效")
        image = pad_glyph_image(
            image,
            (self.input_width, self.input_height),
        )
        array = np.asarray(image, dtype=np.float32) / 255.0
        if self.input_color == "BGR":
            array = array[:, :, ::-1]
        array = (array - self.input_mean) / self.input_std
        return np.ascontiguousarray(array.transpose(2, 0, 1)[None])

    def _encode_batch(self, crops) -> np.ndarray:
        crop_list = list(crops)
        embedding_size = self.prototypes.shape[1]
        if not crop_list:
            return np.empty((0, embedding_size), dtype=np.float32)
        tensor = np.concatenate(
            [self._preprocess(crop) for crop in crop_list],
            axis=0,
        )
        session = self._ensure_session()
        output = session.run(
            [self._output_name],
            {self._input_name: tensor},
        )[0]
        output = np.asarray(output, dtype=np.float32)
        batch_size = len(crop_list)
        if output.ndim == 0 or output.shape[0] != batch_size:
            output_batch_size = 0 if output.ndim == 0 else output.shape[0]
            raise ValueError(
                "模型 embedding batch 数与输入不一致："
                f"{output_batch_size} != {batch_size}"
            )
        embeddings = output.reshape(batch_size, -1)
        if embeddings.shape[1] != embedding_size:
            raise ValueError(
                "模型 embedding 维度与原型不一致："
                f"{embeddings.shape[1]} != {embedding_size}"
            )
        if not np.isfinite(embeddings).all():
            raise ValueError("模型 embedding 含非有限值")
        norms = np.linalg.norm(embeddings, axis=1)
        if np.any(norms <= 1e-12):
            raise ValueError("模型 embedding 范数为 0")
        return embeddings / norms[:, None]

    def _encode(self, crop_rgb) -> np.ndarray:
        return self._encode_batch([crop_rgb])[0]

    def _ensure_prototype_norms(self) -> np.ndarray:
        if self._prototype_norms is None:
            norms = np.empty(self.prototypes.shape[0], dtype=np.float32)
            count = self.prototypes.shape[0]
            for start in range(0, count, self.prototype_block_size):
                end = min(count, start + self.prototype_block_size)
                block = np.asarray(
                    self.prototypes[start:end],
                    dtype=np.float32,
                )
                norms[start:end] = np.linalg.norm(block, axis=1)
            self._prototype_norms = norms
        return self._prototype_norms

    def _match_embeddings(
        self,
        embeddings: np.ndarray,
        top_k: int,
    ) -> list:
        if top_k <= 0:
            return [[] for _embedding in embeddings]
        best_by_embedding = [[] for _embedding in embeddings]
        prototype_norms = self._ensure_prototype_norms()
        count = self.prototypes.shape[0]
        for start in range(0, count, self.prototype_block_size):
            end = min(count, start + self.prototype_block_size)
            block = np.asarray(
                self.prototypes[start:end],
                dtype=np.float32,
            )
            norms = prototype_norms[start:end]
            valid = norms > 1e-12
            cosine = np.full(
                (block.shape[0], embeddings.shape[0]),
                -1.0,
                dtype=np.float32,
            )
            cosine[valid] = (
                block[valid] @ embeddings.T
            ) / norms[valid, None]
            scores = np.clip((cosine + 1.0) * 0.5, 0.0, 1.0)
            block_k = min(top_k, scores.shape[0])
            if block_k <= 0:
                continue
            for embedding_index in range(embeddings.shape[0]):
                column_scores = scores[:, embedding_index]
                indices = np.argpartition(
                    column_scores,
                    -block_k,
                )[-block_k:]
                best_by_embedding[embedding_index].extend(
                    (
                        float(column_scores[index]),
                        start + int(index),
                    )
                    for index in indices
                )

        results = []
        for best in best_by_embedding:
            best.sort(key=lambda pair: (-pair[0], pair[1]))
            best = best[: min(top_k, count)]
            results.append(
                [
                    {
                        "char": self.characters[index],
                        "score": score,
                        "rank": rank,
                    }
                    for rank, (score, index) in enumerate(best, start=1)
                ]
            )
        return results

    def review_batch(self, crops, top_k: int = 5) -> list:
        """批量编码、一次分块匹配多个切片，避免重复推理与扫描。"""

        crop_list = list(crops)
        if not crop_list:
            return []
        if top_k <= 0:
            return [[] for _crop in crop_list]
        embeddings = self._encode_batch(crop_list)
        return self._match_embeddings(embeddings, top_k)

    def review(self, crop_rgb, top_k: int = 5) -> list:
        """返回 ``[{char, score, rank}, ...]``，score 为归一化 cosine。"""

        return self.review_batch([crop_rgb], top_k=top_k)[0]


class LazyOnnxGlyphReviewer:
    """首次收到待复核切片时才校验资产并构建 ``OnnxGlyphReviewer``。"""

    def __init__(
        self,
        asset_dir: str,
        *,
        reviewer_factory: Callable[[str], object] = OnnxGlyphReviewer,
    ):
        self.asset_dir = asset_dir
        self._reviewer_factory = reviewer_factory
        self._reviewer = None
        self.metadata = {}
        self.load_error = ""

    def _ensure_reviewer(self):
        if self._reviewer is None:
            if self.load_error:
                raise RuntimeError(self.load_error)
            try:
                self._reviewer = self._reviewer_factory(self.asset_dir)
                metadata = getattr(self._reviewer, "metadata", {})
                if isinstance(metadata, dict):
                    self.metadata = dict(metadata)
            except Exception as exc:
                self.load_error = str(exc)
                raise
        return self._reviewer

    def _remember_session_error(self) -> None:
        error = str(getattr(self._reviewer, "_session_error", ""))
        if error:
            self.load_error = error

    def review(self, crop_rgb, top_k: int = 5) -> list:
        reviewer = self._ensure_reviewer()
        try:
            return reviewer.review(crop_rgb, top_k=top_k)
        except Exception:
            self._remember_session_error()
            raise

    def review_batch(self, crops, top_k: int = 5) -> list:
        reviewer = self._ensure_reviewer()
        try:
            batch_method = getattr(reviewer, "review_batch", None)
            if batch_method is not None:
                return batch_method(crops, top_k=top_k)
            return [
                reviewer.review(crop, top_k=top_k)
                for crop in crops
            ]
        except Exception:
            self._remember_session_error()
            raise
