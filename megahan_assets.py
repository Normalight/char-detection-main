# -*- coding: utf-8 -*-
"""MegaHan97K 官方码本与 IDS 元数据的解析工具。"""

import hashlib
from os import PathLike
from pathlib import Path
from typing import Iterable, Optional, Union


MEGAHAN_CHARACTER_COUNT = 97_455
MEGAHAN_SOURCE = "https://github.com/SCUT-DLVCLab/MegaHan97K"
MEGAHAN_LICENSE = "CC BY-NC-ND 4.0"

PathValue = Union[str, PathLike]


def parse_megahan_codebook(
    path: PathValue,
    expected_count: int = MEGAHAN_CHARACTER_COUNT,
) -> list[str]:
    """读取并严格校验 MegaHan 每行一个字符的 UTF-8 码本。"""

    lines = Path(path).read_text(encoding="utf-8-sig").splitlines()
    characters = []
    seen = set()

    for line_number, line in enumerate(lines, start=1):
        if not line:
            raise ValueError(f"MegaHan 码本第 {line_number} 行为空行")
        if len(line) != 1:
            raise ValueError(
                f"MegaHan 码本第 {line_number} 行必须是单个 Unicode 字符"
            )
        if line in seen:
            raise ValueError(
                f"MegaHan 码本第 {line_number} 行包含重复字符：{line}"
            )
        characters.append(line)
        seen.add(line)

    if len(characters) != expected_count:
        raise ValueError(
            "MegaHan 码本字符数量不符："
            f"期望 {expected_count}，实际 {len(characters)}"
        )
    return characters


def parse_megahan_ids(
    path: PathValue,
    characters: Iterable[str],
) -> dict[str, str]:
    """解析 MegaHan IDS，支持冒号、制表符和普通空白分隔。"""

    known_characters = set(characters)
    parsed = {}
    lines = Path(path).read_text(encoding="utf-8-sig").splitlines()

    for line_number, raw_line in enumerate(lines, start=1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue

        key = line[0]
        remainder = line[1:]
        if not remainder:
            expression = ""
        elif remainder.startswith(":"):
            expression = remainder[1:]
        elif remainder[0].isspace():
            expression = remainder.lstrip()
            if expression.startswith(":"):
                expression = expression[1:]
        else:
            raise ValueError(f"MegaHan IDS 第 {line_number} 行为非法行")

        expression = expression.strip()
        if len(key) != 1:
            raise ValueError(f"MegaHan IDS 第 {line_number} 行为非法行")
        if not expression:
            raise ValueError(
                f"MegaHan IDS 第 {line_number} 行缺少 IDS 表达式"
            )
        if key not in known_characters:
            raise ValueError(
                f"MegaHan IDS 第 {line_number} 行包含未知字符：{key}"
            )
        if key in parsed:
            raise ValueError(
                f"MegaHan IDS 第 {line_number} 行包含重复字符：{key}"
            )
        parsed[key] = expression

    return parsed


def sha256_file(path: PathValue) -> str:
    """流式计算文件的 SHA-256 十六进制摘要。"""

    digest = hashlib.sha256()
    with Path(path).open("rb") as input_file:
        for block in iter(lambda: input_file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_megahan_provenance(
    codebook_sha256: str,
    ids_sha256: Optional[str] = None,
) -> dict:
    """生成可随模型资产保存的基础 MegaHan 来源信息。"""

    provenance = {
        "source": MEGAHAN_SOURCE,
        "license": MEGAHAN_LICENSE,
        "character_count": MEGAHAN_CHARACTER_COUNT,
        "codebook_sha256": codebook_sha256,
    }
    if ids_sha256 is not None:
        provenance["ids_sha256"] = ids_sha256
    return provenance


def _positive_integer(value, field_name: str) -> int:
    if (
        not isinstance(value, int)
        or isinstance(value, bool)
        or value <= 0
    ):
        raise ValueError(f"{field_name} must be a positive integer")
    return value


def _non_empty_text(value, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")
    return value


def _sha256_digest(value, field_name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdefABCDEF" for character in value)
    ):
        raise ValueError(f"{field_name} must be a SHA-256 hex digest")
    return value.lower()


def build_reviewer_manifest(
    asset_dir: PathValue,
    model_id: str,
    prototype_version: str,
    character_count: int,
    input_size: int,
    embedding_size: int,
    *,
    architecture: str,
    codebook_sha256: str,
    characters_sha256: str,
    dataset_fingerprint: str,
    checkpoint_sha256: str,
) -> dict:
    """Build a version-1 manifest for ``OnnxGlyphReviewer`` assets."""

    root = Path(asset_dir).expanduser().resolve(strict=False)
    if not root.is_dir():
        raise ValueError(f"reviewer asset_dir is not a directory: {root}")

    character_count = _positive_integer(
        character_count,
        "character_count",
    )
    input_size = _positive_integer(input_size, "input_size")
    embedding_size = _positive_integer(
        embedding_size,
        "embedding_size",
    )
    model_id = _non_empty_text(model_id, "model_id")
    prototype_version = _non_empty_text(
        prototype_version,
        "prototype_version",
    )
    architecture = _non_empty_text(architecture, "architecture")
    codebook_sha256 = _sha256_digest(
        codebook_sha256,
        "codebook_sha256",
    )
    expected_characters_sha256 = _sha256_digest(
        characters_sha256,
        "characters_sha256",
    )
    dataset_fingerprint = _sha256_digest(
        dataset_fingerprint,
        "dataset_fingerprint",
    )
    checkpoint_sha256 = _sha256_digest(
        checkpoint_sha256,
        "checkpoint_sha256",
    )

    filenames = {
        "model": "glyph_encoder.onnx",
        "characters": "characters.txt",
        "prototypes": "prototypes.npy",
    }
    paths = {}
    for label, filename in filenames.items():
        path = root / filename
        if not path.is_file():
            raise FileNotFoundError(
                f"required reviewer asset is missing: {path}"
            )
        paths[label] = path

    asset_hashes = {
        label: sha256_file(path) for label, path in paths.items()
    }
    if asset_hashes["characters"] != expected_characters_sha256:
        raise ValueError(
            "characters_sha256 does not match characters.txt: "
            f"{expected_characters_sha256} "
            f"!= {asset_hashes['characters']}"
        )

    return {
        "version": 1,
        "model_id": model_id,
        "prototype_version": prototype_version,
        "source": MEGAHAN_SOURCE,
        "license": MEGAHAN_LICENSE,
        "model": filenames["model"],
        "model_sha256": asset_hashes["model"],
        "characters": filenames["characters"],
        "characters_sha256": asset_hashes["characters"],
        "prototypes": filenames["prototypes"],
        "prototypes_sha256": asset_hashes["prototypes"],
        "input": {
            "height": input_size,
            "width": input_size,
            "color": "RGB",
            "layout": "NCHW",
            "mean": [0.5, 0.5, 0.5],
            "std": [0.5, 0.5, 0.5],
        },
        "megahan": {
            "official_character_count": MEGAHAN_CHARACTER_COUNT,
            "export_character_count": character_count,
            "architecture": architecture,
            "embedding_size": embedding_size,
            "codebook_sha256": codebook_sha256,
            "characters_sha256": asset_hashes["characters"],
            "dataset_fingerprint": dataset_fingerprint,
            "checkpoint_sha256": checkpoint_sha256,
        },
    }
