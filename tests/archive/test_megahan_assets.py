# -*- coding: utf-8 -*-
"""MegaHan97K 码本、IDS 和来源元数据解析测试。"""

import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

import megahan_assets
from rare_glyph_review import OnnxGlyphReviewer


class MegaHanAssetsTestCase(unittest.TestCase):
    def setUp(self):
        self.assets = megahan_assets
        self.temp_dir = tempfile.TemporaryDirectory(prefix="megahan_assets_")
        self.addCleanup(self.temp_dir.cleanup)

    def write_bytes(self, name, content):
        path = Path(self.temp_dir.name, name)
        path.write_bytes(content)
        return path

    def write_text(self, name, content):
        return self.write_bytes(name, content.encode("utf-8"))


class ConstantsTests(MegaHanAssetsTestCase):
    def test_official_metadata_constants(self):
        self.assertEqual(self.assets.MEGAHAN_CHARACTER_COUNT, 97_455)
        self.assertEqual(
            self.assets.MEGAHAN_SOURCE,
            "https://github.com/SCUT-DLVCLab/MegaHan97K",
        )
        self.assertEqual(self.assets.MEGAHAN_LICENSE, "CC BY-NC-ND 4.0")


class ParseCodebookTests(MegaHanAssetsTestCase):
    def test_preserves_non_bmp_character(self):
        path = self.write_text("codebook.txt", "一\n𠀀\n")

        result = self.assets.parse_megahan_codebook(path, expected_count=2)

        self.assertEqual(result, ["一", "𠀀"])
        self.assertEqual(len(result[1]), 1)

    def test_accepts_utf8_bom(self):
        path = self.write_bytes(
            "codebook.txt",
            b"\xef\xbb\xbf" + "一\n乙\n".encode(),
        )

        result = self.assets.parse_megahan_codebook(path, expected_count=2)

        self.assertEqual(result, ["一", "乙"])

    def test_rejects_character_count_mismatch(self):
        path = self.write_text("codebook.txt", "一\n乙\n")

        with self.assertRaisesRegex(ValueError, "数量"):
            self.assets.parse_megahan_codebook(path, expected_count=3)

    def test_rejects_empty_line(self):
        path = self.write_text("codebook.txt", "一\n\n乙\n")

        with self.assertRaisesRegex(ValueError, "空行"):
            self.assets.parse_megahan_codebook(path, expected_count=3)

    def test_rejects_line_with_multiple_code_points(self):
        path = self.write_text("codebook.txt", "一\n甲乙\n")

        with self.assertRaisesRegex(ValueError, "单个 Unicode 字符"):
            self.assets.parse_megahan_codebook(path, expected_count=2)

    def test_rejects_duplicate_character(self):
        path = self.write_text("codebook.txt", "一\n一\n")

        with self.assertRaisesRegex(ValueError, "重复字符"):
            self.assets.parse_megahan_codebook(path, expected_count=2)


class ParseIdsTests(MegaHanAssetsTestCase):
    def test_parses_official_colon_format_and_preserves_internal_spaces(self):
        path = self.write_text(
            "MegaHan_IDS.txt",
            "# MegaHan IDS\n\n一:一\n丒:⿱ 刃 一\n",
        )

        result = self.assets.parse_megahan_ids(path, ["一", "丒"])

        self.assertEqual(result, {"一": "一", "丒": "⿱ 刃 一"})

    def test_parses_tab_and_whitespace_variants(self):
        path = self.write_text(
            "MegaHan_IDS.txt",
            "一\t一\n乙   ⿱ 一 乚\n丒 : ⿱ 刃 一\n",
        )

        result = self.assets.parse_megahan_ids(path, ["一", "乙", "丒"])

        self.assertEqual(
            result,
            {
                "一": "一",
                "乙": "⿱ 一 乚",
                "丒": "⿱ 刃 一",
            },
        )

    def test_splits_only_on_first_colon(self):
        path = self.write_text("MegaHan_IDS.txt", "一:⿰ 一:一\n")

        result = self.assets.parse_megahan_ids(path, ["一"])

        self.assertEqual(result["一"], "⿰ 一:一")

    def test_whitespace_delimiter_preserves_colon_in_expression(self):
        path = self.write_text("MegaHan_IDS.txt", "一 ⿰ 一:一\n")

        result = self.assets.parse_megahan_ids(path, ["一"])

        self.assertEqual(result, {"一": "⿰ 一:一"})

    def test_tab_delimiter_preserves_colon_in_expression(self):
        path = self.write_text("MegaHan_IDS.txt", "一\t⿰ 一:一\n")

        result = self.assets.parse_megahan_ids(path, ["一"])

        self.assertEqual(result, {"一": "⿰ 一:一"})

    def test_allows_partial_ids_coverage(self):
        path = self.write_text("MegaHan_IDS.txt", "一:一\n")

        result = self.assets.parse_megahan_ids(path, ["一", "乙"])

        self.assertEqual(result, {"一": "一"})

    def test_rejects_unknown_character(self):
        path = self.write_text("MegaHan_IDS.txt", "乙:乙\n")

        with self.assertRaisesRegex(ValueError, "未知字符"):
            self.assets.parse_megahan_ids(path, ["一"])

    def test_rejects_duplicate_character(self):
        path = self.write_text("MegaHan_IDS.txt", "一:一\n一:⿰ 一 一\n")

        with self.assertRaisesRegex(ValueError, "重复字符"):
            self.assets.parse_megahan_ids(path, ["一"])

    def test_rejects_missing_expression(self):
        path = self.write_text("MegaHan_IDS.txt", "一:\n")

        with self.assertRaisesRegex(ValueError, "缺少 IDS 表达式"):
            self.assets.parse_megahan_ids(path, ["一"])

    def test_rejects_invalid_key_field(self):
        path = self.write_text("MegaHan_IDS.txt", "甲乙:⿰ 甲 乙\n")

        with self.assertRaisesRegex(ValueError, "非法行"):
            self.assets.parse_megahan_ids(path, ["甲", "乙"])


class HashAndProvenanceTests(MegaHanAssetsTestCase):
    def test_sha256_file_hashes_binary_content(self):
        content = b"MegaHan97K\x00metadata\n"
        path = self.write_bytes("asset.bin", content)

        result = self.assets.sha256_file(path)

        self.assertEqual(result, hashlib.sha256(content).hexdigest())

    def test_build_provenance_with_ids_hash(self):
        result = self.assets.build_megahan_provenance(
            codebook_sha256="codebook-hash",
            ids_sha256="ids-hash",
        )

        self.assertEqual(
            result,
            {
                "source": "https://github.com/SCUT-DLVCLab/MegaHan97K",
                "license": "CC BY-NC-ND 4.0",
                "character_count": 97_455,
                "codebook_sha256": "codebook-hash",
                "ids_sha256": "ids-hash",
            },
        )

    def test_build_provenance_omits_absent_ids_hash(self):
        result = self.assets.build_megahan_provenance("codebook-hash")

        self.assertNotIn("ids_sha256", result)
        self.assertEqual(result["character_count"], 97_455)


class ReviewerManifestTests(MegaHanAssetsTestCase):
    def write_reviewer_assets(self):
        model = self.write_bytes("glyph_encoder.onnx", b"fake-onnx")
        characters = self.write_text("characters.txt", "一\n𠀀\n")
        prototypes = Path(self.temp_dir.name, "prototypes.npy")
        np.save(
            prototypes,
            np.asarray(
                [[1.0, 0.0], [0.0, 1.0]],
                dtype=np.float16,
            ),
            allow_pickle=False,
        )
        return model, characters, prototypes

    def build_manifest(self, **overrides):
        _model, characters, _prototypes = self.write_reviewer_assets()
        arguments = {
            "asset_dir": self.temp_dir.name,
            "model_id": "megahan-lite-test",
            "prototype_version": "2026-07-test",
            "character_count": 2,
            "input_size": 64,
            "embedding_size": 2,
            "architecture": "test-encoder",
            "codebook_sha256": "a" * 64,
            "characters_sha256": self.assets.sha256_file(characters),
            "dataset_fingerprint": "b" * 64,
            "checkpoint_sha256": "c" * 64,
        }
        arguments.update(overrides)
        return self.assets.build_reviewer_manifest(**arguments)

    def test_builds_runtime_compatible_manifest_with_nested_provenance(self):
        manifest = self.build_manifest()

        self.assertEqual(manifest["version"], 1)
        self.assertEqual(manifest["model_id"], "megahan-lite-test")
        self.assertEqual(manifest["prototype_version"], "2026-07-test")
        self.assertEqual(manifest["source"], self.assets.MEGAHAN_SOURCE)
        self.assertEqual(manifest["license"], self.assets.MEGAHAN_LICENSE)
        self.assertEqual(manifest["model"], "glyph_encoder.onnx")
        self.assertEqual(manifest["characters"], "characters.txt")
        self.assertEqual(manifest["prototypes"], "prototypes.npy")
        self.assertEqual(
            manifest["model_sha256"],
            hashlib.sha256(b"fake-onnx").hexdigest(),
        )
        self.assertEqual(
            manifest["input"],
            {
                "height": 64,
                "width": 64,
                "color": "RGB",
                "layout": "NCHW",
                "mean": [0.5, 0.5, 0.5],
                "std": [0.5, 0.5, 0.5],
            },
        )
        self.assertEqual(
            manifest["megahan"],
            {
                "official_character_count": 97_455,
                "export_character_count": 2,
                "architecture": "test-encoder",
                "embedding_size": 2,
                "codebook_sha256": "a" * 64,
                "characters_sha256": manifest["characters_sha256"],
                "dataset_fingerprint": "b" * 64,
                "checkpoint_sha256": "c" * 64,
            },
        )
        for field in ("model", "characters", "prototypes"):
            self.assertFalse(Path(manifest[field]).is_absolute())

        Path(self.temp_dir.name, "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False),
            encoding="utf-8",
        )
        reviewer = OnnxGlyphReviewer(self.temp_dir.name)
        self.assertEqual(reviewer.characters, ("一", "𠀀"))
        self.assertEqual(reviewer.prototypes.shape, (2, 2))

    def test_rejects_missing_required_asset(self):
        self.write_reviewer_assets()
        Path(self.temp_dir.name, "glyph_encoder.onnx").unlink()

        with self.assertRaisesRegex(FileNotFoundError, "glyph_encoder"):
            self.assets.build_reviewer_manifest(
                asset_dir=self.temp_dir.name,
                model_id="megahan-lite-test",
                prototype_version="v1",
                character_count=2,
                input_size=64,
                embedding_size=2,
                architecture="test-encoder",
                codebook_sha256="a" * 64,
                characters_sha256="b" * 64,
                dataset_fingerprint="c" * 64,
                checkpoint_sha256="d" * 64,
            )

    def test_rejects_characters_digest_that_does_not_match_asset(self):
        with self.assertRaisesRegex(ValueError, "characters_sha256"):
            self.build_manifest(characters_sha256="f" * 64)

    def test_rejects_invalid_counts_and_digest_metadata(self):
        for override in (
            {"character_count": 0},
            {"input_size": True},
            {"embedding_size": -1},
            {"checkpoint_sha256": "not-a-digest"},
        ):
            with self.subTest(override=override):
                with self.assertRaises(ValueError):
                    self.build_manifest(**override)
