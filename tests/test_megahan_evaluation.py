# -*- coding: utf-8 -*-
"""Tests for the MegaHan reviewer evaluation CLI and pure metrics."""

import importlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from PIL import Image


_MODULE_NAME = "bench.evaluate_megahan_reviewer"
_MODULE_SPEC = importlib.util.find_spec(_MODULE_NAME)
evaluation = (
    importlib.import_module(_MODULE_NAME) if _MODULE_SPEC is not None else None
)


class EvaluationModuleContractTests(unittest.TestCase):
    def test_evaluation_module_exists(self):
        self.assertIsNotNone(
            evaluation,
            "bench.evaluate_megahan_reviewer must exist",
        )


@unittest.skipIf(evaluation is None, "implementation module does not exist yet")
class EvaluatePredictionsTests(unittest.TestCase):
    def test_measures_top1_top5_and_fallback_without_promoting_top5_truth(self):
        metrics = evaluation.evaluate_predictions(
            [
                {
                    "truth": "澆",
                    "fallback": "浇",
                    "candidates": [
                        {"char": "澆", "score": 0.9, "rank": 1},
                        {"char": "浇", "score": 0.8, "rank": 2},
                    ],
                },
                {
                    "truth": "𠀀",
                    "fallback": "",
                    "candidates": ["一", "丁"],
                },
            ]
        )

        self.assertEqual(metrics["count"], 2)
        self.assertEqual(metrics["fallback_top1_count"], 0)
        self.assertEqual(metrics["fallback_top1_correct_count"], 0)
        self.assertEqual(metrics["fallback_top1_accuracy"], 0.0)
        self.assertEqual(metrics["top1_count"], 1)
        self.assertEqual(metrics["top1_correct_count"], 1)
        self.assertEqual(metrics["top1_accuracy"], 0.5)
        self.assertEqual(metrics["top5_count"], 1)
        self.assertEqual(metrics["top5_correct_count"], 1)
        self.assertEqual(metrics["top5_recall"], 0.5)
        self.assertEqual(metrics["candidate_recall"], 0.5)
        self.assertEqual(metrics["wrong_top1_count"], 1)
        self.assertEqual(metrics["fallback_wrong_count"], 2)
        self.assertEqual(metrics["top1_gain"], 0.5)
        self.assertEqual(metrics["top5_gain_vs_fallback"], 0.5)

    def test_preserves_candidate_order_while_filtering_invalid_and_duplicates(self):
        metrics = evaluation.evaluate_predictions(
            [
                {
                    "truth": "澆",
                    "fallback": "澆",
                    "candidates": [
                        "",
                        {"char": "浇", "score": 0.99, "rank": 99},
                        {"char": "澆", "score": 1.0, "rank": 1},
                        "浇",
                        {"char": "两个", "score": 0.8, "rank": 2},
                        {"score": 0.7},
                    ],
                }
            ]
        )

        self.assertEqual(metrics["top1_correct_count"], 0)
        self.assertEqual(metrics["top5_correct_count"], 1)
        self.assertEqual(metrics["wrong_top1_count"], 1)
        self.assertEqual(metrics["fallback_wrong_count"], 0)

    def test_rejects_empty_records_instead_of_reporting_vacuous_success(self):
        with self.assertRaisesRegex(ValueError, "至少一条"):
            evaluation.evaluate_predictions([])

    def test_normalizes_whitespace_only_fallback_to_empty(self):
        metrics = evaluation.evaluate_predictions(
            [{"truth": "字", "fallback": " \t", "candidates": ["字"]}]
        )

        self.assertEqual(metrics["fallback_top1_count"], 0)
        self.assertEqual(metrics["fallback_wrong_count"], 1)

    def test_rejects_invalid_truth_fallback_or_candidate_container(self):
        invalid_records = (
            [{"truth": "", "fallback": "", "candidates": []}],
            [{"truth": "两个", "fallback": "", "candidates": []}],
            [{"truth": "字", "fallback": "两个", "candidates": []}],
            [{"truth": "字", "fallback": "", "candidates": "字"}],
        )
        for records in invalid_records:
            with self.subTest(records=records):
                with self.assertRaises(ValueError):
                    evaluation.evaluate_predictions(records)


@unittest.skipIf(evaluation is None, "implementation module does not exist yet")
class RegressionGateTests(unittest.TestCase):
    @staticmethod
    def metrics(top5, fallback, wrong=1, fallback_wrong=1):
        return {
            "top5_recall": top5,
            "fallback_top1_accuracy": fallback,
            "wrong_top1_count": wrong,
            "fallback_wrong_count": fallback_wrong,
        }

    def test_passes_only_for_strict_recall_gain_without_more_top1_errors(self):
        gate = evaluation.regression_gate(
            self.metrics(0.8, 0.7, wrong=2, fallback_wrong=2)
        )

        self.assertTrue(gate["passed"])
        self.assertIn("strict", gate["reason"].lower())

    def test_equal_recall_is_not_an_improvement(self):
        gate = evaluation.regression_gate(self.metrics(0.7, 0.7))

        self.assertFalse(gate["passed"])
        self.assertIn("strict", gate["reason"].lower())

    def test_recall_regression_fails(self):
        gate = evaluation.regression_gate(self.metrics(0.6, 0.7))

        self.assertFalse(gate["passed"])
        self.assertIn("recall", gate["reason"].lower())

    def test_more_wrong_top1_predictions_fails_even_with_recall_gain(self):
        gate = evaluation.regression_gate(
            self.metrics(0.8, 0.7, wrong=3, fallback_wrong=2)
        )

        self.assertFalse(gate["passed"])
        self.assertIn("wrong_top1", gate["reason"])


@unittest.skipIf(evaluation is None, "implementation module does not exist yet")
class EvaluationEndToEndTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory(
            prefix="megahan_evaluation_"
        )
        self.addCleanup(self.temp_dir.cleanup)
        self.root = Path(self.temp_dir.name)
        self.asset_dir = self.root / "assets"
        self.asset_dir.mkdir()
        (self.asset_dir / "glyph_encoder.onnx").write_bytes(b"model")
        (self.asset_dir / "characters.txt").write_text(
            "澆\n浇\n𠀀\n",
            encoding="utf-8",
        )
        (self.asset_dir / "prototypes.npy").write_bytes(b"prototype-data")
        (self.asset_dir / "ignored.bin").write_bytes(b"not-in-manifest")
        (self.asset_dir / "manifest.json").write_text(
            json.dumps(
                {
                    "model": "glyph_encoder.onnx",
                    "characters": "characters.txt",
                    "prototypes": "prototypes.npy",
                }
            ),
            encoding="utf-8",
        )
        self.crops = self.root / "crops"
        self.crops.mkdir()
        Image.new("L", (4, 3), 128).save(self.crops / "one.png")
        Image.new("RGBA", (5, 2), (1, 2, 3, 200)).save(
            self.crops / "two.png"
        )
        self.input_path = self.root / "cases.jsonl"
        self.input_path.write_text(
            "\n".join(
                [
                    json.dumps(
                        {
                            "crop": "crops/one.png",
                            "truth": "澆",
                            "fallback": "浇",
                        },
                        ensure_ascii=False,
                    ),
                    json.dumps(
                        {
                            "crop": str((self.crops / "two.png").resolve()),
                            "truth": "𠀀",
                        },
                        ensure_ascii=False,
                    ),
                ]
            )
            + "\n",
            encoding="utf-8",
        )

    def test_asset_sizes_include_characters_and_sum_only_manifest_assets(self):
        _root, sizes = evaluation._asset_sizes(self.asset_dir)

        character_size = len("澆\n浇\n𠀀\n".encode("utf-8"))
        self.assertEqual(sizes["characters_bytes"], character_size)
        self.assertEqual(
            sizes["asset_total_bytes"],
            len(b"model") + character_size + len(b"prototype-data"),
        )

    def test_reports_raw_batch_and_per_crop_latency_percentiles(self):
        class TimingReviewer:
            metadata = {}

            def __init__(self, _asset_dir):
                self.call_count = 0

            def review_batch(self, crops, top_k=5):
                outputs = [["澆"], ["𠀀"]]
                result = [outputs[self.call_count] for _crop in crops]
                self.call_count += 1
                return result

        clock = iter([1.0, 1.0, 2.0, 2.2, 3.0, 3.6])
        report = evaluation.run_evaluation(
            input_path=self.input_path,
            asset_dir=self.asset_dir,
            batch_size=1,
            top_k=5,
            reviewer_factory=TimingReviewer,
            clock=lambda: next(clock),
            peak_rss_provider=lambda: None,
        )

        self.assertEqual(report["batch_p50_seconds"], 0.4)
        self.assertEqual(report["batch_p95_seconds"], 0.58)
        self.assertEqual(report["per_crop_seconds"]["p50"], 0.4)
        self.assertEqual(report["per_crop_seconds"]["p95"], 0.58)

    def test_builds_auditable_report_with_batches_sizes_metadata_and_rgb(self):
        instances = []
        lifecycle_events = []

        class FakeReviewer:
            metadata = {
                "model_id": "fake-megahan",
                "prototype_version": "test-v1",
            }

            def __init__(self, asset_dir):
                self.asset_dir = asset_dir
                self.calls = []
                self.loaded = False
                instances.append(self)
                lifecycle_events.append("constructed")

            def _ensure_session(self):
                self.loaded = True
                lifecycle_events.append("loaded")

            def review_batch(self, crops, top_k=5):
                if not self.loaded:
                    raise AssertionError("lazy session was not primed")
                lifecycle_events.append("reviewed")
                self.calls.append((list(crops), top_k))
                return [
                    [
                        {"char": "澆", "score": 0.9, "rank": 1},
                        {"char": "浇", "score": 0.7, "rank": 2},
                    ]
                    if len(self.calls) == 1
                    else [{"char": "一", "score": 0.8, "rank": 1}]
                    for _crop in crops
                ]

        clock = iter([10.0, 10.25, 11.0, 11.4, 12.0, 12.6])
        report = evaluation.run_evaluation(
            input_path=self.input_path,
            asset_dir=self.asset_dir,
            batch_size=1,
            top_k=5,
            reviewer_factory=FakeReviewer,
            clock=lambda: next(clock),
            peak_rss_provider=lambda: 4321,
        )

        self.assertEqual(report["input"], str(self.input_path.resolve()))
        self.assertEqual(report["reviewer"], FakeReviewer.metadata)
        self.assertEqual(report["model_load_seconds"], 0.25)
        self.assertEqual(report["total_review_seconds"], 1.0)
        self.assertEqual(
            report["batch_timings"],
            [
                {
                    "batch_index": 1,
                    "count": 1,
                    "elapsed_seconds": 0.4,
                    "per_crop_seconds": 0.4,
                },
                {
                    "batch_index": 2,
                    "count": 1,
                    "elapsed_seconds": 0.6,
                    "per_crop_seconds": 0.6,
                },
            ],
        )
        self.assertEqual(report["per_crop_seconds"]["samples"], [0.4, 0.6])
        self.assertEqual(report["per_crop_seconds"]["p50"], 0.5)
        self.assertEqual(report["per_crop_seconds"]["p95"], 0.59)
        self.assertEqual(report["batch_p50_seconds"], 0.5)
        self.assertEqual(report["batch_p95_seconds"], 0.59)
        self.assertEqual(report["model_bytes"], len(b"model"))
        self.assertEqual(
            report["characters_bytes"],
            len("澆\n浇\n𠀀\n".encode("utf-8")),
        )
        self.assertEqual(report["prototype_bytes"], len(b"prototype-data"))
        expected_total = (
            len(b"model")
            + len("澆\n浇\n𠀀\n".encode("utf-8"))
            + len(b"prototype-data")
        )
        self.assertEqual(report["asset_total_bytes"], expected_total)
        self.assertEqual(report["peak_rss_bytes"], 4321)
        self.assertEqual(report["metrics"]["top1_accuracy"], 0.5)
        self.assertEqual(len(report["predictions"]), 2)
        self.assertEqual(report["predictions"][0]["fallback"], "浇")
        self.assertEqual(report["predictions"][1]["fallback"], "")
        self.assertEqual(
            report["predictions"][0]["candidates"][0]["char"],
            "澆",
        )
        self.assertIn("generated_at_utc", report)
        self.assertIn("gate", report)
        self.assertEqual(len(instances), 1)
        self.assertEqual(len(instances[0].calls), 2)
        self.assertEqual(
            lifecycle_events,
            ["constructed", "loaded", "reviewed", "reviewed"],
        )
        self.assertTrue(
            all(
                crop.mode == "RGB"
                for call, _top_k in instances[0].calls
                for crop in call
            )
        )
        self.assertTrue(all(top_k == 5 for _crops, top_k in instances[0].calls))

    def test_rejects_missing_crop_with_line_number(self):
        self.input_path.write_text(
            '{"crop":"missing.png","truth":"字"}\n',
            encoding="utf-8",
        )

        with self.assertRaisesRegex(FileNotFoundError, "第 1 行.*missing"):
            evaluation.run_evaluation(
                input_path=self.input_path,
                asset_dir=self.asset_dir,
                reviewer_factory=lambda _path: self.fail(
                    "reviewer must not load"
                ),
            )

    def test_model_error_propagates_instead_of_becoming_fallback(self):
        class FailingReviewer:
            metadata = {}

            def __init__(self, _asset_dir):
                pass

            def review_batch(self, _crops, top_k=5):
                raise RuntimeError("model exploded")

        with self.assertRaisesRegex(RuntimeError, "model exploded"):
            evaluation.run_evaluation(
                input_path=self.input_path,
                asset_dir=self.asset_dir,
                reviewer_factory=FailingReviewer,
            )

    def test_rejects_wrong_number_of_batch_results(self):
        class ShortReviewer:
            metadata = {}

            def __init__(self, _asset_dir):
                pass

            def review_batch(self, _crops, top_k=5):
                return []

        with self.assertRaisesRegex(ValueError, "结果数量"):
            evaluation.run_evaluation(
                input_path=self.input_path,
                asset_dir=self.asset_dir,
                reviewer_factory=ShortReviewer,
            )


@unittest.skipIf(evaluation is None, "implementation module does not exist yet")
class CliAndPlatformTests(unittest.TestCase):
    def test_cli_returns_two_when_improvement_is_required_but_gate_fails(self):
        report = {"gate": {"passed": False, "reason": "no strict gain"}}
        with mock.patch.object(
            evaluation,
            "run_evaluation",
            return_value=report,
        ), mock.patch.object(evaluation, "atomic_write_json") as write_json:
            result = evaluation.main(
                [
                    "--input",
                    "cases.jsonl",
                    "--asset-dir",
                    "assets",
                    "--output",
                    "report.json",
                    "--require-improvement",
                ]
            )

        self.assertEqual(result, 2)
        write_json.assert_called_once_with(
            "report.json",
            report,
            overwrite=False,
        )

    def test_cli_force_explicitly_allows_replacing_the_output(self):
        report = {"gate": {"passed": True, "reason": "passed"}}
        with mock.patch.object(
            evaluation,
            "run_evaluation",
            return_value=report,
        ), mock.patch.object(evaluation, "atomic_write_json") as write_json:
            result = evaluation.main(
                [
                    "--input",
                    "cases.jsonl",
                    "--asset-dir",
                    "assets",
                    "--output",
                    "report.json",
                    "--force",
                ]
            )

        self.assertEqual(result, 0)
        write_json.assert_called_once_with(
            "report.json",
            report,
            overwrite=True,
        )

    def test_cli_returns_one_and_logs_unexpected_errors(self):
        with mock.patch.object(
            evaluation,
            "run_evaluation",
            side_effect=RuntimeError("broken model"),
        ), mock.patch.object(evaluation.logger, "exception") as log_exception:
            result = evaluation.main(
                [
                    "--input",
                    "cases.jsonl",
                    "--asset-dir",
                    "assets",
                    "--output",
                    "report.json",
                ]
            )

        self.assertEqual(result, 1)
        log_exception.assert_called_once()

    def test_argument_parser_requires_top_k_at_least_five(self):
        parser = evaluation.build_argument_parser()
        with self.assertRaisesRegex(
            ValueError,
            "top-k.*至少为 5",
        ):
            evaluation._parse_cli_args(
                parser,
                [
                    "--input",
                    "cases.jsonl",
                    "--asset-dir",
                    "assets",
                    "--output",
                    "report.json",
                    "--top-k",
                    "4",
                ],
            )

    def test_peak_rss_conversion_handles_macos_bytes_and_linux_kib(self):
        self.assertEqual(
            evaluation.peak_rss_bytes(123, platform_name="darwin"),
            123,
        )
        self.assertEqual(
            evaluation.peak_rss_bytes(123, platform_name="linux"),
            123 * 1024,
        )
        self.assertIsNone(
            evaluation.peak_rss_bytes(None, platform_name="linux")
        )


@unittest.skipIf(evaluation is None, "implementation module does not exist yet")
class AtomicOutputTests(unittest.TestCase):
    def test_default_atomically_creates_new_utf8_output_with_link(self):
        with tempfile.TemporaryDirectory(prefix="megahan_atomic_") as folder:
            output = Path(folder) / "report.json"
            real_link = evaluation.os.link
            link_calls = []

            def recording_link(source, destination):
                link_calls.append((Path(source), Path(destination)))
                return real_link(source, destination)

            with mock.patch.object(
                evaluation.os,
                "link",
                side_effect=recording_link,
            ), mock.patch.object(
                evaluation.os,
                "replace",
                side_effect=AssertionError("default must not replace"),
            ):
                evaluation.atomic_write_json(
                    output,
                    {"truth": "𠀀", "char": "澆"},
                )

            self.assertEqual(len(link_calls), 1)
            self.assertEqual(
                link_calls[0][1],
                output.resolve(strict=False),
            )
            text = output.read_text(encoding="utf-8")
            self.assertNotIn("\\u", text)
            self.assertEqual(json.loads(text), {"truth": "𠀀", "char": "澆"})
            self.assertEqual(
                [path.name for path in Path(folder).iterdir()],
                ["report.json"],
            )

    def test_default_refuses_existing_output_without_changing_its_content(self):
        with tempfile.TemporaryDirectory(prefix="megahan_atomic_") as folder:
            output = Path(folder) / "report.json"
            original = '{"old":true}\n'
            output.write_text(original, encoding="utf-8")

            with self.assertRaises(FileExistsError):
                evaluation.atomic_write_json(
                    output,
                    {"truth": "𠀀", "nested": {"char": "澆"}},
                )

            self.assertEqual(output.read_text(encoding="utf-8"), original)
            self.assertEqual(
                [path.name for path in Path(folder).iterdir()],
                ["report.json"],
            )

    def test_force_atomically_replaces_json_with_utf8_and_fsync(self):
        with tempfile.TemporaryDirectory(prefix="megahan_atomic_") as folder:
            output = Path(folder) / "report.json"
            output.write_text('{"old":true}', encoding="utf-8")
            real_fsync = evaluation.os.fsync
            real_replace = evaluation.os.replace
            fsync_calls = []
            replace_calls = []

            def recording_fsync(fd):
                fsync_calls.append(fd)
                return real_fsync(fd)

            def recording_replace(source, destination):
                replace_calls.append((Path(source), Path(destination)))
                return real_replace(source, destination)

            with mock.patch.object(
                evaluation.os,
                "fsync",
                side_effect=recording_fsync,
            ), mock.patch.object(
                evaluation.os,
                "replace",
                side_effect=recording_replace,
            ):
                evaluation.atomic_write_json(
                    output,
                    {"truth": "𠀀", "char": "澆"},
                    overwrite=True,
                )

            self.assertTrue(fsync_calls)
            self.assertEqual(len(replace_calls), 1)
            self.assertEqual(
                replace_calls[0][1],
                output.resolve(strict=False),
            )
            text = output.read_text(encoding="utf-8")
            self.assertNotIn("\\u", text)
            self.assertEqual(json.loads(text), {"truth": "𠀀", "char": "澆"})
            self.assertEqual(
                [path.name for path in Path(folder).iterdir()],
                ["report.json"],
            )

    def test_no_clobber_publish_loses_race_without_overwriting_winner(self):
        with tempfile.TemporaryDirectory(prefix="megahan_atomic_") as folder:
            output = Path(folder) / "report.json"
            real_link = evaluation.os.link

            def competing_link(source, destination):
                Path(destination).write_text(
                    '{"winner":true}\n',
                    encoding="utf-8",
                )
                return real_link(source, destination)

            with mock.patch.object(
                evaluation.os,
                "link",
                side_effect=competing_link,
            ):
                with self.assertRaises(FileExistsError):
                    evaluation.atomic_write_json(output, {"ours": True})

            self.assertEqual(
                json.loads(output.read_text(encoding="utf-8")),
                {"winner": True},
            )
            self.assertEqual(
                [path.name for path in Path(folder).iterdir()],
                ["report.json"],
            )
