# -*- coding: utf-8 -*-
"""Evaluate the MegaHan glyph reviewer against a JSONL ground truth set."""

import argparse
from collections.abc import Iterable, Mapping
from datetime import datetime, timezone
import json
import logging
import os
from pathlib import Path
import platform
import sys
import tempfile
import time
import unicodedata

from PIL import Image


logger = logging.getLogger(__name__)

_INVALID_PATH_CHARS = frozenset('/\\:*?"<>|')
_PLACEHOLDER_CHARS = frozenset({"?", "."})
_MANIFEST_ASSET_FIELDS = ("model", "characters", "prototypes")


def _normalized_character(value):
    if not isinstance(value, str):
        return ""
    normalized = unicodedata.normalize("NFC", "".join(value.split()))
    if len(normalized) != 1:
        return ""
    character = normalized[0]
    if (
        character in _PLACEHOLDER_CHARS
        or character in _INVALID_PATH_CHARS
        or unicodedata.category(character).startswith("C")
    ):
        return ""
    return character


def _record_character(record, field, *, allow_empty=False):
    if field not in record:
        raise ValueError(f"评测记录缺少必填字段：{field}")
    value = record[field]
    if (
        allow_empty
        and isinstance(value, str)
        and not "".join(value.split())
    ):
        return ""
    character = _normalized_character(value)
    if not character:
        requirement = "单个 Unicode 字符或空串" if allow_empty else (
            "单个 Unicode 字符"
        )
        raise ValueError(f"{field} 必须是{requirement}")
    return character


def _ordered_candidate_characters(candidates):
    if (
        isinstance(candidates, (str, bytes, Mapping))
        or not isinstance(candidates, Iterable)
    ):
        raise ValueError("candidates 必须是候选数组")

    ordered = []
    seen = set()
    for candidate in candidates:
        if isinstance(candidate, str):
            value = candidate
        elif isinstance(candidate, Mapping):
            value = candidate.get("char")
        else:
            continue
        character = _normalized_character(value)
        if not character or character in seen:
            continue
        ordered.append(character)
        seen.add(character)
    return ordered


def evaluate_predictions(records):
    """Return exact fallback, top-1, and top-5 metrics for predictions."""

    if (
        isinstance(records, (str, bytes, Mapping))
        or not isinstance(records, Iterable)
    ):
        raise ValueError("records 必须是评测记录数组")
    record_list = list(records)
    if not record_list:
        raise ValueError("评测至少一条记录")

    fallback_correct = 0
    top1_correct = 0
    top5_correct = 0
    for index, record in enumerate(record_list, start=1):
        if not isinstance(record, Mapping):
            raise ValueError(f"第 {index} 条评测记录必须是对象")
        truth = _record_character(record, "truth")
        fallback = _record_character(record, "fallback", allow_empty=True)
        if "candidates" not in record:
            raise ValueError("评测记录缺少必填字段：candidates")
        candidates = _ordered_candidate_characters(record["candidates"])

        fallback_correct += int(fallback == truth)
        top1_correct += int(bool(candidates) and candidates[0] == truth)
        top5_correct += int(truth in candidates[:5])

    count = len(record_list)
    fallback_accuracy = fallback_correct / count
    top1_accuracy = top1_correct / count
    top5_recall = top5_correct / count
    return {
        "count": count,
        "fallback_top1_count": fallback_correct,
        "fallback_top1_correct_count": fallback_correct,
        "fallback_top1_accuracy": fallback_accuracy,
        "top1_count": top1_correct,
        "top1_correct_count": top1_correct,
        "top1_accuracy": top1_accuracy,
        "top5_count": top5_correct,
        "top5_correct_count": top5_correct,
        "top5_recall": top5_recall,
        "candidate_recall": top5_recall,
        "wrong_top1_count": count - top1_correct,
        "fallback_wrong_count": count - fallback_correct,
        "top1_gain": top1_accuracy - fallback_accuracy,
        "top5_gain_vs_fallback": top5_recall - fallback_accuracy,
    }


def regression_gate(metrics):
    """Require a strict top-5 gain without increasing top-1 errors."""

    top5_recall = float(metrics["top5_recall"])
    fallback_accuracy = float(metrics["fallback_top1_accuracy"])
    wrong_top1 = int(metrics["wrong_top1_count"])
    fallback_wrong = int(metrics["fallback_wrong_count"])

    recall_improved = top5_recall > fallback_accuracy
    wrong_not_increased = wrong_top1 <= fallback_wrong
    if recall_improved and wrong_not_increased:
        reason = (
            "passed: strict top5 recall improvement and wrong_top1_count "
            "did not increase"
        )
    else:
        failures = []
        if not recall_improved:
            failures.append(
                "top5 recall requires a strict improvement over fallback "
                "top1 accuracy"
            )
        if not wrong_not_increased:
            failures.append(
                "wrong_top1_count exceeds fallback_wrong_count"
            )
        reason = "failed: " + "; ".join(failures)
    return {"passed": recall_improved and wrong_not_increased, "reason": reason}


def _load_jsonl_records(input_path):
    path = Path(input_path).expanduser().resolve(strict=True)
    if not path.is_file():
        raise ValueError(f"--input 必须是 JSONL 文件：{path}")

    records = []
    with path.open(encoding="utf-8-sig") as input_file:
        for line_number, raw_line in enumerate(input_file, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"JSONL 第 {line_number} 行不是有效 JSON：{error.msg}"
                ) from error
            if not isinstance(record, Mapping):
                raise ValueError(f"JSONL 第 {line_number} 行必须是对象")

            crop_value = record.get("crop")
            if not isinstance(crop_value, str) or not crop_value.strip():
                raise ValueError(
                    f"JSONL 第 {line_number} 行 crop 必须是非空路径"
                )
            crop = Path(crop_value).expanduser()
            if not crop.is_absolute():
                crop = path.parent / crop
            try:
                crop = crop.resolve(strict=True)
            except FileNotFoundError:
                raise FileNotFoundError(
                    f"JSONL 第 {line_number} 行 crop 不存在：{crop_value}"
                ) from None
            if not crop.is_file():
                raise ValueError(
                    f"JSONL 第 {line_number} 行 crop 不是文件：{crop}"
                )

            normalized = {
                "crop": str(crop),
                "truth": _record_character(record, "truth"),
                "fallback": _record_character(
                    {**record, "fallback": record.get("fallback", "")},
                    "fallback",
                    allow_empty=True,
                ),
            }
            records.append(normalized)
    if not records:
        raise ValueError("JSONL 评测集至少一条记录")
    return path, records


def _default_reviewer_factory(asset_dir):
    from rare_glyph_review import OnnxGlyphReviewer

    return OnnxGlyphReviewer(str(asset_dir))


def _safe_manifest_asset_path(asset_dir, relative_value, label):
    if not isinstance(relative_value, str) or not relative_value:
        raise ValueError(f"manifest 缺少资产字段：{label}")
    relative_path = Path(relative_value)
    if relative_path.is_absolute():
        raise ValueError(f"manifest {label} 必须是相对路径")
    resolved = (asset_dir / relative_path).resolve(strict=True)
    try:
        resolved.relative_to(asset_dir)
    except ValueError:
        raise ValueError(f"manifest {label} 资产路径越界") from None
    if not resolved.is_file():
        raise ValueError(f"manifest {label} 资产不是文件：{resolved}")
    return resolved


def _asset_sizes(asset_dir):
    root = Path(asset_dir).expanduser().resolve(strict=True)
    if not root.is_dir():
        raise ValueError(f"--asset-dir 必须是目录：{root}")
    manifest_path = root / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"缺少模型清单：{manifest_path}")
    with manifest_path.open(encoding="utf-8") as manifest_file:
        manifest = json.load(manifest_file)
    if not isinstance(manifest, Mapping):
        raise ValueError("manifest.json 必须是对象")

    paths = {
        field: _safe_manifest_asset_path(root, manifest.get(field), field)
        for field in _MANIFEST_ASSET_FIELDS
    }
    sizes = {field: path.stat().st_size for field, path in paths.items()}
    return root, {
        "model_bytes": sizes["model"],
        "characters_bytes": sizes["characters"],
        "prototype_bytes": sizes["prototypes"],
        "asset_total_bytes": sum(sizes.values()),
    }


def _percentile(samples, percentile):
    ordered = sorted(float(sample) for sample in samples)
    if not ordered:
        return None
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * percentile
    lower_index = int(position)
    upper_index = min(lower_index + 1, len(ordered) - 1)
    fraction = position - lower_index
    return ordered[lower_index] + (
        ordered[upper_index] - ordered[lower_index]
    ) * fraction


def peak_rss_bytes(raw_peak_rss, *, platform_name=None):
    """Convert ``resource.ru_maxrss`` to bytes on supported Unix platforms."""

    if raw_peak_rss is None:
        return None
    system = (platform_name or platform.system()).lower()
    value = int(raw_peak_rss)
    if system == "darwin":
        return value
    if system.startswith("linux"):
        return value * 1024
    return value


def _current_peak_rss_bytes():
    try:
        import resource
    except ImportError:
        return None
    try:
        usage = resource.getrusage(resource.RUSAGE_SELF)
    except (AttributeError, OSError):
        return None
    return peak_rss_bytes(usage.ru_maxrss)


def _elapsed_seconds(start, end):
    return round(max(0.0, float(end) - float(start)), 12)


def _prime_reviewer(reviewer):
    load_hook = getattr(reviewer, "ensure_loaded", None)
    if not callable(load_hook):
        load_hook = getattr(reviewer, "_ensure_session", None)
    if callable(load_hook):
        load_hook()


def run_evaluation(
    *,
    input_path,
    asset_dir,
    batch_size=16,
    top_k=5,
    reviewer_factory=None,
    clock=time.perf_counter,
    peak_rss_provider=_current_peak_rss_bytes,
):
    """Review all JSONL crops and return a complete auditable report."""

    if not isinstance(batch_size, int) or isinstance(batch_size, bool):
        raise ValueError("batch_size 必须是正整数")
    if batch_size <= 0:
        raise ValueError("batch_size 必须是正整数")
    if not isinstance(top_k, int) or isinstance(top_k, bool) or top_k < 5:
        raise ValueError("top_k 必须是至少为 5 的整数")

    resolved_input, records = _load_jsonl_records(input_path)
    resolved_asset_dir, asset_sizes = _asset_sizes(asset_dir)
    factory = reviewer_factory or _default_reviewer_factory

    load_started = clock()
    reviewer = factory(str(resolved_asset_dir))
    _prime_reviewer(reviewer)
    model_load_seconds = _elapsed_seconds(load_started, clock())

    predictions = []
    batch_timings = []
    per_crop_samples = []
    for start in range(0, len(records), batch_size):
        batch_records = records[start : start + batch_size]
        crops = []
        batch_started = clock()
        try:
            for record in batch_records:
                with Image.open(record["crop"]) as image:
                    crops.append(image.convert("RGB"))
            batch_candidates = reviewer.review_batch(crops, top_k=top_k)
        finally:
            for crop in crops:
                crop.close()
        elapsed = _elapsed_seconds(batch_started, clock())
        batch_candidates = list(batch_candidates)
        if len(batch_candidates) != len(batch_records):
            raise ValueError(
                "review_batch 结果数量与输入 crop 数量不一致："
                f"{len(batch_candidates)} != {len(batch_records)}"
            )

        per_crop = round(elapsed / len(batch_records), 12)
        per_crop_samples.append(per_crop)
        batch_timings.append(
            {
                "batch_index": len(batch_timings) + 1,
                "count": len(batch_records),
                "elapsed_seconds": elapsed,
                "per_crop_seconds": per_crop,
            }
        )
        for record, candidates in zip(batch_records, batch_candidates):
            predictions.append(
                {
                    "crop": record["crop"],
                    "truth": record["truth"],
                    "fallback": record["fallback"],
                    "candidates": list(candidates),
                }
            )

    metrics = evaluate_predictions(predictions)
    reviewer_metadata = getattr(reviewer, "metadata", {})
    if not isinstance(reviewer_metadata, Mapping):
        reviewer_metadata = {}
    batch_seconds = [
        timing["elapsed_seconds"] for timing in batch_timings
    ]
    report = {
        "generated_at_utc": datetime.now(timezone.utc)
        .isoformat()
        .replace("+00:00", "Z"),
        "input": str(resolved_input),
        "reviewer": dict(reviewer_metadata),
        "model_load_seconds": model_load_seconds,
        "batch_timings": batch_timings,
        "batch_p50_seconds": round(_percentile(batch_seconds, 0.50), 12),
        "batch_p95_seconds": round(_percentile(batch_seconds, 0.95), 12),
        "per_crop_seconds": {
            "samples": per_crop_samples,
            "p50": round(_percentile(per_crop_samples, 0.50), 12),
            "p95": round(_percentile(per_crop_samples, 0.95), 12),
        },
        "total_review_seconds": round(
            sum(timing["elapsed_seconds"] for timing in batch_timings),
            12,
        ),
        **asset_sizes,
        "peak_rss_bytes": peak_rss_provider(),
        "metrics": metrics,
        "gate": regression_gate(metrics),
        "predictions": predictions,
    }
    return report


def atomic_write_json(output_path, value, *, overwrite=False):
    """Write UTF-8 JSON through a same-directory fsynced temporary file."""

    destination = Path(output_path).expanduser()
    if not destination.is_absolute():
        destination = Path.cwd() / destination
    destination = destination.resolve(strict=False)
    destination.parent.mkdir(parents=True, exist_ok=True)

    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        suffix=".tmp",
        dir=destination.parent,
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as output_file:
            json.dump(value, output_file, ensure_ascii=False, indent=2)
            output_file.write("\n")
            output_file.flush()
            os.fsync(output_file.fileno())
        if overwrite:
            os.replace(temporary_path, destination)
        else:
            os.link(temporary_path, destination)
            temporary_path.unlink()
        if os.name != "nt":
            directory_fd = os.open(destination.parent, os.O_RDONLY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
    except BaseException:
        try:
            temporary_path.unlink()
        except FileNotFoundError:
            pass
        raise


def build_argument_parser():
    parser = argparse.ArgumentParser(
        description="评测 MegaHan 单字 reviewer 的 Top-1/Top-5 表现",
    )
    parser.add_argument("--input", required=True, help="评测 JSONL 文件")
    parser.add_argument(
        "--asset-dir",
        required=True,
        help="OnnxGlyphReviewer 资产目录",
    )
    parser.add_argument("--output", required=True, help="输出 JSON 报告")
    parser.add_argument(
        "--force",
        action="store_true",
        help="允许原子替换已存在的输出报告",
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument(
        "--require-improvement",
        action="store_true",
        help="regression gate 未通过时返回状态码 2",
    )
    return parser


def _parse_cli_args(parser, argv=None):
    arguments = parser.parse_args(argv)
    if arguments.batch_size <= 0:
        raise ValueError("--batch-size 必须是正整数")
    if arguments.top_k < 5:
        raise ValueError("--top-k 必须至少为 5，以便计算 Top-5")
    return arguments


def main(argv=None):
    try:
        arguments = _parse_cli_args(build_argument_parser(), argv)
        report = run_evaluation(
            input_path=arguments.input,
            asset_dir=arguments.asset_dir,
            batch_size=arguments.batch_size,
            top_k=arguments.top_k,
        )
        atomic_write_json(
            arguments.output,
            report,
            overwrite=arguments.force,
        )
        if arguments.require_improvement and not report["gate"]["passed"]:
            logger.error("MegaHan regression gate failed: %s", report["gate"])
            return 2
        logger.info("MegaHan evaluation report written to %s", arguments.output)
        return 0
    except Exception:
        logger.exception("MegaHan evaluation failed")
        return 1


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    raise SystemExit(main())
