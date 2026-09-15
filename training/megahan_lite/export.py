# -*- coding: utf-8 -*-
"""Export a trained MegaHan-lite checkpoint as reviewer assets."""

import argparse
import errno
import json
import logging
import os
from pathlib import Path
import shutil
import sys
import tempfile

import numpy as np
import onnxruntime as ort
import torch
from torch.utils.data import DataLoader
from torchvision import transforms

if os.name == "nt":
    import msvcrt
else:
    import fcntl


if __package__ in (None, ""):
    REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
    if str(REPOSITORY_ROOT) not in sys.path:
        sys.path.insert(0, str(REPOSITORY_ROOT))
    from atomic_publish import atomic_rename_no_replace  # noqa: E402
    from glyph_preprocess import PadGlyphImage  # noqa: E402
    from megahan_assets import build_reviewer_manifest  # noqa: E402
    from training.megahan_lite import (  # noqa: E402
        ARCHITECTURE,
        EMBEDDING_SIZE,
        INPUT_SIZE,
    )
    from training.megahan_lite.dataset import (  # noqa: E402
        build_megahan_dataset,
    )
    from training.megahan_lite.model import (  # noqa: E402
        MegaHanLiteEncoder,
    )
    from training.megahan_lite.train import (  # noqa: E402
        _load_prepared_metadata,
        _sha256_file,
    )
else:
    from atomic_publish import atomic_rename_no_replace
    from glyph_preprocess import PadGlyphImage
    from megahan_assets import build_reviewer_manifest
    from . import ARCHITECTURE, EMBEDDING_SIZE, INPUT_SIZE
    from .dataset import build_megahan_dataset
    from .model import MegaHanLiteEncoder
    from .train import _load_prepared_metadata, _sha256_file


logger = logging.getLogger(__name__)
MODEL_ID = "megahan-lite-v1"
DEFAULT_MAX_MODEL_SIZE_MIB = 15.0
MAX_ONNX_ABS_ERROR = 1e-4
MIN_ONNX_COSINE = 0.9999


def _positive_int(value):
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def _non_negative_int(value):
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("value must be non-negative")
    return parsed


def _positive_float(value):
    parsed = float(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def build_argument_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--batch-size", type=_positive_int, default=256)
    parser.add_argument("--workers", type=_non_negative_int, default=0)
    parser.add_argument("--prototype-version", required=True)
    parser.add_argument(
        "--skip-lmdb-hash-verification",
        action="store_true",
        help=(
            "skip streaming data.mdb SHA-256 verification; "
            "existence and size are still checked"
        ),
    )
    parser.add_argument(
        "--max-model-size-mib",
        type=_positive_float,
        default=DEFAULT_MAX_MODEL_SIZE_MIB,
    )
    return parser


def _load_checkpoint(checkpoint_path, metadata):
    path = Path(checkpoint_path).expanduser().resolve(strict=False)
    if not path.is_file():
        raise FileNotFoundError(f"checkpoint does not exist: {path}")
    checkpoint_sha256 = _sha256_file(path)
    checkpoint = torch.load(
        path,
        map_location="cpu",
        weights_only=False,
    )
    if not isinstance(checkpoint, dict):
        raise ValueError("checkpoint must contain a mapping")

    expected = {
        "architecture": ARCHITECTURE,
        "input_size": INPUT_SIZE,
        "embedding_size": EMBEDDING_SIZE,
        "character_count": metadata["character_count"],
        "codebook_sha256": metadata["codebook_sha256"],
        "characters_sha256": metadata["characters_sha256"],
        "dataset_fingerprint": metadata["dataset_fingerprint"],
    }
    for key, expected_value in expected.items():
        if checkpoint.get(key) != expected_value:
            raise ValueError(
                f"checkpoint {key} mismatch: "
                f"{checkpoint.get(key)!r} != {expected_value!r}"
            )
    if "model_state" not in checkpoint:
        raise ValueError("checkpoint is missing model_state")

    model = MegaHanLiteEncoder(embedding_size=EMBEDDING_SIZE)
    model.load_state_dict(checkpoint["model_state"], strict=True)
    model.eval()
    return model, checkpoint_sha256


def _export_onnx(model, model_path):
    model.eval()
    example = torch.zeros(
        (1, 3, INPUT_SIZE, INPUT_SIZE),
        dtype=torch.float32,
    )
    torch.onnx.export(
        model,
        example,
        model_path,
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        dynamo=False,
        external_data=False,
        input_names=["images"],
        output_names=["embeddings"],
        dynamic_axes={
            "images": {0: "batch"},
            "embeddings": {0: "batch"},
        },
    )


def _reject_onnx_sidecars(model_path):
    unexpected = sorted(
        path
        for path in model_path.parent.glob(f"{model_path.name}*")
        if path != model_path
    )
    if unexpected:
        names = ", ".join(path.name for path in unexpected)
        raise ValueError(
            f"ONNX export created untracked sidecar assets: {names}"
        )


def _validate_onnx(model, model_path):
    session = ort.InferenceSession(
        str(model_path),
        providers=["CPUExecutionProvider"],
    )
    inputs = session.get_inputs()
    outputs = session.get_outputs()
    if len(inputs) != 1 or inputs[0].name != "images":
        raise ValueError("ONNX model must expose exactly one input named images")
    if len(outputs) != 1 or outputs[0].name != "embeddings":
        raise ValueError(
            "ONNX model must expose exactly one output named embeddings"
        )

    generator = np.random.default_rng(2026)
    fixed_random_tensors = (
        generator.standard_normal(
            (1, 3, INPUT_SIZE, INPUT_SIZE)
        ).astype(np.float32),
        generator.standard_normal(
            (2, 3, INPUT_SIZE, INPUT_SIZE)
        ).astype(np.float32),
    )
    model.eval()
    for tensor in fixed_random_tensors:
        with torch.no_grad():
            torch_output = (
                model(torch.from_numpy(tensor))
                .detach()
                .cpu()
                .numpy()
                .astype(np.float32, copy=False)
            )
        ort_output = np.asarray(
            session.run(["embeddings"], {"images": tensor})[0],
            dtype=np.float32,
        )
        expected_shape = (tensor.shape[0], EMBEDDING_SIZE)
        if torch_output.shape != expected_shape:
            raise ValueError(
                f"PyTorch output shape mismatch: "
                f"{torch_output.shape} != {expected_shape}"
            )
        if ort_output.shape != expected_shape:
            raise ValueError(
                f"ONNX output shape mismatch: "
                f"{ort_output.shape} != {expected_shape}"
            )
        if (
            not np.isfinite(torch_output).all()
            or not np.isfinite(ort_output).all()
        ):
            raise ValueError("PyTorch/ONNX verification output is non-finite")

        torch_norms = np.linalg.norm(torch_output, axis=1)
        ort_norms = np.linalg.norm(ort_output, axis=1)
        if not np.allclose(torch_norms, 1.0, atol=1e-5, rtol=1e-5):
            raise ValueError("PyTorch verification output is not L2 normalized")
        if not np.allclose(ort_norms, 1.0, atol=1e-5, rtol=1e-5):
            raise ValueError("ONNX verification output is not L2 normalized")

        max_abs = float(np.max(np.abs(torch_output - ort_output)))
        cosine = np.sum(torch_output * ort_output, axis=1) / (
            torch_norms * ort_norms
        )
        if max_abs > MAX_ONNX_ABS_ERROR:
            raise ValueError(
                f"ONNX max absolute error is too large: {max_abs}"
            )
        if np.any(cosine < MIN_ONNX_COSINE):
            raise ValueError(
                f"ONNX cosine agreement is too low: {cosine.min()}"
            )


def _deterministic_transform():
    return transforms.Compose(
        [
            PadGlyphImage(INPUT_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.5, 0.5, 0.5),
                std=(0.5, 0.5, 0.5),
            ),
        ]
    )


def _build_prototypes(model, metadata, batch_size, workers):
    character_count = metadata["character_count"]
    dataset = build_megahan_dataset(
        roots=metadata["roots"],
        transform=_deterministic_transform(),
        character_count=character_count,
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=False,
        persistent_workers=False,
    )
    sums = np.zeros(
        (character_count, EMBEDDING_SIZE),
        dtype=np.float32,
    )
    counts = np.zeros(character_count, dtype=np.int64)
    model.eval()
    with torch.no_grad():
        for images, labels in loader:
            embeddings = (
                model(images)
                .detach()
                .cpu()
                .numpy()
                .astype(np.float32, copy=False)
            )
            labels = np.asarray(
                labels.detach().cpu().numpy(),
                dtype=np.int64,
            )
            if labels.ndim != 1 or labels.shape[0] != embeddings.shape[0]:
                raise ValueError("prototype batch labels have invalid shape")
            if np.any(labels < 0) or np.any(labels >= character_count):
                raise ValueError(
                    "prototype label is outside the character codebook"
                )
            if (
                embeddings.ndim != 2
                or embeddings.shape[1] != EMBEDDING_SIZE
                or not np.isfinite(embeddings).all()
            ):
                raise ValueError("prototype batch embeddings are invalid")
            np.add.at(sums, labels, embeddings)
            np.add.at(counts, labels, 1)

    missing = np.flatnonzero(counts == 0)
    if missing.size:
        preview = ", ".join(str(int(label)) for label in missing[:10])
        raise ValueError(
            f"{missing.size} character classes have no approved samples: "
            f"{preview}"
        )
    prototypes = sums / counts[:, None].astype(np.float32)
    norms = np.linalg.norm(prototypes, axis=1)
    if not np.isfinite(prototypes).all() or not np.isfinite(norms).all():
        raise ValueError("prototype means contain non-finite values")
    if np.any(norms <= 1e-12):
        raise ValueError("prototype means contain zero-norm rows")
    prototypes = prototypes / norms[:, None]
    if not np.isfinite(prototypes).all():
        raise ValueError("normalized prototypes contain non-finite values")
    return prototypes.astype(np.float16)


def _write_bytes_fsynced(path, content):
    with path.open("xb") as output_file:
        output_file.write(content)
        output_file.flush()
        os.fsync(output_file.fileno())


def _write_text_fsynced(path, content):
    _write_bytes_fsynced(path, content.encode("utf-8"))


def _fsync_file(path):
    with path.open("rb") as asset_file:
        os.fsync(asset_file.fileno())


def _fsync_directory(path):
    try:
        directory_fd = os.open(str(path), os.O_RDONLY)
    except OSError as error:
        logger.debug("Could not open directory for fsync: %s", error)
        return
    try:
        os.fsync(directory_fd)
    except OSError as error:
        logger.debug("Could not fsync directory: %s", error)
    finally:
        os.close(directory_fd)


def _acquire_output_lock(lock_path):
    lock_fd = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o600)
    try:
        if os.fstat(lock_fd).st_size == 0:
            os.write(lock_fd, b"\0")
            os.fsync(lock_fd)
        os.lseek(lock_fd, 0, os.SEEK_SET)
        try:
            if os.name == "nt":
                msvcrt.locking(lock_fd, msvcrt.LK_NBLCK, 1)
            else:
                fcntl.flock(
                    lock_fd,
                    fcntl.LOCK_EX | fcntl.LOCK_NB,
                )
        except OSError as error:
            if error.errno not in (errno.EACCES, errno.EAGAIN):
                raise
            raise RuntimeError(
                f"another export is using the output lock: {lock_path}"
            ) from None
    except BaseException:
        os.close(lock_fd)
        raise
    return lock_fd


def _release_output_lock(lock_fd):
    try:
        if os.name == "nt":
            os.lseek(lock_fd, 0, os.SEEK_SET)
            msvcrt.locking(lock_fd, msvcrt.LK_UNLCK, 1)
        else:
            fcntl.flock(lock_fd, fcntl.LOCK_UN)
    finally:
        os.close(lock_fd)


def export_reviewer_assets(
    *,
    prepared,
    checkpoint,
    output,
    batch_size,
    workers,
    prototype_version,
    skip_lmdb_hash_verification=False,
    max_model_size_mib=DEFAULT_MAX_MODEL_SIZE_MIB,
):
    output_path = Path(output).expanduser().resolve(strict=False)
    if os.path.lexists(output_path):
        raise FileExistsError(
            f"output already exists and will not be overwritten: {output_path}"
        )
    metadata = _load_prepared_metadata(
        prepared,
        skip_lmdb_hash_verification=skip_lmdb_hash_verification,
    )
    model, checkpoint_sha256 = _load_checkpoint(checkpoint, metadata)
    characters_path = (
        Path(prepared).expanduser().resolve(strict=False) / "characters.txt"
    )
    characters = characters_path.read_text(encoding="utf-8").splitlines()
    characters_content = "".join(f"{character}\n" for character in characters)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = output_path.parent / f".{output_path.name}.megahan.lock"
    lock_fd = _acquire_output_lock(lock_path)
    staging_path = None
    published = False
    try:
        if os.path.lexists(output_path):
            raise FileExistsError(
                f"output appeared while waiting for lock: {output_path}"
            )
        staging_path = Path(
            tempfile.mkdtemp(
                prefix=f".{output_path.name}.tmp-",
                dir=output_path.parent,
            )
        )

        model_path = staging_path / "glyph_encoder.onnx"
        _export_onnx(model, model_path)
        _reject_onnx_sidecars(model_path)
        _fsync_file(model_path)
        model_size_limit = int(max_model_size_mib * 1024 * 1024)
        model_size = model_path.stat().st_size
        if model_size <= 0 or model_size > model_size_limit:
            raise ValueError(
                f"ONNX model size {model_size} bytes exceeds "
                f"limit {model_size_limit} bytes"
            )
        _validate_onnx(model, model_path)

        prototypes = _build_prototypes(
            model,
            metadata,
            batch_size=batch_size,
            workers=workers,
        )
        prototypes_path = staging_path / "prototypes.npy"
        np.save(prototypes_path, prototypes, allow_pickle=False)
        _fsync_file(prototypes_path)

        exported_characters_path = staging_path / "characters.txt"
        _write_text_fsynced(
            exported_characters_path,
            characters_content,
        )
        manifest = build_reviewer_manifest(
            asset_dir=staging_path,
            model_id=MODEL_ID,
            prototype_version=prototype_version,
            character_count=metadata["character_count"],
            input_size=INPUT_SIZE,
            embedding_size=EMBEDDING_SIZE,
            architecture=ARCHITECTURE,
            codebook_sha256=metadata["codebook_sha256"],
            characters_sha256=metadata["characters_sha256"],
            dataset_fingerprint=metadata["dataset_fingerprint"],
            checkpoint_sha256=checkpoint_sha256,
        )
        manifest_text = json.dumps(
            manifest,
            ensure_ascii=False,
            indent=2,
        ) + "\n"
        _write_text_fsynced(
            staging_path / "manifest.json",
            manifest_text,
        )
        _fsync_directory(staging_path)

        if os.path.lexists(output_path):
            raise FileExistsError(
                f"output appeared during export: {output_path}"
            )
        atomic_rename_no_replace(staging_path, output_path)
        published = True
        _fsync_directory(output_path.parent)
    finally:
        try:
            if (
                not published
                and staging_path is not None
                and staging_path.exists()
            ):
                shutil.rmtree(staging_path)
        finally:
            _release_output_lock(lock_fd)
    return manifest


def main(argv=None):
    arguments = build_argument_parser().parse_args(argv)
    if arguments.skip_lmdb_hash_verification:
        logger.warning(
            "LMDB SHA-256 verification was explicitly skipped"
        )
    export_reviewer_assets(
        prepared=arguments.prepared,
        checkpoint=arguments.checkpoint,
        output=arguments.output,
        batch_size=arguments.batch_size,
        workers=arguments.workers,
        prototype_version=arguments.prototype_version,
        skip_lmdb_hash_verification=(
            arguments.skip_lmdb_hash_verification
        ),
        max_model_size_mib=arguments.max_model_size_mib,
    )
    logger.info("MegaHan reviewer assets exported: %s", arguments.output)
    return 0


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )
    main()
