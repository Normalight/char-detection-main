# -*- coding: utf-8 -*-
"""Train the lightweight MegaHan glyph encoder on approved local LMDBs."""

import argparse
import hashlib
import json
import logging
import os
from pathlib import Path
import random
import sys
import tempfile

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms


if __package__ in (None, ""):
    REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
    if str(REPOSITORY_ROOT) not in sys.path:
        sys.path.insert(0, str(REPOSITORY_ROOT))
    from glyph_preprocess import PadGlyphImage  # noqa: E402
    from training.megahan_lite import (  # noqa: E402
        ARCHITECTURE,
        EMBEDDING_SIZE,
        INPUT_SIZE,
    )
    from training.megahan_lite.dataset import (  # noqa: E402
        ClassBalancedBatchSampler,
        build_megahan_dataset,
    )
    from training.megahan_lite.model import (  # noqa: E402
        MegaHanLiteEncoder,
        supervised_contrastive_loss,
    )
else:
    from glyph_preprocess import PadGlyphImage
    from . import ARCHITECTURE, EMBEDDING_SIZE, INPUT_SIZE
    from .dataset import (
        ClassBalancedBatchSampler,
        build_megahan_dataset,
    )
    from .model import MegaHanLiteEncoder, supervised_contrastive_loss


logger = logging.getLogger(__name__)
SELECTION_METRIC = "augmented_training_loss"


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
    parser.add_argument(
        "--prepared",
        required=True,
        help="Task 2 prepared metadata directory",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="checkpoint output directory",
    )
    parser.add_argument("--epochs", type=_positive_int, default=30)
    parser.add_argument(
        "--classes-per-batch",
        type=_positive_int,
        default=32,
    )
    parser.add_argument(
        "--samples-per-class",
        type=_positive_int,
        default=2,
    )
    parser.add_argument("--lr", type=_positive_float, default=1e-3)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument(
        "--workers",
        type=_non_negative_int,
        default=0,
    )
    parser.add_argument("--resume")
    parser.add_argument(
        "--skip-lmdb-hash-verification",
        action="store_true",
        help=(
            "skip streaming data.mdb SHA-256 verification; "
            "existence and size are still checked"
        ),
    )
    return parser


def _read_json(path, description):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise FileNotFoundError(
            f"prepared {description} is missing: {path}"
        ) from None
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(
            f"prepared {description} is invalid: {path}: {error}"
        ) from error


def _sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as input_file:
        for block in iter(lambda: input_file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _validate_sha256(value, description):
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(
            character not in "0123456789abcdefABCDEF"
            for character in value
        )
    ):
        raise ValueError(
            f"{description} must be a SHA-256 hex digest"
        )
    return value.lower()


def _dataset_fingerprint(characters_sha256, data_records):
    identity = {
        "characters_sha256": characters_sha256.lower(),
        "data_mdb": [
            {
                "sha256": record["sha256"].lower(),
                "size": record["size"],
            }
            for record in data_records
        ],
    }
    serialized = json.dumps(
        identity,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(serialized).hexdigest()


def _verify_lmdb_sources(sources, skip_hash_verification=False):
    if not isinstance(sources, list):
        raise ValueError("prepared lmdb_sources.json must be a list")
    if not sources:
        raise ValueError(
            "prepared metadata contains no approved MegaHan LMDB roots"
        )

    roots = []
    data_records = []
    for source_index, source in enumerate(sources):
        if not isinstance(source, dict):
            raise ValueError(
                f"LMDB source {source_index} must be an object"
            )
        root_value = source.get("root")
        if not isinstance(root_value, str) or not root_value:
            raise ValueError(
                f"LMDB source {source_index} has no valid root"
            )
        root = Path(root_value).expanduser().resolve(strict=False)
        if not root.is_dir():
            raise ValueError(
                f"LMDB source {source_index} root is not a directory: "
                f"{root}"
            )
        data_path = root / "data.mdb"
        if not data_path.is_file():
            raise FileNotFoundError(
                f"LMDB source {source_index} is missing data.mdb: "
                f"{data_path}"
            )

        data_metadata = source.get("data_mdb")
        if not isinstance(data_metadata, dict):
            raise ValueError(
                f"LMDB source {source_index} has no data_mdb metadata"
            )
        expected_size = data_metadata.get("size")
        if (
            not isinstance(expected_size, int)
            or isinstance(expected_size, bool)
            or expected_size < 0
        ):
            raise ValueError(
                f"LMDB source {source_index} data.mdb size is invalid"
            )
        expected_sha256 = _validate_sha256(
            data_metadata.get("sha256"),
            f"LMDB source {source_index} data.mdb SHA-256",
        )
        actual_size = data_path.stat().st_size
        if actual_size != expected_size:
            raise ValueError(
                f"LMDB source {source_index} data.mdb size mismatch: "
                f"{actual_size} != {expected_size}"
            )
        if not skip_hash_verification:
            actual_sha256 = _sha256_file(data_path)
            if actual_sha256 != expected_sha256:
                raise ValueError(
                    f"LMDB source {source_index} data.mdb SHA-256 "
                    f"mismatch: {actual_sha256} != {expected_sha256}"
                )

        roots.append(str(root))
        data_records.append(
            {
                "size": expected_size,
                "sha256": expected_sha256,
            }
        )
    return roots, data_records


def _load_prepared_metadata(
    prepared,
    skip_lmdb_hash_verification=False,
):
    prepared_path = Path(prepared).expanduser().resolve(strict=False)
    if not prepared_path.is_dir():
        raise ValueError(
            f"prepared path is not a directory: {prepared_path}"
        )

    characters_path = prepared_path / "characters.txt"
    try:
        characters = characters_path.read_text(
            encoding="utf-8"
        ).splitlines()
    except FileNotFoundError:
        raise FileNotFoundError(
            f"prepared characters are missing: {characters_path}"
        ) from None
    except (OSError, UnicodeError) as error:
        raise ValueError(
            f"prepared characters are invalid: {characters_path}: {error}"
        ) from error

    if not characters:
        raise ValueError("prepared characters.txt must not be empty")
    seen = set()
    for line_number, character in enumerate(characters, start=1):
        if len(character) != 1:
            raise ValueError(
                f"characters.txt line {line_number} "
                "must contain one Unicode character"
            )
        if character in seen:
            raise ValueError(
                f"characters.txt line {line_number} is duplicated"
            )
        seen.add(character)

    provenance = _read_json(
        prepared_path / "provenance.json",
        "provenance",
    )
    if not isinstance(provenance, dict):
        raise ValueError("prepared provenance.json must be an object")
    character_count = provenance.get("character_count")
    if character_count != len(characters):
        raise ValueError(
            "prepared provenance character_count does not match "
            f"characters.txt: {character_count!r} != {len(characters)}"
        )
    codebook_sha256 = _validate_sha256(
        provenance.get("codebook_sha256"),
        "prepared provenance codebook_sha256",
    )
    characters_sha256 = _validate_sha256(
        provenance.get("characters_sha256"),
        "prepared provenance characters_sha256",
    )
    actual_characters_sha256 = _sha256_file(characters_path)
    if actual_characters_sha256 != characters_sha256:
        raise ValueError(
            "prepared provenance characters_sha256 does not match "
            f"characters.txt: {characters_sha256} "
            f"!= {actual_characters_sha256}"
        )

    sources = _read_json(
        prepared_path / "lmdb_sources.json",
        "LMDB sources",
    )
    roots, data_records = _verify_lmdb_sources(
        sources,
        skip_hash_verification=skip_lmdb_hash_verification,
    )

    return {
        "character_count": character_count,
        "codebook_sha256": codebook_sha256,
        "characters_sha256": characters_sha256,
        "dataset_fingerprint": _dataset_fingerprint(
            characters_sha256,
            data_records,
        ),
        "roots": roots,
    }


def _build_transform():
    return transforms.Compose(
        [
            PadGlyphImage(INPUT_SIZE),
            transforms.RandomInvert(p=0.20),
            transforms.RandomAffine(
                degrees=5,
                translate=(0.05, 0.05),
                fill=255,
            ),
            transforms.RandomApply(
                [transforms.GaussianBlur(3, sigma=(0.1, 1.0))],
                p=0.20,
            ),
            transforms.RandomApply(
                [
                    transforms.ColorJitter(
                        brightness=0.15,
                        contrast=0.15,
                        saturation=0.05,
                    )
                ],
                p=0.25,
            ),
            transforms.RandomGrayscale(p=0.10),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.5, 0.5, 0.5),
                std=(0.5, 0.5, 0.5),
            ),
        ]
    )


def _seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def _seed_worker(worker_id):
    del worker_id
    worker_seed = torch.initial_seed() % (2**32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def _capture_rng_state():
    return {
        "python_random_state": random.getstate(),
        "numpy_random_state": np.random.get_state(),
        "torch_cpu_rng_state": torch.get_rng_state(),
        "torch_cuda_rng_state": (
            torch.cuda.get_rng_state_all()
            if torch.cuda.is_available()
            else None
        ),
    }


def _restore_rng_state(checkpoint):
    required = (
        "python_random_state",
        "numpy_random_state",
        "torch_cpu_rng_state",
        "torch_cuda_rng_state",
    )
    missing = [key for key in required if key not in checkpoint]
    if missing:
        raise ValueError(
            "resume checkpoint is missing RNG state: "
            + ", ".join(missing)
        )

    random.setstate(checkpoint["python_random_state"])
    np.random.set_state(checkpoint["numpy_random_state"])
    torch.set_rng_state(checkpoint["torch_cpu_rng_state"])
    cuda_state = checkpoint["torch_cuda_rng_state"]
    if cuda_state is not None:
        if not torch.cuda.is_available():
            raise ValueError(
                "resume checkpoint contains CUDA RNG state "
                "but CUDA is unavailable"
            )
        if len(cuda_state) != torch.cuda.device_count():
            raise ValueError(
                "resume checkpoint CUDA RNG device count mismatch"
            )
        torch.cuda.set_rng_state_all(cuda_state)


def _select_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _use_non_blocking_transfer(device_type):
    return device_type == "cuda"


def _fsync_directory(directory):
    try:
        directory_fd = os.open(str(directory), os.O_RDONLY)
    except OSError as error:
        logger.debug(
            "Could not open checkpoint directory for fsync: %s",
            error,
        )
        return
    try:
        os.fsync(directory_fd)
    except OSError as error:
        logger.debug(
            "Could not fsync checkpoint directory: %s",
            error,
        )
    finally:
        os.close(directory_fd)


def _atomic_save_checkpoint(payload, target):
    target.parent.mkdir(parents=True, exist_ok=True)
    file_descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{target.name}.",
        suffix=".tmp",
        dir=target.parent,
    )
    os.close(file_descriptor)
    temporary_path = Path(temporary_name)
    try:
        torch.save(payload, temporary_path)
        with temporary_path.open("rb") as checkpoint_file:
            os.fsync(checkpoint_file.fileno())
        os.replace(temporary_path, target)
        _fsync_directory(target.parent)
    finally:
        try:
            temporary_path.unlink()
        except FileNotFoundError:
            pass


def _checkpoint_payload(
    model,
    optimizer,
    epoch,
    best_loss,
    metadata,
):
    payload = {
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "epoch": epoch,
        "embedding_size": EMBEDDING_SIZE,
        "input_size": INPUT_SIZE,
        "codebook_sha256": metadata["codebook_sha256"],
        "characters_sha256": metadata["characters_sha256"],
        "dataset_fingerprint": metadata["dataset_fingerprint"],
        "character_count": metadata["character_count"],
        "architecture": ARCHITECTURE,
        "selection_metric": SELECTION_METRIC,
        "best_loss": best_loss,
    }
    payload.update(_capture_rng_state())
    return payload


def _resume_training(resume_path, model, optimizer, metadata, device):
    checkpoint_path = Path(resume_path).expanduser().resolve(strict=False)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"resume checkpoint does not exist: {checkpoint_path}"
        )
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )
    if not isinstance(checkpoint, dict):
        raise ValueError("resume checkpoint must contain a mapping")

    expected = {
        "embedding_size": EMBEDDING_SIZE,
        "input_size": INPUT_SIZE,
        "codebook_sha256": metadata["codebook_sha256"],
        "characters_sha256": metadata["characters_sha256"],
        "dataset_fingerprint": metadata["dataset_fingerprint"],
        "character_count": metadata["character_count"],
        "architecture": ARCHITECTURE,
        "selection_metric": SELECTION_METRIC,
    }
    for key, expected_value in expected.items():
        if checkpoint.get(key) != expected_value:
            raise ValueError(
                f"resume checkpoint {key} mismatch: "
                f"{checkpoint.get(key)!r} != {expected_value!r}"
            )
    if "model_state" not in checkpoint:
        raise ValueError("resume checkpoint is missing model_state")
    if "optimizer_state" not in checkpoint:
        raise ValueError("resume checkpoint is missing optimizer_state")
    epoch = checkpoint.get("epoch")
    if not isinstance(epoch, int) or isinstance(epoch, bool) or epoch < 0:
        raise ValueError("resume checkpoint has invalid epoch")

    model.load_state_dict(checkpoint["model_state"])
    optimizer.load_state_dict(checkpoint["optimizer_state"])
    _restore_rng_state(checkpoint)
    best_loss = float(checkpoint.get("best_loss", float("inf")))
    logger.info(
        "Resumed checkpoint %s after epoch %d",
        checkpoint_path,
        epoch,
    )
    return epoch + 1, best_loss


def train(arguments):
    logger.warning(
        "best.pt uses augmented training loss only and "
        "does not estimate generalization"
    )
    if arguments.skip_lmdb_hash_verification:
        logger.warning(
            "LMDB SHA-256 verification was explicitly skipped; "
            "existence and size will still be verified"
        )
    metadata = _load_prepared_metadata(
        arguments.prepared,
        skip_lmdb_hash_verification=(
            arguments.skip_lmdb_hash_verification
        ),
    )
    if arguments.samples_per_class < 2:
        raise ValueError(
            "samples_per_class must be at least 2 for positive pairs"
        )
    _seed_everything(arguments.seed)
    device = _select_device()
    non_blocking = _use_non_blocking_transfer(device.type)

    dataset = build_megahan_dataset(
        roots=metadata["roots"],
        transform=_build_transform(),
        character_count=metadata["character_count"],
    )
    labels = tuple(
        label
        for child_dataset in dataset.datasets
        for label in child_dataset.labels
    )
    sampler = ClassBalancedBatchSampler(
        labels,
        classes_per_batch=arguments.classes_per_batch,
        samples_per_class=arguments.samples_per_class,
        seed=arguments.seed,
    )
    if len(sampler) == 0:
        raise ValueError(
            "not enough eligible classes for one balanced batch: "
            f"need {arguments.classes_per_batch} classes with at least "
            f"{arguments.samples_per_class} samples each"
        )

    loader = DataLoader(
        dataset,
        batch_sampler=sampler,
        num_workers=arguments.workers,
        pin_memory=device.type == "cuda",
        worker_init_fn=_seed_worker,
        persistent_workers=False,
    )

    model = MegaHanLiteEncoder(
        embedding_size=EMBEDDING_SIZE
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=arguments.lr,
        weight_decay=1e-4,
    )
    start_epoch = 0
    best_loss = float("inf")
    if arguments.resume is not None:
        start_epoch, best_loss = _resume_training(
            arguments.resume,
            model,
            optimizer,
            metadata,
            device,
        )
    if start_epoch >= arguments.epochs:
        raise ValueError(
            f"resume epoch {start_epoch} is not below "
            f"requested epochs {arguments.epochs}"
        )

    output = Path(arguments.output).expanduser().resolve(strict=False)
    output.mkdir(parents=True, exist_ok=True)
    logger.info(
        "Training MegaHan encoder: device=%s samples=%d batches=%d "
        "batch_size=%d",
        device,
        len(dataset),
        len(sampler),
        arguments.classes_per_batch * arguments.samples_per_class,
    )

    for epoch in range(start_epoch, arguments.epochs):
        sampler.set_epoch(epoch)
        model.train()
        total_loss = 0.0
        batch_count = 0
        for images, labels_batch in loader:
            images = images.to(device, non_blocking=non_blocking)
            labels_batch = labels_batch.to(
                device,
                non_blocking=non_blocking,
            )
            optimizer.zero_grad(set_to_none=True)
            embeddings = model(images)
            loss = supervised_contrastive_loss(
                embeddings,
                labels_batch,
            )
            if not bool(torch.isfinite(loss).item()):
                raise RuntimeError(
                    f"non-finite training loss at epoch {epoch + 1}"
                )
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach().cpu())
            batch_count += 1

        if batch_count == 0:
            raise RuntimeError("balanced sampler produced no batches")
        average_loss = total_loss / batch_count
        improved = average_loss < best_loss
        if improved:
            best_loss = average_loss
        payload = _checkpoint_payload(
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            best_loss=best_loss,
            metadata=metadata,
        )
        _atomic_save_checkpoint(payload, output / "last.pt")
        if improved:
            _atomic_save_checkpoint(payload, output / "best.pt")
            logger.info(
                "Saved training-loss best checkpoint for epoch %d",
                epoch + 1,
            )
        logger.info(
            "Epoch %d/%d training_loss=%.6f best_training_loss=%.6f",
            epoch + 1,
            arguments.epochs,
            average_loss,
            best_loss,
        )

    return 0


def main(argv=None):
    arguments = build_argument_parser().parse_args(argv)
    return train(arguments)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )
    main()
