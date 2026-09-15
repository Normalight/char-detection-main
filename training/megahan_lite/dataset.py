# -*- coding: utf-8 -*-
"""Strict MegaHan LMDB loading and balanced metric-learning batches."""

import io
from pathlib import Path
import random
import re

import lmdb
from PIL import Image, UnidentifiedImageError
from torch.utils.data import ConcatDataset, Dataset, Sampler


_INTEGER_PATTERN = re.compile(r"[0-9]+\Z")
_NUM_SAMPLES_KEY = "num-samples"


def _validate_batch_shape(classes_per_batch, samples_per_class):
    if (
        not isinstance(classes_per_batch, int)
        or isinstance(classes_per_batch, bool)
        or classes_per_batch <= 0
    ):
        raise ValueError("classes_per_batch must be a positive integer")
    if (
        not isinstance(samples_per_class, int)
        or isinstance(samples_per_class, bool)
        or samples_per_class < 2
    ):
        raise ValueError("samples_per_class must be at least 2")


def _build_class_index(labels, samples_per_class):
    class_to_indices = {}
    for index, label in enumerate(labels):
        if (
            not isinstance(label, int)
            or isinstance(label, bool)
            or label < 0
        ):
            raise ValueError(
                f"labels[{index}] must be a non-negative integer"
            )
        class_to_indices.setdefault(label, []).append(index)
    return {
        label: tuple(indices)
        for label, indices in class_to_indices.items()
        if len(indices) >= samples_per_class
    }


def _balanced_batch_indices_from_class_index(
    class_to_indices,
    classes_per_batch,
    samples_per_class,
    seed,
    epoch,
):
    eligible_classes = sorted(class_to_indices)
    generator = random.Random(seed + epoch * 1_000_003)
    generator.shuffle(eligible_classes)

    batch_count = len(eligible_classes) // classes_per_batch
    batches = []
    for batch_index in range(batch_count):
        start = batch_index * classes_per_batch
        chosen_classes = eligible_classes[
            start : start + classes_per_batch
        ]
        batch = []
        for label in chosen_classes:
            candidates = list(class_to_indices[label])
            generator.shuffle(candidates)
            batch.extend(candidates[:samples_per_class])
        batches.append(batch)
    return batches


def _balanced_batch_indices(
    labels,
    classes_per_batch,
    samples_per_class,
    seed,
    epoch,
):
    """Build one deterministic, class-balanced set of batches."""

    _validate_batch_shape(classes_per_batch, samples_per_class)
    class_to_indices = _build_class_index(labels, samples_per_class)
    return _balanced_batch_indices_from_class_index(
        class_to_indices,
        classes_per_batch=classes_per_batch,
        samples_per_class=samples_per_class,
        seed=seed,
        epoch=epoch,
    )


class MegaHanLmdbDataset(Dataset):
    """Read a single official MegaHan LMDB without skipping bad records."""

    def __init__(
        self,
        root,
        transform=None,
        character_count=97_455,
    ):
        self.root = Path(root).expanduser().resolve(strict=False)
        self.transform = transform
        if (
            not isinstance(character_count, int)
            or isinstance(character_count, bool)
            or character_count <= 0
        ):
            raise ValueError("character_count must be a positive integer")
        self.character_count = character_count
        self._labels_cache = None

        if not self.root.is_dir():
            raise RuntimeError(
                f"MegaHan LMDB root is not a directory: {self.root}"
            )
        self._environment = self._open_environment()
        try:
            with self._environment.begin(write=False) as transaction:
                raw_count = transaction.get(
                    _NUM_SAMPLES_KEY.encode("ascii")
                )
            self._length = self._parse_sample_count(raw_count)
        except BaseException:
            self.close()
            raise

    def _open_environment(self):
        try:
            return lmdb.open(
                str(self.root),
                readonly=True,
                lock=False,
                readahead=False,
                create=False,
                max_readers=64,
            )
        except Exception as error:
            raise RuntimeError(
                f"failed to open MegaHan LMDB at {self.root}: {error}"
            ) from error

    def _parse_sample_count(self, raw_count):
        if raw_count is None:
            raise RuntimeError(
                f"MegaHan LMDB is missing num-samples: {self.root}"
            )
        try:
            value = bytes(raw_count).decode("ascii")
        except (UnicodeDecodeError, TypeError, ValueError) as error:
            raise RuntimeError(
                f"MegaHan LMDB has invalid num-samples: {self.root}"
            ) from error
        if not _INTEGER_PATTERN.fullmatch(value):
            raise RuntimeError(
                f"MegaHan LMDB has invalid num-samples: {value!r}"
            )
        return int(value)

    @staticmethod
    def _record_keys(index):
        sample_number = index + 1
        return (
            ("image-%09d" % sample_number).encode("ascii"),
            ("label-%09d" % sample_number).encode("ascii"),
        )

    def _validate_index(self, index):
        if not isinstance(index, int) or isinstance(index, bool):
            raise TypeError("MegaHan dataset index must be an integer")
        if not 0 <= index < len(self):
            raise IndexError(
                f"MegaHan dataset index {index} is outside "
                f"[0, {len(self)})"
            )

    def _decode_label(self, raw_label, key):
        if raw_label is None:
            raise RuntimeError(
                f"MegaHan LMDB is missing record {key.decode('ascii')}"
            )
        try:
            text = bytes(raw_label).decode("ascii")
        except (UnicodeDecodeError, TypeError, ValueError) as error:
            raise ValueError(
                f"MegaHan label {key.decode('ascii')} is not an integer"
            ) from error
        if not _INTEGER_PATTERN.fullmatch(text):
            raise ValueError(
                f"MegaHan label {key.decode('ascii')} "
                f"is not an integer: {text!r}"
            )
        label = int(text)
        if not 0 <= label < self.character_count:
            raise ValueError(
                f"MegaHan label {key.decode('ascii')}={label} is outside "
                f"[0, {self.character_count})"
            )
        return label

    def __len__(self):
        return self._length

    def __getitem__(self, index):
        self._validate_index(index)
        image_key, label_key = self._record_keys(index)
        with self._environment.begin(write=False) as transaction:
            raw_image = transaction.get(image_key)
            raw_label = transaction.get(label_key)

        if raw_image is None:
            raise RuntimeError(
                f"MegaHan LMDB is missing record "
                f"{image_key.decode('ascii')}"
            )
        label = self._decode_label(raw_label, label_key)
        try:
            with Image.open(io.BytesIO(bytes(raw_image))) as image:
                rgb_image = image.convert("RGB")
                rgb_image.load()
        except (
            UnidentifiedImageError,
            OSError,
            TypeError,
            ValueError,
        ) as error:
            raise RuntimeError(
                f"MegaHan image {image_key.decode('ascii')} is invalid"
            ) from error

        if self.transform is not None:
            rgb_image = self.transform(rgb_image)
        return rgb_image, label

    def read_labels(self):
        """Read and validate every label in a single LMDB transaction."""

        if self._labels_cache is not None:
            return self._labels_cache

        labels = []
        with self._environment.begin(write=False) as transaction:
            for index in range(len(self)):
                _, label_key = self._record_keys(index)
                labels.append(
                    self._decode_label(
                        transaction.get(label_key),
                        label_key,
                    )
                )
        self._labels_cache = tuple(labels)
        return self._labels_cache

    @property
    def labels(self):
        return self.read_labels()

    def close(self):
        environment = getattr(self, "_environment", None)
        if environment is not None:
            environment.close()
            self._environment = None

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_environment"] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._environment = self._open_environment()

    def __del__(self):
        self.close()


def build_megahan_dataset(
    roots,
    transform=None,
    character_count=97_455,
):
    """Build one concatenated dataset from approved local LMDB roots."""

    root_list = list(roots)
    if not root_list:
        raise ValueError("at least one MegaHan LMDB root is required")
    datasets = [
        MegaHanLmdbDataset(
            root=root,
            transform=transform,
            character_count=character_count,
        )
        for root in root_list
    ]
    return ConcatDataset(datasets)


class ClassBalancedBatchSampler(Sampler):
    """Sample distinct classes and positive pairs in deterministic batches."""

    def __init__(
        self,
        labels,
        classes_per_batch=32,
        samples_per_class=2,
        seed=0,
    ):
        self.labels = tuple(labels)
        self.classes_per_batch = classes_per_batch
        self.samples_per_class = samples_per_class
        self.seed = seed
        self.epoch = 0

        _validate_batch_shape(
            self.classes_per_batch,
            self.samples_per_class,
        )
        self._class_to_indices = _build_class_index(
            self.labels,
            self.samples_per_class,
        )
        self._length = (
            len(self._class_to_indices) // self.classes_per_batch
        )

    def set_epoch(self, epoch):
        if not isinstance(epoch, int) or isinstance(epoch, bool) or epoch < 0:
            raise ValueError("epoch must be a non-negative integer")
        self.epoch = epoch

    def __len__(self):
        """Return eligible class groups per epoch, rounded down."""

        return self._length

    def __iter__(self):
        return iter(
            _balanced_batch_indices_from_class_index(
                self._class_to_indices,
                classes_per_batch=self.classes_per_batch,
                samples_per_class=self.samples_per_class,
                seed=self.seed,
                epoch=self.epoch,
            )
        )
