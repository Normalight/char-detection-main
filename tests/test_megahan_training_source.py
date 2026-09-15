# -*- coding: utf-8 -*-
"""Source-level contracts for the optional MegaHan training package.

The runtime ``chardet`` environment intentionally has neither PyTorch nor LMDB,
so these tests inspect source and execute only dependency-free sampler logic.
"""

import ast
import hashlib
import json
import py_compile
from pathlib import Path
import random

import numpy as np
from PIL import Image, ImageOps
import pytest
import yaml


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
TRAINING_ROOT = REPOSITORY_ROOT / "training" / "megahan_lite"
RUNTIME_REQUIREMENTS = REPOSITORY_ROOT / "requirements.txt"


def _read_source(name):
    return (TRAINING_ROOT / name).read_text(encoding="utf-8")


def _parse_source(name):
    return ast.parse(_read_source(name), filename=name)


def _calls_named(tree, name):
    return [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and (
            isinstance(node.func, ast.Name)
            and node.func.id == name
            or isinstance(node.func, ast.Attribute)
            and node.func.attr == name
        )
    ]


def _keyword_value(call, name):
    for keyword in call.keywords:
        if keyword.arg == name:
            return keyword.value
    return None


def _literal_strings(tree):
    return {
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
    }


def _load_pure_sampler_function():
    tree = _parse_source("dataset.py")
    functions = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and (
            node.name == "_balanced_batch_indices"
            or "class_index" in node.name
            or node.name == "_validate_batch_shape"
        )
    ]
    namespace = {"random": random}
    exec(
        compile(
            ast.Module(body=functions, type_ignores=[]),
            filename="dataset.py",
            mode="exec",
        ),
        namespace,
    )
    return namespace["_balanced_batch_indices"]


def _load_train_integrity_functions():
    required = {
        "_dataset_fingerprint",
        "_load_prepared_metadata",
        "_read_json",
        "_sha256_file",
        "_validate_sha256",
        "_verify_lmdb_sources",
    }
    tree = _parse_source("train.py")
    functions = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name in required
    ]
    found = {node.name for node in functions}
    assert found == required, f"missing integrity helpers: {required - found}"
    namespace = {
        "Path": Path,
        "hashlib": hashlib,
        "json": json,
    }
    exec(
        compile(
            ast.Module(body=functions, type_ignores=[]),
            filename="train.py",
            mode="exec",
        ),
        namespace,
    )
    return namespace


def test_training_files_exist_and_compile_without_importing_training_deps():
    expected = {
        "__init__.py",
        "dataset.py",
        "environment.yml",
        "export.py",
        "model.py",
        "preprocess.py",
        "train.py",
    }
    assert expected <= {path.name for path in TRAINING_ROOT.iterdir()}

    for filename in (
        "__init__.py",
        "dataset.py",
        "export.py",
        "model.py",
        "preprocess.py",
        "train.py",
    ):
        py_compile.compile(
            str(TRAINING_ROOT / filename),
            doraise=True,
        )


def test_runtime_requirements_remain_free_of_training_dependencies():
    requirements = RUNTIME_REQUIREMENTS.read_text(encoding="utf-8").lower()
    active_lines = [
        line.split("#", 1)[0].strip()
        for line in requirements.splitlines()
        if line.split("#", 1)[0].strip()
    ]
    assert not any(line.startswith("torch") for line in active_lines)
    assert not any(line.startswith("torchvision") for line in active_lines)
    assert not any(line.startswith("lmdb") for line in active_lines)


def test_training_environment_is_isolated_and_cpu_mac_installable():
    environment = yaml.safe_load(
        (TRAINING_ROOT / "environment.yml").read_text(encoding="utf-8")
    )
    assert environment["name"] == "megahan-lite-train"

    dependencies = environment["dependencies"]
    conda_dependencies = [
        item.lower() for item in dependencies if isinstance(item, str)
    ]
    pip_dependencies = next(
        item["pip"] for item in dependencies if isinstance(item, dict)
    )
    pip_dependencies = [item.lower() for item in pip_dependencies]

    assert "python=3.10" in conda_dependencies
    for dependency in (
        "lmdb",
        "pillow",
        "numpy",
        "onnx",
        "onnxruntime",
        "pyyaml",
    ):
        assert any(item.startswith(dependency) for item in conda_dependencies)
    assert any(item.startswith("numpy") and "<2" in item for item in conda_dependencies)
    assert "torch>=2.6,<3" in pip_dependencies
    assert "torchvision>=0.21,<1" in pip_dependencies
    assert not any(
        item.startswith(("torch", "torchvision"))
        for item in conda_dependencies
    )

    serialized = (TRAINING_ROOT / "environment.yml").read_text(
        encoding="utf-8"
    ).lower()
    assert "cuda" not in serialized


def test_dataset_uses_official_lmdb_keys_and_strict_readonly_open():
    tree = _parse_source("dataset.py")
    strings = _literal_strings(tree)
    source = _read_source("dataset.py")

    assert "num-samples" in strings
    assert "image-%09d" in strings
    assert "label-%09d" in strings
    assert "class MegaHanLmdbDataset" in source
    assert "class ClassBalancedBatchSampler" in source
    assert "ConcatDataset" in source

    open_calls = _calls_named(tree, "open")
    lmdb_open = next(
        call
        for call in open_calls
        if isinstance(call.func, ast.Attribute)
        and isinstance(call.func.value, ast.Name)
        and call.func.value.id == "lmdb"
    )
    for keyword in ("readonly", "lock", "readahead"):
        value = _keyword_value(lmdb_open, keyword)
        assert isinstance(value, ast.Constant)
    assert _keyword_value(lmdb_open, "readonly").value is True
    assert _keyword_value(lmdb_open, "lock").value is False
    assert _keyword_value(lmdb_open, "readahead").value is False


def test_balanced_sampler_core_is_deterministic_and_has_positive_pairs():
    make_batches = _load_pure_sampler_function()
    labels = [0, 0, 0, 1, 1, 2, 2, 3]

    first = make_batches(
        labels,
        classes_per_batch=2,
        samples_per_class=2,
        seed=17,
        epoch=0,
    )
    repeated = make_batches(
        labels,
        classes_per_batch=2,
        samples_per_class=2,
        seed=17,
        epoch=0,
    )
    next_epoch = make_batches(
        labels,
        classes_per_batch=2,
        samples_per_class=2,
        seed=17,
        epoch=1,
    )

    assert first == repeated
    assert first != next_epoch
    assert len(first) == 1
    for batch in first:
        batch_labels = [labels[index] for index in batch]
        assert len(batch) == 4
        assert len(set(batch)) == len(batch)
        assert len(set(batch_labels)) == 2
        assert all(batch_labels.count(label) == 2 for label in set(batch_labels))
    assert all(labels[index] != 3 for batch in first for index in batch)


def test_balanced_sampler_core_rejects_invalid_batch_shape():
    make_batches = _load_pure_sampler_function()

    for arguments in (
        {"classes_per_batch": 0, "samples_per_class": 2},
        {"classes_per_batch": 2, "samples_per_class": 1},
        {"classes_per_batch": True, "samples_per_class": 2},
        {"classes_per_batch": 2, "samples_per_class": True},
    ):
        try:
            make_batches([0, 0, 1, 1], seed=1, epoch=0, **arguments)
        except ValueError:
            pass
        else:
            raise AssertionError(f"expected ValueError for {arguments}")


def test_sampler_caches_class_index_and_len_does_not_build_batches():
    tree = _parse_source("dataset.py")
    sampler = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef)
        and node.name == "ClassBalancedBatchSampler"
    )
    source = ast.unparse(sampler)
    length_method = next(
        node
        for node in sampler.body
        if isinstance(node, ast.FunctionDef) and node.name == "__len__"
    )

    assert "self._class_to_indices" in source
    assert not any(
        isinstance(node, ast.Call)
        and (
            isinstance(node.func, ast.Name)
            and node.func.id == "_balanced_batch_indices"
        )
        for node in ast.walk(length_method)
    )


def test_prepared_metadata_verifies_characters_and_lmdb_integrity(tmp_path):
    helpers = _load_train_integrity_functions()
    sha256_file = helpers["_sha256_file"]
    load_metadata = helpers["_load_prepared_metadata"]

    prepared = tmp_path / "prepared"
    lmdb_root = tmp_path / "lmdb"
    prepared.mkdir()
    lmdb_root.mkdir()
    characters_path = prepared / "characters.txt"
    data_path = lmdb_root / "data.mdb"
    characters_path.write_text("一\n", encoding="utf-8")
    data_path.write_bytes(b"approved-lmdb-data")
    characters_hash = sha256_file(characters_path)
    data_hash = sha256_file(data_path)
    (prepared / "provenance.json").write_text(
        json.dumps(
            {
                "character_count": 1,
                "codebook_sha256": "a" * 64,
                "characters_sha256": characters_hash,
            }
        ),
        encoding="utf-8",
    )
    (prepared / "lmdb_sources.json").write_text(
        json.dumps(
            [
                {
                    "root": str(lmdb_root),
                    "data_mdb": {
                        "size": data_path.stat().st_size,
                        "sha256": data_hash,
                    },
                }
            ]
        ),
        encoding="utf-8",
    )

    metadata = load_metadata(
        prepared,
        skip_lmdb_hash_verification=False,
    )

    assert metadata["characters_sha256"] == characters_hash
    assert len(metadata["dataset_fingerprint"]) == 64
    characters_path.write_text("乙\n", encoding="utf-8")
    with pytest.raises(ValueError, match="characters_sha256"):
        load_metadata(prepared, skip_lmdb_hash_verification=False)


def test_lmdb_verification_checks_size_and_hash_but_has_explicit_hash_skip(
    tmp_path,
):
    helpers = _load_train_integrity_functions()
    sha256_file = helpers["_sha256_file"]
    verify = helpers["_verify_lmdb_sources"]

    lmdb_root = tmp_path / "lmdb"
    lmdb_root.mkdir()
    data_path = lmdb_root / "data.mdb"
    data_path.write_bytes(b"first")
    source = {
        "root": str(lmdb_root),
        "data_mdb": {
            "size": 5,
            "sha256": sha256_file(data_path),
        },
    }
    roots, records = verify([source], skip_hash_verification=False)
    assert roots == [str(lmdb_root.resolve())]
    assert records == [{"size": 5, "sha256": sha256_file(data_path)}]

    data_path.write_bytes(b"other")
    with pytest.raises(ValueError, match="SHA-256"):
        verify([source], skip_hash_verification=False)
    roots, records = verify([source], skip_hash_verification=True)
    assert roots == [str(lmdb_root.resolve())]
    assert records == [{"size": 5, "sha256": source["data_mdb"]["sha256"]}]

    data_path.write_bytes(b"longer")
    with pytest.raises(ValueError, match="size"):
        verify([source], skip_hash_verification=True)


def test_dataset_fingerprint_ignores_root_relocation():
    helpers = _load_train_integrity_functions()
    fingerprint = helpers["_dataset_fingerprint"]
    characters_hash = "c" * 64
    first = [
        {"root": "/private/a", "size": 10, "sha256": "a" * 64},
        {"root": "/private/b", "size": 20, "sha256": "b" * 64},
    ]
    relocated = [
        {"root": "/other/a", "size": 10, "sha256": "a" * 64},
        {"root": "/other/b", "size": 20, "sha256": "b" * 64},
    ]

    assert fingerprint(characters_hash, first) == fingerprint(
        characters_hash,
        relocated,
    )
    assert fingerprint(characters_hash, first) != fingerprint(
        "d" * 64,
        first,
    )


def test_dataset_fingerprint_preserves_lmdb_source_order():
    helpers = _load_train_integrity_functions()
    fingerprint = helpers["_dataset_fingerprint"]
    characters_hash = "c" * 64
    published_order = [
        {"root": "/private/a", "size": 10, "sha256": "a" * 64},
        {"root": "/private/b", "size": 20, "sha256": "b" * 64},
    ]
    reversed_order = list(reversed(published_order))

    assert fingerprint(characters_hash, published_order) != fingerprint(
        characters_hash,
        reversed_order,
    )


def test_encoder_is_lightweight_normalized_and_has_no_97k_head():
    tree = _parse_source("model.py")
    source = _read_source("model.py")

    assert "mobilenet_v3_small" in source
    assert "AdaptiveAvgPool2d" in source
    assert "normalize" in source
    assert "supervised_contrastive_loss" in source

    mobile_call = _calls_named(tree, "mobilenet_v3_small")[0]
    weights = _keyword_value(mobile_call, "weights")
    assert isinstance(weights, ast.Constant) and weights.value is None

    linear_calls = _calls_named(tree, "Linear")
    assert linear_calls
    for call in linear_calls:
        assert not any(
            isinstance(argument, ast.Constant) and argument.value == 97_455
            for argument in call.args
        )
    assert "97455" not in source.replace("_", "")


def test_train_cli_checkpoint_and_provenance_contracts_are_present():
    tree = _parse_source("train.py")
    strings = _literal_strings(tree)
    source = _read_source("train.py")

    for option in (
        "--prepared",
        "--output",
        "--epochs",
        "--classes-per-batch",
        "--samples-per-class",
        "--lr",
        "--seed",
        "--workers",
        "--resume",
        "--skip-lmdb-hash-verification",
    ):
        assert option in strings
    for filename in (
        "characters.txt",
        "provenance.json",
        "lmdb_sources.json",
        "last.pt",
        "best.pt",
    ):
        assert filename in strings
    for metadata_key in (
        "model_state",
        "optimizer_state",
        "epoch",
        "embedding_size",
        "input_size",
        "codebook_sha256",
        "character_count",
        "architecture",
        "characters_sha256",
        "dataset_fingerprint",
        "selection_metric",
        "augmented_training_loss",
        "python_random_state",
        "numpy_random_state",
        "torch_cpu_rng_state",
        "torch_cuda_rng_state",
    ):
        assert metadata_key in strings

    assert "AdamW" in source
    assert "set_epoch" in source
    assert "os.replace" in source
    assert "RandomInvert" in source
    assert "RandomAffine" in source
    assert "GaussianBlur" in source
    assert "ColorJitter" in source
    assert "RandomGrayscale" in source
    assert "os.fsync" in source


def test_training_non_blocking_transfers_are_cuda_only():
    tree = _parse_source("train.py")
    helper = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "_use_non_blocking_transfer"
    )
    namespace = {}
    exec(
        compile(
            ast.Module(body=[helper], type_ignores=[]),
            filename="train.py",
            mode="exec",
        ),
        namespace,
    )
    use_non_blocking = namespace["_use_non_blocking_transfer"]
    assert use_non_blocking("cuda") is True
    assert use_non_blocking("mps") is False
    assert use_non_blocking("cpu") is False

    transfer_calls = [
        call
        for call in _calls_named(tree, "to")
        if _keyword_value(call, "non_blocking") is not None
    ]
    assert len(transfer_calls) == 2
    transfer_flags = [
        _keyword_value(call, "non_blocking") for call in transfer_calls
    ]
    assert all(
        isinstance(flag, ast.Name) and flag.id == "non_blocking"
        for flag in transfer_flags
    )
    assert not any(
        isinstance(flag, ast.Constant) and flag.value is True
        for flag in transfer_flags
    )

    assignments = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name)
            and target.id == "non_blocking"
            for target in node.targets
        )
    ]
    assert len(assignments) == 1
    assignment = assignments[0].value
    assert isinstance(assignment, ast.Call)
    assert isinstance(assignment.func, ast.Name)
    assert assignment.func.id == "_use_non_blocking_transfer"
    assert len(assignment.args) == 1
    assert isinstance(assignment.args[0], ast.Attribute)
    assert assignment.args[0].attr == "type"


@pytest.mark.parametrize("size", [(12, 4), (4, 12)])
def test_pil_preprocess_preserves_aspect_ratio_and_uses_border_median(size):
    from glyph_preprocess import pad_glyph_image

    border_color = (12, 34, 56)
    image = Image.new("RGB", size, border_color)
    for x in range(1, image.width - 1):
        for y in range(1, image.height - 1):
            image.putpixel((x, y), (220, 210, 200))

    result = pad_glyph_image(image, 64)
    expected = ImageOps.pad(
        image,
        (64, 64),
        method=Image.Resampling.LANCZOS,
        color=border_color,
        centering=(0.5, 0.5),
    )

    assert result.mode == "RGB"
    assert result.size == (64, 64)
    assert np.array_equal(np.asarray(result), np.asarray(expected))
    assert result.getpixel((0, 0)) == border_color
    if image.width > image.height:
        assert result.getpixel((32, 4)) == border_color
        assert result.getpixel((32, 32)) != border_color
    else:
        assert result.getpixel((4, 32)) == border_color
        assert result.getpixel((32, 32)) != border_color


def test_training_preprocess_is_a_torch_free_reexport_of_root_helper():
    import glyph_preprocess
    from training.megahan_lite import preprocess

    assert preprocess.pad_glyph_image is glyph_preprocess.pad_glyph_image
    assert preprocess.border_color is glyph_preprocess.border_color
    assert preprocess.PadGlyphImage is glyph_preprocess.PadGlyphImage

    root_source = (
        REPOSITORY_ROOT / "glyph_preprocess.py"
    ).read_text(encoding="utf-8")
    root_tree = ast.parse(root_source, filename="glyph_preprocess.py")
    imported_modules = {
        alias.name
        for node in ast.walk(root_tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    } | {
        node.module or ""
        for node in ast.walk(root_tree)
        if isinstance(node, ast.ImportFrom)
    }
    assert not any(
        module == "torch" or module.startswith("torch.")
        for module in imported_modules
    )
    training_tree = _parse_source("preprocess.py")
    assert not any(
        isinstance(node, (ast.FunctionDef, ast.ClassDef))
        for node in training_tree.body
    )


def test_training_transform_begins_with_shared_padding_not_resize():
    tree = _parse_source("train.py")
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "_build_transform"
    )
    compose_call = next(
        call for call in ast.walk(function)
        if isinstance(call, ast.Call)
        and isinstance(call.func, ast.Attribute)
        and call.func.attr == "Compose"
    )
    steps = compose_call.args[0]

    assert isinstance(steps, (ast.List, ast.Tuple))
    assert isinstance(steps.elts[0], ast.Call)
    assert isinstance(steps.elts[0].func, ast.Name)
    assert steps.elts[0].func.id == "PadGlyphImage"
    assert not _calls_named(function, "Resize")
    assert "from glyph_preprocess import PadGlyphImage" in _read_source(
        "train.py"
    )


def test_export_cli_and_checkpoint_contracts_are_present():
    tree = _parse_source("export.py")
    strings = _literal_strings(tree)
    source = _read_source("export.py")

    for option in (
        "--prepared",
        "--checkpoint",
        "--output",
        "--batch-size",
        "--workers",
        "--prototype-version",
        "--skip-lmdb-hash-verification",
        "--max-model-size-mib",
    ):
        assert option in strings

    checkpoint_function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "_load_checkpoint"
    )
    checkpoint_source = ast.unparse(checkpoint_function)
    for metadata_key in (
        "architecture",
        "input_size",
        "embedding_size",
        "character_count",
        "codebook_sha256",
        "characters_sha256",
        "dataset_fingerprint",
        "model_state",
    ):
        assert repr(metadata_key) in checkpoint_source
    assert "map_location='cpu'" in checkpoint_source
    assert "strict=True" in checkpoint_source
    assert "weights_only=False" in checkpoint_source
    assert "logging.getLogger(__name__)" in source


def test_export_onnx_contract_has_fixed_names_and_batch_only_dynamic_axes():
    tree = _parse_source("export.py")
    export_call = next(
        call for call in _calls_named(tree, "export")
        if isinstance(call.func, ast.Attribute)
        and isinstance(call.func.value, ast.Attribute)
        and call.func.value.attr == "onnx"
    )

    assert ast.literal_eval(_keyword_value(export_call, "opset_version")) == 17
    assert ast.literal_eval(_keyword_value(export_call, "dynamo")) is False
    assert (
        ast.literal_eval(_keyword_value(export_call, "external_data"))
        is False
    )
    assert ast.literal_eval(_keyword_value(export_call, "input_names")) == [
        "images"
    ]
    assert ast.literal_eval(_keyword_value(export_call, "output_names")) == [
        "embeddings"
    ]
    assert ast.literal_eval(_keyword_value(export_call, "dynamic_axes")) == {
        "images": {0: "batch"},
        "embeddings": {0: "batch"},
    }


def test_export_rejects_onnx_sidecars_before_asset_validation():
    tree = _parse_source("export.py")
    source = _read_source("export.py")
    helper = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "_reject_onnx_sidecars"
    )
    helper_source = ast.unparse(helper)
    publish = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "export_reviewer_assets"
    )
    publish_source = ast.unparse(publish)

    assert "glob" in helper_source
    assert "model_path.name" in helper_source
    assert "unexpected" in helper_source
    assert "raise ValueError" in helper_source
    assert "_reject_onnx_sidecars(model_path)" in publish_source
    assert publish_source.index("_export_onnx") < publish_source.index(
        "_reject_onnx_sidecars"
    )
    assert publish_source.index("_reject_onnx_sidecars") < (
        publish_source.index("_fsync_file(model_path)")
    )
    assert "glyph_encoder.onnx" in source


def test_export_validates_ort_outputs_with_fixed_random_inputs():
    tree = _parse_source("export.py")
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "_validate_onnx"
    )
    source = ast.unparse(function)

    assert "CPUExecutionProvider" in source
    assert "default_rng" in source
    assert "np.isfinite" in source
    assert "np.linalg.norm" in source
    assert "max_abs" in source
    assert "cosine" in source
    assert source.count("standard_normal") >= 2
    assert "'images'" in source
    assert "'embeddings'" in source


def test_export_prototypes_stream_sums_counts_and_require_every_class():
    tree = _parse_source("export.py")
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "_build_prototypes"
    )
    source = ast.unparse(function)

    assert "np.zeros" in source
    assert "np.add.at" in source
    assert "counts" in source
    assert "counts == 0" in source
    assert "np.linalg.norm" in source
    assert "np.float32" in source
    assert "np.float16" in source
    assert "0 <= labels" in source or "labels < 0" in source
    assert "character_count" in source
    assert "PadGlyphImage" in _read_source("export.py")
    assert "from glyph_preprocess import PadGlyphImage" in _read_source(
        "export.py"
    )
    assert "transforms.Resize" not in _read_source("export.py")


def test_export_publication_is_fsynced_atomic_and_never_clobbers_output():
    tree = _parse_source("export.py")
    source = _read_source("export.py")
    publish_function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "export_reviewer_assets"
    )
    publish_source = ast.unparse(publish_function)

    assert "tempfile.mkdtemp" in publish_source
    assert "atomic_rename_no_replace" in publish_source
    assert "os.replace" not in source
    assert "shutil.rmtree" in publish_source
    assert "os.fsync" in source
    assert "manifest.json" in publish_source
    assert publish_source.index("glyph_encoder.onnx") < publish_source.index(
        "manifest.json"
    )
    assert publish_source.index("prototypes.npy") < publish_source.index(
        "manifest.json"
    )
    assert "15.0" in source
    assert "ensure_ascii=False" in source


@pytest.mark.parametrize("target_has_content", [False, True])
def test_atomic_no_replace_never_overwrites_existing_directory(
    tmp_path,
    target_has_content,
):
    from atomic_publish import atomic_rename_no_replace

    staging = tmp_path / "staging"
    target = tmp_path / "target"
    staging.mkdir()
    target.mkdir()
    (staging / "new.txt").write_text("new", encoding="utf-8")
    if target_has_content:
        (target / "keep.txt").write_text("keep", encoding="utf-8")

    with pytest.raises(FileExistsError):
        atomic_rename_no_replace(staging, target)

    assert (staging / "new.txt").read_text(encoding="utf-8") == "new"
    assert target.is_dir()
    if target_has_content:
        assert (target / "keep.txt").read_text(encoding="utf-8") == "keep"
    else:
        assert list(target.iterdir()) == []


def test_atomic_no_replace_moves_staging_when_target_is_absent(tmp_path):
    from atomic_publish import atomic_rename_no_replace

    staging = tmp_path / "staging"
    target = tmp_path / "target"
    staging.mkdir()
    (staging / "asset.txt").write_text("asset", encoding="utf-8")

    atomic_rename_no_replace(staging, target)

    assert not staging.exists()
    assert (target / "asset.txt").read_text(encoding="utf-8") == "asset"


def test_atomic_no_replace_rejects_platform_without_atomic_primitive(
    tmp_path,
    monkeypatch,
):
    import atomic_publish

    staging = tmp_path / "staging"
    target = tmp_path / "target"
    staging.mkdir()
    monkeypatch.setattr(atomic_publish.sys, "platform", "unsupported-os")

    with pytest.raises(RuntimeError, match="atomic no-replace"):
        atomic_publish.atomic_rename_no_replace(staging, target)

    assert staging.is_dir()
    assert not target.exists()


def test_atomic_no_replace_reports_unavailable_kernel_primitive():
    import errno

    import atomic_publish

    with pytest.raises(RuntimeError, match="atomic no-replace"):
        atomic_publish._raise_rename_error(
            errno.ENOSYS,
            "staging",
            "target",
        )


def test_all_training_sources_use_logging_without_print_or_sys_exit():
    for filename in (
        "dataset.py",
        "export.py",
        "model.py",
        "preprocess.py",
        "train.py",
    ):
        tree = _parse_source(filename)
        assert not _calls_named(tree, "print")
        assert not any(
            isinstance(call.func, ast.Attribute)
            and isinstance(call.func.value, ast.Name)
            and call.func.value.id == "sys"
            and call.func.attr == "exit"
            for call in _calls_named(tree, "exit")
        )


def test_resume_restores_rng_before_any_epoch_iterator_is_created():
    tree = _parse_source("train.py")
    source = _read_source("train.py")
    resume_function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "_resume_training"
    )
    train_function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "train"
    )

    resume_source = ast.unparse(resume_function)
    assert "_restore_rng_state(checkpoint)" in resume_source
    resume_call_line = min(
        node.lineno
        for node in ast.walk(train_function)
        if isinstance(node, ast.Call)
        and (
            isinstance(node.func, ast.Name)
            and node.func.id == "_resume_training"
        )
    )
    first_loader_loop_line = min(
        node.lineno
        for node in ast.walk(train_function)
        if isinstance(node, ast.For)
        and "loader" in ast.unparse(node.iter)
    )
    assert resume_call_line < first_loader_loop_line
    for call in (
        "random.getstate",
        "random.setstate",
        "np.random.get_state",
        "np.random.set_state",
        "torch.get_rng_state",
        "torch.set_rng_state",
    ):
        assert call in source


def test_best_checkpoint_explicitly_uses_augmented_training_loss():
    source = _read_source("train.py")

    assert 'SELECTION_METRIC = "augmented_training_loss"' in source
    assert "logger.warning" in source
    assert "does not estimate generalization" in source
    assert "training-loss best" in source


def test_training_sources_use_logging_without_print_or_sys_exit():
    for filename in ("dataset.py", "model.py", "train.py"):
        tree = _parse_source(filename)
        assert not _calls_named(tree, "print")
        assert not any(
            isinstance(call.func, ast.Attribute)
            and isinstance(call.func.value, ast.Name)
            and call.func.value.id == "sys"
            and call.func.attr == "exit"
            for call in _calls_named(tree, "exit")
        )

    assert "logging.getLogger(__name__)" in _read_source("train.py")
