# -*- coding: utf-8 -*-
"""Tests for preparing approved, local MegaHan97K assets."""

import hashlib
import importlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

import megahan_assets


_MODULE_NAME = "bench.prepare_megahan_assets"
_MODULE_SPEC = importlib.util.find_spec(_MODULE_NAME)
prepare_module = (
    importlib.import_module(_MODULE_NAME) if _MODULE_SPEC is not None else None
)


class PrepareModuleContractTests(unittest.TestCase):
    def test_exports_prepare_megahan_inputs(self):
        self.assertIsNotNone(
            prepare_module,
            "bench.prepare_megahan_assets must exist",
        )
        self.assertTrue(
            callable(
                getattr(prepare_module, "prepare_megahan_inputs", None)
            )
        )


@unittest.skipIf(
    prepare_module is None,
    "implementation module does not exist yet",
)
class PrepareMegaHanInputsTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory(
            prefix="prepare_megahan_assets_"
        )
        self.addCleanup(self.temp_dir.cleanup)
        self.root = Path(self.temp_dir.name)
        self.repository_root = self.root / "repo"
        self.repository_root.mkdir()

    def write_bytes(self, relative_path, content):
        path = self.root / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
        return path

    def write_text(self, relative_path, content):
        return self.write_bytes(relative_path, content.encode("utf-8"))

    def prepare(self, output_name="prepared", **overrides):
        arguments = {
            "codebook_path": self.write_text(
                "inputs/codebook.txt",
                "一\n𠀀\n",
            ),
            "ids_path": None,
            "lmdb_roots": [],
            "output_dir": self.root / output_name,
            "expected_count": 2,
            "repository_root": self.repository_root,
        }
        arguments.update(overrides)
        return prepare_module.prepare_megahan_inputs(**arguments)

    def test_writes_canonical_characters_provenance_and_empty_lmdb_list(self):
        codebook = self.write_bytes(
            "canonical/codebook.txt",
            b"\xef\xbb\xbf" + "一\n𠀀\n".encode("utf-8"),
        )
        output = self.root / "prepared"

        result = self.prepare(codebook_path=codebook, output_dir=output)

        codebook_hash = hashlib.sha256(codebook.read_bytes()).hexdigest()
        canonical_characters = "一\n𠀀\n".encode("utf-8")
        expected_provenance = {
            "format_version": 1,
            "source": megahan_assets.MEGAHAN_SOURCE,
            "license": megahan_assets.MEGAHAN_LICENSE,
            "character_count": 2,
            "codebook_sha256": codebook_hash,
            "characters_sha256": hashlib.sha256(
                canonical_characters
            ).hexdigest(),
        }
        self.assertEqual(
            (output / "characters.txt").read_bytes(),
            canonical_characters,
        )
        self.assertEqual(
            json.loads(
                (output / "provenance.json").read_text(encoding="utf-8")
            ),
            expected_provenance,
        )
        self.assertEqual(
            json.loads(
                (output / "lmdb_sources.json").read_text(encoding="utf-8")
            ),
            [],
        )
        self.assertEqual(result, expected_provenance)

    def test_ids_are_optional_and_hash_is_recorded_when_present(self):
        ids = self.write_text("inputs/MegaHan_IDS.txt", "一:一\n")

        without_ids = self.prepare(output_name="without-ids")
        with_ids = self.prepare(output_name="with-ids", ids_path=ids)

        self.assertNotIn("ids_sha256", without_ids)
        self.assertEqual(
            with_ids["ids_sha256"],
            hashlib.sha256(ids.read_bytes()).hexdigest(),
        )

    def test_records_each_lmdb_data_file_and_optional_lock_file(self):
        first = self.root / "lmdb-one"
        second = self.root / "lmdb-two"
        first.mkdir()
        second.mkdir()
        first_data = self.write_bytes("lmdb-one/data.mdb", b"first-data")
        first_lock = self.write_bytes("lmdb-one/lock.mdb", b"first-lock")
        second_data = self.write_bytes("lmdb-two/data.mdb", b"second-data")
        output = self.root / "prepared"

        self.prepare(
            lmdb_roots=[first, second],
            output_dir=output,
        )

        sources = json.loads(
            (output / "lmdb_sources.json").read_text(encoding="utf-8")
        )
        self.assertEqual(
            sources,
            [
                {
                    "root": str(first.resolve()),
                    "data_mdb": {
                        "size": len(b"first-data"),
                        "sha256": hashlib.sha256(b"first-data").hexdigest(),
                    },
                    "lock_mdb": {
                        "size": len(b"first-lock"),
                        "sha256": hashlib.sha256(b"first-lock").hexdigest(),
                    },
                },
                {
                    "root": str(second.resolve()),
                    "data_mdb": {
                        "size": len(b"second-data"),
                        "sha256": hashlib.sha256(b"second-data").hexdigest(),
                    },
                },
            ],
        )
        self.assertEqual(first_data.stat().st_size, len(b"first-data"))
        self.assertEqual(first_lock.stat().st_size, len(b"first-lock"))
        self.assertEqual(second_data.stat().st_size, len(b"second-data"))

    def test_rejects_lmdb_directory_without_data_mdb(self):
        lmdb_root = self.root / "missing-data"
        lmdb_root.mkdir()
        output = self.root / "prepared"

        with self.assertRaisesRegex(FileNotFoundError, "data\\.mdb"):
            self.prepare(lmdb_roots=[lmdb_root], output_dir=output)

        self.assertFalse(output.exists())

    def test_rejects_file_as_lmdb_root(self):
        lmdb_file = self.write_bytes("not-an-lmdb-root.bin", b"not-lmdb")
        output = self.root / "prepared"

        with self.assertRaisesRegex(ValueError, "目录"):
            self.prepare(lmdb_roots=[lmdb_file], output_dir=output)

        self.assertFalse(output.exists())

    def test_cli_help_describes_lmdb_root_as_directory(self):
        help_text = prepare_module.build_argument_parser().format_help()

        self.assertIn("本地 LMDB 根目录", help_text)
        self.assertNotIn("根目录或文件", help_text)

    def test_rejects_output_inside_repository_by_default(self):
        output = self.repository_root / "generated" / "megahan"

        with self.assertRaisesRegex(ValueError, "仓库"):
            self.prepare(output_dir=output)

        self.assertFalse(output.exists())

    def test_rejects_repository_root_with_repository_specific_error(self):
        with self.assertRaisesRegex(ValueError, "仓库"):
            self.prepare(output_dir=self.repository_root)

    def test_allows_similar_repository_prefix_outside_repository(self):
        output = self.root / "repo-other" / "megahan"

        self.prepare(output_dir=output)

        self.assertTrue((output / "characters.txt").is_file())

    def test_explicit_flag_allows_output_inside_repository(self):
        output = self.repository_root / "generated" / "megahan"

        self.prepare(
            output_dir=output,
            allow_repository_output=True,
        )

        self.assertTrue((output / "characters.txt").is_file())

    def test_refuses_to_overwrite_existing_output(self):
        output = self.root / "prepared"
        output.mkdir()
        sentinel = output / "keep.txt"
        sentinel.write_text("do not replace", encoding="utf-8")

        with self.assertRaisesRegex(FileExistsError, "已存在"):
            self.prepare(output_dir=output)

        self.assertEqual(
            sentinel.read_text(encoding="utf-8"),
            "do not replace",
        )

    def test_existing_unlocked_lock_file_is_reused(self):
        output = self.root / "prepared"
        lock = output.parent / f".{output.name}.megahan.lock"
        lock.write_text("persistent lock", encoding="utf-8")

        self.prepare(output_dir=output)

        self.assertEqual(
            lock.read_text(encoding="utf-8"),
            "persistent lock",
        )
        self.assertTrue((output / "characters.txt").is_file())

    def test_locked_advisory_file_reports_running_task(self):
        output = self.root / "prepared"
        lock = output.parent / f".{output.name}.megahan.lock"
        lock_fd = os.open(lock, os.O_CREAT | os.O_RDWR, 0o600)
        if os.fstat(lock_fd).st_size == 0:
            os.write(lock_fd, b"\0")
        os.lseek(lock_fd, 0, os.SEEK_SET)
        if os.name == "nt":
            import msvcrt

            msvcrt.locking(lock_fd, msvcrt.LK_NBLCK, 1)
        else:
            import fcntl

            fcntl.flock(
                lock_fd,
                fcntl.LOCK_EX | fcntl.LOCK_NB,
            )
        try:
            with self.assertRaisesRegex(
                RuntimeError,
                "另一个准备任务正在运行",
            ):
                self.prepare(output_dir=output)
        finally:
            if os.name == "nt":
                os.lseek(lock_fd, 0, os.SEEK_SET)
                msvcrt.locking(lock_fd, msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(lock_fd, fcntl.LOCK_UN)
            os.close(lock_fd)

        self.assertTrue(lock.is_file())
        self.assertFalse(output.exists())

    def test_released_lock_file_persists_and_can_be_reacquired(self):
        output = self.root / "prepared"
        lock = output.parent / f".{output.name}.megahan.lock"

        self.prepare(output_dir=output)
        self.assertTrue(lock.is_file())
        shutil.rmtree(output)

        self.prepare(output_dir=output)

        self.assertTrue(lock.is_file())
        self.assertTrue((output / "characters.txt").is_file())

    def test_output_created_after_lock_is_not_overwritten(self):
        output = self.root / "prepared"
        lock = output.parent / f".{output.name}.megahan.lock"
        sentinel = output / "keep.txt"
        original_write = prepare_module._write_text_fsynced

        def create_output_during_write(path, content):
            self.assertTrue(lock.is_file())
            if not output.exists():
                output.mkdir()
                sentinel.write_text("do not replace", encoding="utf-8")
            original_write(path, content)

        with mock.patch.object(
            prepare_module,
            "_write_text_fsynced",
            side_effect=create_output_during_write,
        ):
            with self.assertRaisesRegex(FileExistsError, "避免覆盖"):
                self.prepare(output_dir=output)

        self.assertEqual(
            sentinel.read_text(encoding="utf-8"),
            "do not replace",
        )
        self.assertTrue(lock.is_file())
        self.assertEqual(
            list(output.parent.glob(f".{output.name}.tmp-*")),
            [],
        )

    def test_publish_failure_cleans_temporary_directory_and_owned_lock(self):
        output = self.root / "prepared"
        lock = output.parent / f".{output.name}.megahan.lock"
        saw_owned_lock = []

        def fail_publish(source, destination):
            saw_owned_lock.append(lock.is_file())
            raise OSError("publish failed")

        with mock.patch.object(
            prepare_module.os,
            "replace",
            side_effect=fail_publish,
        ):
            with self.assertRaisesRegex(OSError, "publish failed"):
                self.prepare(output_dir=output)

        self.assertEqual(saw_owned_lock, [True])
        self.assertFalse(output.exists())
        self.assertTrue(lock.is_file())
        self.assertEqual(
            list(output.parent.glob(f".{output.name}.tmp-*")),
            [],
        )

    def test_write_failure_cleans_temporary_directory_and_owned_lock(self):
        output = self.root / "prepared"
        lock = output.parent / f".{output.name}.megahan.lock"
        saw_owned_lock = []

        def fail_write(path, content):
            saw_owned_lock.append(lock.is_file())
            raise OSError("write failed")

        with mock.patch.object(
            prepare_module,
            "_write_text_fsynced",
            side_effect=fail_write,
        ):
            with self.assertRaisesRegex(OSError, "write failed"):
                self.prepare(output_dir=output)

        self.assertEqual(saw_owned_lock, [True])
        self.assertFalse(output.exists())
        self.assertTrue(lock.is_file())
        self.assertEqual(
            list(output.parent.glob(f".{output.name}.tmp-*")),
            [],
        )

    def test_replaced_lock_is_not_removed_when_owned_lock_is_released(self):
        output = self.root / "prepared"
        lock = output.parent / f".{output.name}.megahan.lock"

        def replace_lock_and_fail(path, content):
            self.assertTrue(lock.is_file())
            lock.unlink()
            lock.write_text("replacement lock", encoding="utf-8")
            raise OSError("write failed after lock replacement")

        with mock.patch.object(
            prepare_module,
            "_write_text_fsynced",
            side_effect=replace_lock_and_fail,
        ):
            with self.assertRaisesRegex(
                OSError,
                "write failed after lock replacement",
            ):
                self.prepare(output_dir=output)

        try:
            self.assertTrue(lock.is_file())
            self.assertEqual(
                lock.read_text(encoding="utf-8"),
                "replacement lock",
            )
            self.assertFalse(output.exists())
            self.assertEqual(
                list(output.parent.glob(f".{output.name}.tmp-*")),
                [],
            )
        finally:
            lock.unlink(missing_ok=True)

    def test_replaced_temporary_path_is_not_recursively_removed(self):
        output = self.root / "prepared"
        lock = output.parent / f".{output.name}.megahan.lock"
        paths = {}
        original_write = prepare_module._write_text_fsynced

        def replace_temporary_directory_and_fail(path, content):
            if not paths:
                original_write(path, content)
                paths["first_write_completed"] = True
                return
            owned_temporary = path.parent
            moved_owned = owned_temporary.with_name(
                f"{owned_temporary.name}.owned-original"
            )
            owned_temporary.rename(moved_owned)
            owned_temporary.mkdir()
            replacement_sentinel = owned_temporary / "keep.txt"
            replacement_sentinel.write_text(
                "replacement directory",
                encoding="utf-8",
            )
            paths.update(
                replacement=owned_temporary,
                replacement_sentinel=replacement_sentinel,
                moved_owned=moved_owned,
            )
            raise OSError("write failed after temp replacement")

        with mock.patch.object(
            prepare_module,
            "_write_text_fsynced",
            side_effect=replace_temporary_directory_and_fail,
        ):
            with self.assertRaisesRegex(
                OSError,
                "write failed after temp replacement",
            ):
                self.prepare(output_dir=output)

        try:
            self.assertTrue(paths["replacement"].is_dir())
            self.assertEqual(
                paths["replacement_sentinel"].read_text(encoding="utf-8"),
                "replacement directory",
            )
            self.assertTrue(paths["moved_owned"].is_dir())
            self.assertEqual(
                list(paths["moved_owned"].iterdir()),
                [],
            )
            self.assertTrue(lock.is_file())
            self.assertFalse(output.exists())
        finally:
            for path in (
                paths.get("replacement"),
                paths.get("moved_owned"),
            ):
                if path is not None and path.exists():
                    shutil.rmtree(path)
            lock.unlink(missing_ok=True)

    def test_parse_failure_leaves_no_output_or_temporary_directory(self):
        invalid_codebook = self.write_text(
            "invalid/codebook.txt",
            "一\n一\n",
        )
        output = self.root / "prepared"

        with self.assertRaisesRegex(ValueError, "重复字符"):
            self.prepare(
                codebook_path=invalid_codebook,
                output_dir=output,
            )

        self.assertFalse(output.exists())
        self.assertEqual(
            list(output.parent.glob(f".{output.name}.tmp-*")),
            [],
        )

    def test_cli_runs_from_an_unrelated_working_directory(self):
        codebook = self.write_text("cli/codebook.txt", "一\n乙\n")
        output = self.root / "cli-output"
        unrelated_cwd = self.root / "unrelated"
        unrelated_cwd.mkdir()
        script = (
            Path(prepare_module.__file__).resolve()
            if prepare_module is not None
            else Path(__file__).parents[1]
            / "bench"
            / "prepare_megahan_assets.py"
        )

        completed = subprocess.run(
            [
                sys.executable,
                str(script),
                "--codebook",
                str(codebook),
                "--output",
                str(output),
                "--expected-count",
                "2",
            ],
            cwd=unrelated_cwd,
            capture_output=True,
            text=True,
            check=False,
        )

        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertEqual(
            (output / "characters.txt").read_text(encoding="utf-8"),
            "一\n乙\n",
        )


if __name__ == "__main__":
    unittest.main()
