# -*- coding: utf-8 -*-
"""Validate and prepare approved local MegaHan97K assets."""

import argparse
import errno
import hashlib
import json
import logging
import os
from pathlib import Path
import stat
import sys
import tempfile

if os.name == "nt":
    import msvcrt
else:
    import fcntl


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from megahan_assets import (  # noqa: E402
    MEGAHAN_CHARACTER_COUNT,
    build_megahan_provenance,
    parse_megahan_codebook,
    parse_megahan_ids,
    sha256_file,
)


logger = logging.getLogger(__name__)

_PREPARED_TEMP_FILENAMES = (
    "characters.txt",
    "provenance.json",
    "lmdb_sources.json",
)


def _existing_local_path(path_value, label, require_file=False):
    raw_value = os.fspath(path_value)
    if "://" in raw_value:
        raise ValueError(f"{label}只接受已存在的本地路径：{raw_value}")
    try:
        resolved = Path(raw_value).expanduser().resolve(strict=True)
    except FileNotFoundError:
        raise FileNotFoundError(
            f"{label}不存在，只接受已存在的本地路径：{raw_value}"
        ) from None
    if require_file and not resolved.is_file():
        raise ValueError(f"{label}必须是本地文件：{resolved}")
    return resolved


def _output_path(output_dir):
    candidate = Path(output_dir).expanduser()
    if not candidate.is_absolute():
        candidate = Path.cwd() / candidate
    return candidate.resolve(strict=False)


def _is_within(path, root):
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _file_metadata(path):
    return {
        "size": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def _describe_lmdb_root(root_value):
    root = _existing_local_path(root_value, "LMDB root")
    if not root.is_dir():
        raise ValueError(f"LMDB root 必须是本地目录：{root}")

    entry = {"root": str(root)}
    data_path = root / "data.mdb"
    if not data_path.is_file():
        raise FileNotFoundError(
            f"LMDB 目录缺少必需的 data.mdb：{root}"
        )
    entry["data_mdb"] = _file_metadata(data_path)

    lock_path = root / "lock.mdb"
    if lock_path.exists():
        if not lock_path.is_file():
            raise ValueError(f"LMDB lock.mdb 必须是文件：{lock_path}")
        entry["lock_mdb"] = _file_metadata(lock_path)
    return entry


def _write_bytes_fsynced(path, content):
    with path.open("xb") as output_file:
        output_file.write(content)
        output_file.flush()
        os.fsync(output_file.fileno())


def _write_text_fsynced(path, content):
    _write_bytes_fsynced(path, content.encode("utf-8"))


def _json_text(value):
    return json.dumps(
        value,
        ensure_ascii=False,
        indent=2,
    ) + "\n"


def _acquire_output_lock(lock_path):
    lock_fd = os.open(
        lock_path,
        os.O_CREAT | os.O_RDWR,
        0o600,
    )
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
                "另一个准备任务正在运行，输出锁已被占用："
                f"{lock_path}"
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


def _stat_identity(stat_result):
    return stat_result.st_dev, stat_result.st_ino


def _open_prepared_temp_directory(path):
    if os.unlink in os.supports_dir_fd:
        open_flags = os.O_RDONLY
        if hasattr(os, "O_DIRECTORY"):
            open_flags |= os.O_DIRECTORY
        directory_fd = os.open(path, open_flags)
        return directory_fd, _stat_identity(os.fstat(directory_fd))
    return None, _stat_identity(os.lstat(path))


def _path_is_owned_directory(path, path_identity):
    try:
        current_stat = os.lstat(path)
    except FileNotFoundError:
        return False
    return (
        _stat_identity(current_stat) == path_identity
        and stat.S_ISDIR(current_stat.st_mode)
    )


def _unlink_known_temp_files_by_fd(directory_fd):
    for filename in _PREPARED_TEMP_FILENAMES:
        try:
            os.unlink(filename, dir_fd=directory_fd)
        except FileNotFoundError:
            pass
        except IsADirectoryError:
            pass


def _unlink_known_temp_files_by_path(path):
    for filename in _PREPARED_TEMP_FILENAMES:
        child_path = path / filename
        try:
            child_path.unlink()
        except FileNotFoundError:
            pass
        except IsADirectoryError:
            pass


def _cleanup_prepared_temp(path, directory_fd, path_identity):
    try:
        path_is_owned = _path_is_owned_directory(path, path_identity)
        if directory_fd is not None:
            _unlink_known_temp_files_by_fd(directory_fd)
        elif path_is_owned:
            _unlink_known_temp_files_by_path(path)

        if path_is_owned:
            try:
                os.rmdir(path)
            except OSError as error:
                if error.errno not in (errno.ENOTEMPTY, errno.EEXIST):
                    raise
    finally:
        if directory_fd is not None:
            os.close(directory_fd)


def prepare_megahan_inputs(
    codebook_path,
    ids_path,
    lmdb_roots,
    output_dir,
    expected_count=MEGAHAN_CHARACTER_COUNT,
    allow_repository_output=False,
    repository_root=None,
):
    """Validate local MegaHan inputs and atomically prepare small metadata."""

    output_path = _output_path(output_dir)
    effective_repository_root = Path(
        repository_root if repository_root is not None else REPOSITORY_ROOT
    ).expanduser().resolve(strict=False)
    if (
        not allow_repository_output
        and _is_within(output_path, effective_repository_root)
    ):
        raise ValueError(
            "默认禁止将 MegaHan 数据输出到仓库目录内："
            f"{output_path}；如确有需要请显式允许"
        )
    if os.path.lexists(output_path):
        raise FileExistsError(
            f"输出目录已存在，为避免覆盖请换新目录：{output_path}"
        )

    codebook = _existing_local_path(
        codebook_path,
        "MegaHan 码本",
        require_file=True,
    )
    characters = parse_megahan_codebook(
        codebook,
        expected_count=expected_count,
    )
    characters_content = "".join(
        f"{character}\n" for character in characters
    )
    characters_sha256 = hashlib.sha256(
        characters_content.encode("utf-8")
    ).hexdigest()

    ids_hash = None
    if ids_path is not None:
        ids = _existing_local_path(
            ids_path,
            "MegaHan IDS",
            require_file=True,
        )
        parse_megahan_ids(ids, characters)
        ids_hash = sha256_file(ids)

    provenance = {
        "format_version": 1,
        **build_megahan_provenance(
            codebook_sha256=sha256_file(codebook),
            ids_sha256=ids_hash,
        ),
    }
    provenance["character_count"] = len(characters)
    provenance["characters_sha256"] = characters_sha256
    lmdb_sources = [
        _describe_lmdb_root(root) for root in lmdb_roots
    ]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = output_path.parent / (
        f".{output_path.name}.megahan.lock"
    )
    lock_fd = _acquire_output_lock(lock_path)
    temporary_path = None
    temporary_directory_fd = None
    temporary_identity = None
    published = False
    try:
        if os.path.lexists(output_path):
            raise FileExistsError(
                "获取输出锁后发现目录已存在，为避免覆盖已拒绝发布："
                f"{output_path}"
            )
        temporary_path = Path(
            tempfile.mkdtemp(
                prefix=f".{output_path.name}.tmp-",
                dir=output_path.parent,
            )
        )
        (
            temporary_directory_fd,
            temporary_identity,
        ) = _open_prepared_temp_directory(temporary_path)
        _write_text_fsynced(
            temporary_path / "characters.txt",
            characters_content,
        )
        _write_text_fsynced(
            temporary_path / "provenance.json",
            _json_text(provenance),
        )
        _write_text_fsynced(
            temporary_path / "lmdb_sources.json",
            _json_text(lmdb_sources),
        )
        if os.path.lexists(output_path):
            raise FileExistsError(
                "输出目录在准备过程中已出现，为避免覆盖已拒绝替换："
                f"{output_path}"
            )
        os.replace(temporary_path, output_path)
        published = True
    finally:
        try:
            if (
                temporary_path is not None
                and temporary_identity is not None
            ):
                if published:
                    if temporary_directory_fd is not None:
                        os.close(temporary_directory_fd)
                else:
                    _cleanup_prepared_temp(
                        temporary_path,
                        temporary_directory_fd,
                        temporary_identity,
                    )
        finally:
            _release_output_lock(lock_fd)

    return provenance


def build_argument_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--codebook", required=True)
    parser.add_argument("--ids")
    parser.add_argument(
        "--lmdb-root",
        action="append",
        default=[],
        help="可重复提供的本地 LMDB 根目录",
    )
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--expected-count",
        type=int,
        default=MEGAHAN_CHARACTER_COUNT,
    )
    parser.add_argument(
        "--allow-repository-output",
        action="store_true",
    )
    return parser


def main(argv=None):
    args = build_argument_parser().parse_args(argv)
    provenance = prepare_megahan_inputs(
        codebook_path=args.codebook,
        ids_path=args.ids,
        lmdb_roots=args.lmdb_root,
        output_dir=args.output,
        expected_count=args.expected_count,
        allow_repository_output=args.allow_repository_output,
    )
    logger.info(
        "MegaHan assets prepared: output=%s characters=%d",
        Path(args.output).expanduser().resolve(strict=False),
        provenance["character_count"],
    )
    return 0


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )
    main()
