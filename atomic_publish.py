# -*- coding: utf-8 -*-
"""Platform-native atomic rename that can never replace its target."""

import ctypes
import errno
import os
import sys


def _raise_rename_error(error_number, source, target):
    if error_number in (errno.EEXIST, errno.ENOTEMPTY):
        raise FileExistsError(
            error_number,
            os.strerror(error_number),
            os.fspath(target),
        )
    unavailable_errors = {
        errno.ENOSYS,
        getattr(errno, "ENOTSUP", errno.ENOSYS),
        getattr(errno, "EOPNOTSUPP", errno.ENOSYS),
    }
    if error_number in unavailable_errors:
        raise RuntimeError(
            "atomic no-replace primitive is unavailable: "
            f"{os.strerror(error_number)}"
        )
    if not error_number:
        error_number = errno.EIO
    raise OSError(
        error_number,
        os.strerror(error_number),
        os.fspath(source),
        os.fspath(target),
    )


def _libc_function(name):
    try:
        library = ctypes.CDLL(None, use_errno=True)
    except OSError as error:
        raise RuntimeError(
            f"atomic no-replace C library is unavailable: {error}"
        ) from None
    try:
        return getattr(library, name)
    except AttributeError:
        raise RuntimeError(
            f"atomic no-replace primitive {name} is unavailable"
        ) from None


def atomic_rename_no_replace(source, target):
    """Atomically rename ``source`` only when ``target`` does not exist."""

    source_bytes = os.fsencode(source)
    target_bytes = os.fsencode(target)

    if sys.platform == "darwin":
        renamex_np = _libc_function("renamex_np")
        renamex_np.argtypes = [
            ctypes.c_char_p,
            ctypes.c_char_p,
            ctypes.c_uint,
        ]
        renamex_np.restype = ctypes.c_int
        ctypes.set_errno(0)
        result = renamex_np(source_bytes, target_bytes, 0x4)
        if result != 0:
            _raise_rename_error(ctypes.get_errno(), source, target)
        return

    if sys.platform.startswith("linux"):
        renameat2 = _libc_function("renameat2")
        renameat2.argtypes = [
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_uint,
        ]
        renameat2.restype = ctypes.c_int
        ctypes.set_errno(0)
        result = renameat2(
            -100,
            source_bytes,
            -100,
            target_bytes,
            0x1,
        )
        if result != 0:
            _raise_rename_error(ctypes.get_errno(), source, target)
        return

    if os.name == "nt":
        try:
            os.rename(source, target)
        except OSError as error:
            if error.errno in (errno.EEXIST, errno.ENOTEMPTY):
                raise FileExistsError(
                    error.errno,
                    error.strerror,
                    os.fspath(target),
                ) from None
            raise
        return

    raise RuntimeError(
        f"atomic no-replace rename is unsupported on {sys.platform}"
    )
