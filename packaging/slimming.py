"""Trim dependencies verified with the macOS arm64 release smoke tests.

Keep this allowlist version-specific: library linkage and Qt plugin contents can
change between wheels. Other platforms retain their normal dependency closure.
"""
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import sys


def apply(analysis, root):
    if sys.platform != 'darwin' or os.environ.get('MAC_ARCH', platform.machine()) != 'arm64':
        return
    if os.environ.get('HANZI_SLIM', '1') == '0':
        return
    versions = {name: importlib.metadata.version(name) for name in ('PySide6', 'onnxruntime')}
    qt_frameworks = {
        'QtPdf.framework', 'QtQml.framework', 'QtQmlMeta.framework',
        'QtQmlModels.framework', 'QtQmlWorkerScript.framework',
        'QtQuick.framework', 'QtVirtualKeyboard.framework', 'QtOpenGL.framework',
    }

    def unused(destination):
        parts = Path(destination).parts
        if versions['PySide6'] == '6.8.3' and parts[:2] == ('PySide6', 'Qt'):
            # qpdf and the virtual keyboard are the roots of these extra
            # dependencies. Native platform input and image plugins remain.
            return bool(qt_frameworks.intersection(parts)) or parts[-1] in {
                'libqpdf.dylib', 'libqtvirtualkeyboardplugin.dylib',
            }
        # The 1.22.1 arm64 Python extension contains its inference engine and
        # does not load the separately distributed C API library.
        return versions['onnxruntime'] == '1.22.1' and destination == (
            'onnxruntime/capi/libonnxruntime.1.22.1.dylib')

    removed = []
    def excluded(entry):
        return unused(entry[0]) or (entry[2] == 'SYMLINK' and unused(entry[1]))

    for name in ('binaries', 'datas'):
        entries = getattr(analysis, name)
        removed.extend(entry[0] for entry in entries if excluded(entry))
        entries[:] = [entry for entry in entries if not excluded(entry)]
    report = Path(root) / 'build' / 'slimming.json'
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(json.dumps({'target': 'macos-arm64', 'versions': versions,
                                  'removed': sorted(removed)}, indent=2), encoding='utf-8')
