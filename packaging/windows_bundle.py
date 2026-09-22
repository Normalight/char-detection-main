"""Windows x64 bundle policy for the Qt Widgets / CPU inference clients.

Keep the model files and Python inference binding. Optional native components
are selected by exact destination and checked against every retained PE import.
Dynamic plugin use still requires the packaged application's regression tests.
"""
from pathlib import Path, PurePosixPath
import json
import struct
import sys


COMPONENTS = {
    'video': ('cv2/opencv_videoio_ffmpeg4110_64.dll',),
    'virtual_keyboard': (
        'PySide6/plugins/platforminputcontexts/qtvirtualkeyboardplugin.dll',
        'PySide6/Qt6VirtualKeyboard.dll',
        'PySide6/Qt6Quick.dll',
        'PySide6/Qt6Qml.dll',
        'PySide6/Qt6QmlMeta.dll',
        'PySide6/Qt6QmlModels.dll',
        'PySide6/Qt6QmlWorkerScript.dll',
    ),
    'qt_pdf': ('PySide6/plugins/imageformats/qpdf.dll', 'PySide6/Qt6Pdf.dll'),
    'ort_c_api': ('onnxruntime/capi/onnxruntime.dll',),
}


def normalized(path):
    return str(PurePosixPath(str(path).replace('\\', '/'))).casefold()


def read_pe_imports(path):
    """Read normal and delay-loaded DLL names; reject incompatible/invalid PE."""
    data = Path(path).read_bytes()
    try:
        offset = struct.unpack_from('<I', data, 60)[0]
        if data[:2] != b'MZ' or data[offset:offset + 4] != b'PE\0\0':
            raise ValueError('Missing PE signature')
        machine, section_count = struct.unpack_from('<HH', data, offset + 4)
        optional_size = struct.unpack_from('<H', data, offset + 20)[0]
        optional = offset + 24
        magic = struct.unpack_from('<H', data, optional)[0]
        if machine != 0x8664 or magic != 0x20b:
            raise ValueError('Expected Windows x64 PE32+')
        image_base = struct.unpack_from('<Q', data, optional + 24)[0]
        directory_count = struct.unpack_from('<I', data, optional + 108)[0]
        sections = []
        for index in range(section_count):
            section = optional + optional_size + index * 40
            size, address, raw_size, raw = struct.unpack_from('<IIII', data, section + 8)
            sections.append((address, max(size, raw_size), raw))

        def file_offset(address):
            for virtual, size, raw in sections:
                if virtual <= address < virtual + size:
                    value = raw + address - virtual
                    if 0 <= value < len(data):
                        return value
            raise ValueError(f'Unmapped PE address: {address}')

        names = set()
        for index, width in ((1, 20), (13, 32)):
            if index >= directory_count:
                continue
            address, size = struct.unpack_from('<II', data, optional + 112 + index * 8)
            if not address:
                continue
            table = file_offset(address)
            for entry in range(size // width):
                values = struct.unpack_from('<' + 'I' * (width // 4), data, table + entry * width)
                if not any(values):
                    break
                name_address = values[3] if index == 1 else values[1]
                if index == 13 and not values[0] & 1:
                    name_address -= image_base
                start = file_offset(name_address)
                names.add(data[start:data.index(0, start)].decode('ascii').casefold())
            else:
                raise ValueError('Unterminated PE import table')
        return names
    except (ValueError, IndexError, struct.error, UnicodeDecodeError) as exc:
        raise ValueError(f'Cannot audit {path}: {exc}') from exc


def filter_binaries(entries, components=None):
    """Return retained TOC entries and an audit report, without modifying files."""
    selected = tuple(COMPONENTS) if components is None else tuple(components)
    unknown = set(selected) - COMPONENTS.keys()
    if unknown:
        raise ValueError(f'Unknown Windows components: {sorted(unknown)}')
    paths = {normalized(p): group for group in selected for p in COMPONENTS[group]}
    retained, removed = [], []
    for entry in entries:
        destination, source, kind = entry
        group = paths.get(normalized(destination))
        if group:
            removed.append({'path': destination, 'component': group,
                            'bytes': Path(source).stat().st_size})
        else:
            retained.append(entry)
    removed_names = {PurePosixPath(normalized(x['path'])).name for x in removed}
    for destination, source, kind in retained:
        if Path(destination).suffix.lower() not in ('.dll', '.pyd', '.exe'):
            continue
        imports = read_pe_imports(source)
        conflicts = imports & removed_names
        if conflicts:
            raise ValueError(f'{destination} requires selected components: {sorted(conflicts)}')
    return retained, {'platform': 'windows-x64', 'components': list(selected),
                      'removed': removed, 'saved_bytes': sum(x['bytes'] for x in removed),
                      'retained_entries': len(retained), 'pe_import_check': 'passed'}


def apply(analysis, report_path):
    """Apply the Windows policy after Cython has contributed its native modules."""
    if sys.platform != 'win32':
        raise RuntimeError('The Windows bundle policy must run on Windows')
    retained, report = filter_binaries(analysis.binaries)
    # Ensure a hook cannot reintroduce selected native binaries as data files.
    selected_paths = {normalized(p) for group in COMPONENTS.values() for p in group}
    if any(normalized(entry[0]) in selected_paths for entry in analysis.datas):
        raise ValueError('A selected Windows binary was collected as a data file')
    analysis.binaries[:] = retained
    path = Path(report_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
