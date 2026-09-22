"""Exercise native dependency guards with minimal Windows x64 PE fixtures."""
import importlib.util
from pathlib import Path
import struct
import tempfile
import unittest

spec = importlib.util.spec_from_file_location('windows_bundle',
    Path(__file__).resolve().parents[1] / 'packaging/windows_bundle.py')
policy = importlib.util.module_from_spec(spec)
spec.loader.exec_module(policy)


def make_pe(path, normal=(), delayed=(), machine=0x8664):
    data = bytearray(4096)
    data[:2] = b'MZ'
    struct.pack_into('<I', data, 60, 128)
    data[128:132] = b'PE\0\0'
    struct.pack_into('<HH', data, 132, machine, 1)
    struct.pack_into('<H', data, 148, 240)
    struct.pack_into('<H', data, 152, 0x20b)
    struct.pack_into('<Q', data, 176, 0x140000000)
    struct.pack_into('<I', data, 260, 16)
    struct.pack_into('<IIII', data, 400, 3584, 0x1000, 3584, 512)
    string_offset = 2048
    for directory, names, table, width in ((1, normal, 512, 20), (13, delayed, 1024, 32)):
        if not names:
            continue
        struct.pack_into('<II', data, 264 + 8 * directory,
                         0x1000 + table - 512, (len(names) + 1) * width)
        for i, name in enumerate(names):
            pointer = 0x1000 + string_offset - 512
            values = (0, 0, 0, pointer, 0) if directory == 1 else (1, pointer, 0, 0, 0, 0, 0, 0)
            struct.pack_into('<' + 'I' * len(values), data, table + i * width, *values)
            raw = name.encode('ascii') + b'\0'
            data[string_offset:string_offset + len(raw)] = raw
            string_offset += len(raw)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return str(path)


class BundlePolicyTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def entry(self, name, **kwargs):
        return (name, make_pe(self.root / name, **kwargs), 'BINARY')

    def test_regular_and_delay_imports(self):
        path = make_pe(self.root / 'consumer.dll', normal=('Qt6Core.dll',), delayed=('Qt6Pdf.dll',))
        self.assertEqual(policy.read_pe_imports(path), {'qt6core.dll', 'qt6pdf.dll'})

    def test_dependency_conflicts_fail_before_returning_a_plan(self):
        for option in ('normal', 'delayed'):
            with self.subTest(option=option):
                entries = [self.entry('consumer.pyd', **{option: ('onnxruntime.dll',)}),
                           self.entry('onnxruntime/capi/onnxruntime.dll')]
                with self.assertRaisesRegex(ValueError, 'consumer.pyd requires'):
                    policy.filter_binaries(entries)
                self.assertTrue(Path(entries[1][1]).is_file())

    def test_qt_pdf_is_pruned_as_a_dependency_group(self):
        entries = [self.entry('PySide6/plugins/imageformats/qpdf.dll', normal=('Qt6Pdf.dll',)),
                   self.entry('PySide6/Qt6Pdf.dll'), self.entry('PySide6/Qt6Core.dll')]
        kept, report = policy.filter_binaries(entries, ['qt_pdf'])
        self.assertEqual([x[0] for x in kept], ['PySide6/Qt6Core.dll'])
        self.assertEqual(len(report['removed']), 2)

    def test_models_bindings_and_unrelated_same_names_are_retained(self):
        model = self.root / 'model.onnx'
        model.write_bytes(b'model payload')
        entries = [self.entry('onnxruntime/capi/onnxruntime_pybind11_state.pyd'),
                   self.entry('onnxruntime/capi/onnxruntime_providers_shared.dll'),
                   self.entry('another/onnxruntime.dll'),
                   ('models/inference.onnx', str(model), 'DATA')]
        kept, report = policy.filter_binaries(entries)
        self.assertEqual(kept, entries)
        self.assertEqual(report['saved_bytes'], 0)
        self.assertEqual(model.read_bytes(), b'model payload')

    def test_exact_windows_paths_accept_case_and_separators(self):
        entry = self.entry('onnxruntime/capi/onnxruntime.dll')
        kept, report = policy.filter_binaries([('ONNXRUNTIME\\capi\\ONNXRUNTIME.DLL', entry[1], entry[2])])
        self.assertEqual(kept, [])
        self.assertEqual(report['saved_bytes'], Path(entry[1]).stat().st_size)

    def test_unrecognized_component_and_non_x64_fail(self):
        with self.assertRaisesRegex(ValueError, 'Unknown Windows components'):
            policy.filter_binaries([], ['typo'])
        with self.assertRaisesRegex(ValueError, 'Expected Windows x64'):
            policy.filter_binaries([self.entry('other.dll', machine=0x14c)])

    def test_malformed_pe_fails_closed(self):
        bad = self.root / 'broken.dll'
        bad.write_bytes(b'MZ')
        with self.assertRaisesRegex(ValueError, 'Cannot audit'):
            policy.filter_binaries([('broken.dll', str(bad), 'BINARY')])


if __name__ == '__main__':
    unittest.main()
