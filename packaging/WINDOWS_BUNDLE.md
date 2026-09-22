# Windows x64 bundle policy

`app.spec` applies `windows_bundle.py` after native modules are collected. The
policy targets the Qt Widgets desktop application with ONNX CPU inference,
OpenCV image processing, and PyMuPDF PDF rendering. macOS uses its own spec.

The policy selects exact destinations for four optional component groups:

| Group | Destinations | Current unpacked size |
| --- | --- | ---: |
| Video backend | OpenCV 4.11 FFmpeg DLL | 26.66 MiB when present |
| Virtual keyboard | Qt input-context plugin and its VirtualKeyboard, Quick, QML libraries | 12.41 MiB |
| Qt PDF image handler | qpdf image plugin and Qt6Pdf | 5.13 MiB |
| ONNX C API | onnxruntime/capi/onnxruntime.dll | 14.17 MiB |

The selected versions are OpenCV 4.11.0.86, PySide6 6.8.3 and ONNX Runtime 1.22.1.
The Python binding `onnxruntime_pybind11_state.pyd`, provider support library,
Qt Windows platform plugin, image codecs and all model resources remain part
of the bundle. Existing Windows ICU compatibility handling remains in the spec.

Every retained PE binary is checked for regular and delay-loaded imports of the
selected DLLs. An import conflict, incompatible architecture, unreadable PE, or
selected DLL collected as data stops the build. The policy records its file
selection and byte counts in `build/windows-bundle-report.json`.

PE imports do not enumerate dynamic loading. When dependencies or application
features change, rerun the packaged smoke tests, real page and character OCR,
PDF rendering, image/crop and JSON/Excel export, native Chinese IME entry,
cancellation and restart. Include a clean Windows x64 machine. A Qt Quick/QML
UI, virtual keyboard frontend, Qt PDF reader, video input, or ONNX C API client
requires reassessing the corresponding group before shipping.

The 2026-09-22 local v1.0.5 optimization artifacts reuse the original business
binaries and models. They validate the file-selection policy; they are not a
new source compilation. Keep runtime regression results separate from build
unit tests and clean-system validation.
