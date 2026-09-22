# macOS-only build, independent of the existing Windows app.spec.
import os
from pathlib import Path
if os.environ.get("MAC_ARCH", "arm64") != "arm64":
    raise ValueError("Mac builds require arm64")
os.environ["MAC_ARCH"] = "arm64"
root = Path(SPECPATH).parent
import sys
sys.path.insert(0, str(root))
datas = []
import config
models = []
if config.det_model.startswith('v6-'):
    models.append(Path(config.ppocrv6_det_model_path[config.det_model.split('-', 1)[1]]))
if config.rec_model.startswith('v6-'):
    models.append(Path(config.ppocrv6_rec_model_path[config.rec_model.split('-', 1)[1]]))
for model in models:
    resources = [model]
    if 'ppocrv6' in str(model):
        resources.append(model.with_name('inference.yml'))
    for source in resources:
        if not source.is_file():
            raise FileNotFoundError(f'缺少模型资源：{source}')
        datas.append((str(source), str(source.parent.relative_to(root))))
import runpy
native_paths = []
if os.environ['MAC_ARCH'] == 'arm64' and os.environ.get('HANZI_MINIMAL_OPENCV', '1') != '0':
    native_path, notices = runpy.run_path(str(root / 'packaging/minimal_opencv.py'))['prepare'](root)
    native_paths.append(native_path)
    sys.path.insert(0, native_path)
    datas.append(notices)
protection = runpy.run_path(str(root / 'packaging/protection.py'))
code_root = protection['prepare'](root)
native_release = runpy.run_path(str(root / 'packaging/native_release.py'))
datas.extend(native_release['analysis_data'](root))
datas.append(protection['build_info'](root))
a = Analysis([str(code_root / 'packaging/entry.py')],
    pathex=native_paths + [str(code_root), str(code_root / 'packaging')], datas=datas, binaries=[],
    hiddenimports=['_cffi_backend'], hookspath=[], runtime_hooks=[],
    excludes=['PyQt5', 'PyQt6', 'PySide2', 'tkinter', 'pytest'], noarchive=False)
runpy.run_path(str(root / 'packaging/slimming.py'))['apply'](a, root)
protection['apply'](a, code_root)
pyz = PYZ(a.pure)
exe = EXE(pyz, a.scripts, [], exclude_binaries=True, name='runtime',
    console=False, strip=False, upx=False, target_arch=os.environ['MAC_ARCH'],
    codesign_identity=None)
coll = COLLECT(exe, a.binaries, a.datas, strip=False, upx=False, name='runtime')
app = BUNDLE(coll, name='Runtime.app', bundle_identifier='local.character.runtime',
    info_plist={'LSMinimumSystemVersion': '14.0', 'NSHighResolutionCapable': True})
