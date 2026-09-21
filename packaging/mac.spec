# macOS-only build, independent of the existing Windows app.spec.
import os
from pathlib import Path
root = Path(SPECPATH).parent
import sys
sys.path.insert(0, str(root))
datas = []
import config
models = []
if config.det_model == "legacy":
    models.append(Path(config.model_path))
if config.rec_model == "crnn":
    models.append(Path(config.crnn_model_path))
if config.det_model.startswith('v6-'):
    models.append(Path(config.ppocrv6_det_model_path[config.det_model.split('-', 1)[1]]))
if config.rec_model.startswith('v6-'):
    models.append(Path(config.ppocrv6_rec_model_path[config.rec_model.split('-', 1)[1]]))
if config.angle_detect:
    models.append(Path(config.angle_net_path))
for model in models:
    resources = [model]
    if 'ppocrv6' in str(model):
        resources.append(model.with_name('inference.yml'))
    for source in resources:
        if not source.is_file():
            raise FileNotFoundError(f'缺少模型资源：{source}')
        datas.append((str(source), str(source.parent.relative_to(root))))
import runpy
protection = runpy.run_path(str(root / 'packaging/protection.py'))
code_root = protection['prepare'](root)
datas.append(protection['build_info'](root))
a = Analysis([str(code_root / 'packaging/entry.py')],
    pathex=[str(code_root), str(code_root / 'packaging')], datas=datas, binaries=[],
    hiddenimports=['_cffi_backend'], hookspath=[], runtime_hooks=[],
    excludes=['PyQt5', 'PyQt6', 'PySide2', 'tkinter', 'pytest'], noarchive=False)
protection['apply'](a, code_root)
pyz = PYZ(a.pure)
exe = EXE(pyz, a.scripts, [], exclude_binaries=True, name='runtime',
    console=False, strip=False, upx=False, target_arch=os.environ['MAC_ARCH'],
    codesign_identity=None)
coll = COLLECT(exe, a.binaries, a.datas, strip=False, upx=False, name='runtime')
app = BUNDLE(coll, name='Runtime.app', bundle_identifier='local.character.runtime',
    info_plist={'LSMinimumSystemVersion': '14.0', 'NSHighResolutionCapable': True})
