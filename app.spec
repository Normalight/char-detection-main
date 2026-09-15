# -*- mode: python ; coding: utf-8 -*-
import os
from PyInstaller.building.build_main import Analysis, COLLECT, EXE, PYZ
from PyInstaller.utils.hooks import collect_submodules
import config as app_config

# 项目根目录（用 os.getcwd()，不要用 __file__）
project_dir = os.path.abspath(os.getcwd())

# 只收集默认获胜方案 + legacy 回退，避免 tiny/medium 全部塞进发布目录。
# v6 det/rec 都必须同时带 inference.yml（后处理参数/字典）。
datas = []
selected_models = [app_config.model_path, app_config.crnn_model_path]
if app_config.det_model.startswith('v6-'):
    det_tier = app_config.det_model.split('-', 1)[1]
    selected_models.append(app_config.ppocrv6_det_model_path[det_tier])
if app_config.rec_model.startswith('v6-'):
    rec_tier = app_config.rec_model.split('-', 1)[1]
    selected_models.append(app_config.ppocrv6_rec_model_path[rec_tier])

for model_file in selected_models:
    related_files = [model_file]
    if 'ppocrv6' in model_file:
        related_files.append(os.path.join(os.path.dirname(model_file), 'inference.yml'))
    for src in related_files:
        if os.path.isfile(src):
            dest = os.path.relpath(os.path.dirname(src), project_dir)
            item = (src, dest)
            if item not in datas:
                datas.append(item)

# 可能的隐式依赖：存在就收集，不存在就跳过。PySide6/shiboken6 由
# PyInstaller 的 Qt hooks 按实际 import 收集；不要 collect_submodules，
# 否则会把 WebEngine/3D/SQL 等未使用模块全部打进发布目录。
hiddenimports = []
for pkg in ('fitz', 'pdf2image', 'xlsxwriter', 'yaml'):
    try:
        hiddenimports += collect_submodules(pkg)
    except Exception:
        pass
# 生产 PP-OCRv6 包；bench 只用于开发评测，不进入运行包。
try:
    hiddenimports += collect_submodules('ppocrv6')
except Exception:
    pass

a = Analysis(
    ['app.py'],             # 工程入口
    pathex=[project_dir],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)

# Do not bundle Anaconda's version-suffixed ICU runtime at the application
# root.  PyInstaller can discover it through the parent/base environment even
# when the build uses another conda env.  PySide6's Qt6Core requests the
# unversioned Windows ICU ABI, so the bundled ICU shadows the compatible
# system DLL and fails with WinError 127 ("specified procedure not found").
# Keep any future package-local ICU files; only filter root-level entries.
def _is_incompatible_root_icu(entry):
    destination = os.path.normpath(entry[0])
    if os.path.dirname(destination) not in ('', '.'):
        return False
    filename = os.path.basename(destination).lower()
    return filename == 'icuuc.dll' or (
        filename.startswith('icudt') and filename.endswith('.dll')
    )


a.binaries = [
    entry for entry in a.binaries
    if not _is_incompatible_root_icu(entry)
]

pyz = PYZ(a.pure)

# 生成 onedir 可执行。依赖文件由 COLLECT 放进同名目录，启动时无需先把
# 约 176 MB 的单文件包解压到临时目录，Windows 冷启动会明显更快。
exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='interactive_annotator',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    icon='ico.png',
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='interactive_annotator',
)
