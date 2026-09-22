"""Compile owned modules with Cython after plain-source dependency analysis.

Only the multiprocessing bootstrap remains Python bytecode. Set HANZI_PROTECT=0
explicitly for a development build. Compiler failures stop release packaging.
"""
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys


def prepare(root):
    root = Path(root).resolve()
    if os.environ.get('HANZI_PROTECT', '1') == '0':
        return root
    base = root / 'build' / 'code-protection'
    if base.exists():
        shutil.rmtree(base)
    source = base / 'source'
    source.mkdir(parents=True)
    for file in root.glob('*.py'):
        shutil.copy2(file, source / file.name)
    for package in root.iterdir():
        if package.is_dir() and (package / '__init__.py').is_file():
            shutil.copytree(package, source / package.name,
                            ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    (source / 'packaging').mkdir()
    for name in ('entry.py', 'smoke_test.py'):
        shutil.copy2(root / 'packaging' / name, source / 'packaging' / name)
    spec = importlib.util.find_spec('hanzi_auth')
    if spec is None or not spec.origin:
        raise RuntimeError('请先安装 requirements.txt 中的 hanzi-auth 包')
    auth = Path(spec.origin).parent
    (source / 'hanzi_auth').mkdir()
    for name in ('__init__.py', 'desktop.py', 'policy.py', 'channel.py', 'single_instance.py', 'credentials.py', 'updates.py', 'update_ui.py'):
        shutil.copy2(auth / name, source / 'hanzi_auth' / name)
    update_spec = importlib.util.find_spec('hanzi_update')
    if update_spec is None or not update_spec.origin:
        raise RuntimeError('请先安装 requirements.txt 中的 hanzi-update 包')
    update = Path(update_spec.origin).parent
    shutil.copytree(update, source / 'hanzi_update',
                    ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    return source


def apply(analysis, source):
    if os.environ.get('HANZI_PROTECT', '1') == '0':
        return
    source = Path(source).resolve()
    base = source.parent
    project = source.parents[2]
    selected = []
    for name, path, kind in analysis.pure:
        if not path or path == "-":  # PyInstaller namespace package marker
            continue
        file = Path(path).resolve()
        try:
            relative = file.relative_to(source)
        except ValueError:
            if (file.is_relative_to(project) and not file.is_relative_to(Path(sys.prefix).resolve())) or name in {'hanzi_auth', 'hanzi_update'} or name.startswith(('hanzi_auth.', 'hanzi_update.')):
                raise RuntimeError(f'模块未从保护暂存目录分析：{name}')
            continue
        selected.append({'name': name, 'source': str(file), 'relative': relative.as_posix()})
    required = {'app.py', 'ui/window.py', 'hanzi_auth/desktop.py', 'hanzi_auth/policy.py', 'hanzi_auth/channel.py',
                'hanzi_auth/credentials.py', 'hanzi_auth/single_instance.py',
                'hanzi_update/__init__.py', 'hanzi_update/core.py', 'hanzi_update/native_ui.py'}
    if not required <= {item['relative'] for item in selected}:
        raise RuntimeError('业务入口或账号验证模块未进入保护包')
    (base / 'modules.json').write_text(json.dumps(selected), encoding='utf-8')
    # Use a separate process: setuptools must not consume PyInstaller arguments.
    builder = base / 'compile_modules.py'
    builder.write_text('''import json
from pathlib import Path
from setuptools import setup, Extension
from Cython.Build import cythonize
base = Path(__file__).resolve().parent
modules = json.loads((base / "modules.json").read_text(encoding="utf-8"))
extensions = [Extension(m["name"], [str(Path("source") / m["relative"])]) for m in modules]
setup(name="hanzi-native-modules", ext_modules=cythonize(
    extensions, build_dir="generated",
    compiler_directives={"language_level": 3, "annotation_typing": False,
                         "binding": True, "embedsignature": False},
), script_args=["build_ext", "--build-lib", "native",
                "--build-temp", "objects"])
''', encoding='utf-8')
    try:
        subprocess.run([sys.executable, str(builder)], cwd=base, check=True)
    except subprocess.CalledProcessError as exc:
        raise RuntimeError('主体编译失败。Windows 需 MSVC C++ Build Tools；Mac 需 Xcode Command Line Tools。') from exc
    from importlib.machinery import EXTENSION_SUFFIXES
    binaries = []
    for item in selected:
        relative = Path(item['relative'])
        # A compiled package initializer must retain its __init__ destination.
        stem = Path(*item['name'].split('.'))
        matches = [base / 'native' / (str(stem) + suffix) for suffix in EXTENSION_SUFFIXES]
        matches = [file for file in matches if file.is_file()]
        if len(matches) != 1:
            raise RuntimeError(f'缺少唯一的编译模块：{item["name"]}')
        binary = matches[0]
        destination = binary.relative_to(base / 'native')
        if relative.name == '__init__.py':
            suffix = binary.name[len(stem.name):]
            destination = relative.parent / ('__init__' + suffix)
            target = base / 'native' / destination
            target.parent.mkdir(parents=True, exist_ok=True)
            binary.replace(target)
            binary = target
        binaries.append((destination.as_posix(), str(binary), 'EXTENSION'))
    names = {item['name'] for item in selected}
    analysis.pure[:] = [entry for entry in analysis.pure if entry[0] not in names]
    analysis.binaries.extend(binaries)
    (base / 'manifest.json').write_text(json.dumps({
        'backend': 'cython', 'python': sys.version.split()[0],
        'protected_modules': sorted(item['relative'] for item in selected),
        'native_modules': sorted(entry[0] for entry in binaries),
        'bootstrap': 'packaging/entry.py',
    }, ensure_ascii=False, indent=2), encoding='utf-8')


def build_info(root):
    """Persist the release identity in the bundle independently of protection."""
    from hanzi_auth.updates import native_target
    root = Path(root).resolve()
    path = root / 'build' / 'build_info.json'
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({'product': root.name,
                               'version': os.environ.get('PACKAGE_VERSION', 'v0.0.0-dev'),
                               'target': native_target()}), encoding='utf-8')
    return str(path), '.'
