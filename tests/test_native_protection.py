"""Release protection must fail closed and preserve namespace/package imports."""
import importlib.machinery
import json
from pathlib import Path
import runpy
import subprocess
import sys
from types import SimpleNamespace

import pytest


@pytest.fixture
def protection(tmp_path, monkeypatch):
    monkeypatch.setenv('HANZI_PROTECT', '1')
    helper = runpy.run_path(str(Path(__file__).resolve().parents[1] / 'packaging/protection.py'))
    source = tmp_path / 'project/build/code-protection/source'
    source.mkdir(parents=True)
    entries = []
    for name, relative in [('app', 'app.py'), ('hanzi_auth', 'hanzi_auth/__init__.py'),
                           ('hanzi_auth.desktop', 'hanzi_auth/desktop.py'),
                           ('hanzi_auth.policy', 'hanzi_auth/policy.py'),
                           ('hanzi_auth.updates', 'hanzi_auth/updates.py'),
                           ('hanzi_auth.update_ui', 'hanzi_auth/update_ui.py')]:
        file = source / relative
        file.parent.mkdir(parents=True, exist_ok=True)
        file.write_text('pass\n')
        entries.append((name, str(file), 'PYMODULE'))
    entries.append(('dbnet', '-', 'PYMODULE'))
    analysis = SimpleNamespace(pure=entries, binaries=[])
    return helper, source, analysis


def test_native_replacement_preserves_package_and_namespace(protection, monkeypatch):
    helper, source, analysis = protection
    suffix = importlib.machinery.EXTENSION_SUFFIXES[0]

    def compile_stub(*args, **kwargs):
        modules = json.loads((source.parent / 'modules.json').read_text())
        for module in modules:
            binary = source.parent / 'native' / (module['name'].replace('.', '/') + suffix)
            binary.parent.mkdir(parents=True, exist_ok=True)
            binary.write_bytes(b'native-test-fixture')

    monkeypatch.setattr(subprocess, 'run', compile_stub)
    helper['apply'](analysis, source)
    assert analysis.pure == [('dbnet', '-', 'PYMODULE')]
    assert any(dest == 'hanzi_auth/__init__' + suffix for dest, _, _ in analysis.binaries)
    assert all(Path(path).is_file() for _, path, _ in analysis.binaries)
    manifest = json.loads((source.parent / 'manifest.json').read_text())
    assert manifest['backend'] == 'cython'
    assert 'app.py' in manifest['protected_modules']


def test_compile_failure_never_ships_plain_fallback(protection, monkeypatch):
    helper, source, analysis = protection
    before = list(analysis.pure)

    def fail(*args, **kwargs):
        raise subprocess.CalledProcessError(1, 'compiler')

    monkeypatch.setattr(subprocess, 'run', fail)
    with pytest.raises(RuntimeError, match='主体编译失败'):
        helper['apply'](analysis, source)
    assert analysis.pure == before
    assert not (source.parent / 'manifest.json').exists()


def test_missing_compiled_module_aborts(protection, monkeypatch):
    helper, source, analysis = protection
    monkeypatch.setattr(subprocess, 'run', lambda *args, **kwargs: None)
    with pytest.raises(RuntimeError, match='缺少唯一的编译模块'):
        helper['apply'](analysis, source)


def test_original_checkout_module_is_rejected(protection):
    helper, source, analysis = protection
    analysis.pure.append(('extra', str(source.parents[2] / 'extra.py'), 'PYMODULE'))
    with pytest.raises(RuntimeError, match='模块未从保护暂存目录分析'):
        helper['apply'](analysis, source)


def test_project_local_virtualenv_dependencies_remain_uncompiled(protection, monkeypatch):
    helper, source, analysis = protection
    prefix = source.parents[2] / 'build/mac-env-arm64'
    dependency = ('pkg_resources', str(prefix / 'lib/site-packages/pkg_resources/__init__.py'), 'PYMODULE')
    analysis.pure.append(dependency)
    monkeypatch.setattr(sys, 'prefix', str(prefix))
    suffix = importlib.machinery.EXTENSION_SUFFIXES[0]
    def compile_stub(*args, **kwargs):
        modules = json.loads((source.parent / 'modules.json').read_text())
        assert not any(m['name'] == 'pkg_resources' for m in modules)
        for module in modules:
            binary = source.parent / 'native' / (module['name'].replace('.', '/') + suffix)
            binary.parent.mkdir(parents=True, exist_ok=True)
            binary.write_bytes(b'native-test-fixture')
    monkeypatch.setattr(subprocess, 'run', compile_stub)
    helper['apply'](analysis, source)
    assert dependency in analysis.pure
