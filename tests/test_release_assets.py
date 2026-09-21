"""Release artifacts must contain verified installers and matching update ZIPs."""
import importlib.util
from pathlib import Path
import zipfile
import hashlib
import json

import pytest

spec = importlib.util.spec_from_file_location(
    "release_assets", Path(__file__).resolve().parents[1] / "packaging/release_assets.py")
assets = importlib.util.module_from_spec(spec)
spec.loader.exec_module(assets)


@pytest.mark.parametrize("target", ["windows-x64", "macos-x86_64", "macos-arm64"])
def test_release_contains_installer_and_matching_update(tmp_path, target):
    archive = tmp_path / "旧名称.zip"
    with zipfile.ZipFile(archive, "w") as bundle:
        bundle.writestr("MAC使用说明.txt", "首次打开与使用方法")
        bundle.writestr("中文字符标注工具.app/Contents/Info.plist", "plist")
        bundle.writestr("中文字符标注工具.app/Contents/MacOS/launcher", "launcher")
    archive.with_suffix(".zip.sha256").write_text("obsolete checksum")
    installer = tmp_path / (f"切字-v1.2.3-rc.1-{target}" + ('.dmg' if target.startswith('macos-') else '-Setup.exe'))
    installer.write_bytes(b'installer')
    installer.with_suffix(installer.suffix + '.json').write_text(json.dumps(dict(
        product='char-detection-main', version='v1.2.3-rc.1', target=target,
        name=installer.name, size=installer.stat().st_size,
        sha256=hashlib.sha256(installer.read_bytes()).hexdigest(),
        update_sha256=hashlib.sha256(archive.read_bytes()).hexdigest())))
    result = assets.stage_archive(tmp_path, "char-detection-main", "v1.2.3-rc.1", target)
    assert result.name == f"切字-v1.2.3-rc.1-{target}.zip"
    assert len(list(result.parent.iterdir())) == 3
    assert not list(tmp_path.rglob("*.sha256"))
    with zipfile.ZipFile(result) as bundle:
        assert bundle.read("MAC使用说明.txt").decode() == "首次打开与使用方法"
    assert not list(tmp_path.glob("*.zip*"))


@pytest.mark.parametrize("version", ["v1.2.3", "v0.1.0-beta.2", "dev-abc1234"])
def test_accepts_documented_versions(version):
    assert assets.asset_stem("group-by-character", version, "windows-x64") == f"归并-{version}-windows-x64"


@pytest.mark.parametrize("version", ["1.0.0", "v1", "v01.2.3", "../v1.2.3", "v1.2.3/extra", "v1.2.3\nunsafe"])
def test_rejects_ambiguous_or_unsafe_version_names(version):
    with pytest.raises(ValueError):
        assets.asset_stem("group-by-character", version, "windows-x64")


def test_refuses_to_choose_between_stale_archives(tmp_path):
    for name in ("old.zip", "new.zip"):
        (tmp_path / name).write_bytes(b"placeholder")
    with pytest.raises(ValueError, match="exactly one"):
        assets.stage_archive(tmp_path, "group-by-character", "v1.0.0", "macos-arm64")
    assert not (tmp_path / "release").exists()


def test_release_event_cannot_use_a_development_version(monkeypatch):
    monkeypatch.setenv("GITHUB_EVENT_NAME", "release")
    monkeypatch.setattr("sys.argv", ["release_assets.py", "--project", "group-by-character",
                                   "--version", "dev-abc1234", "--target", "windows-x64", "--validate-only"])
    with pytest.raises(SystemExit) as error:
        assets.main()
    assert error.value.code == 2


@pytest.mark.parametrize("extra", ["outer/", "dependencies-arm64.txt", "打开软件.command", "__MACOSX/._app"])
def test_rejects_extra_mac_zip_entries(tmp_path, extra):
    archive = tmp_path / "app.zip"
    with zipfile.ZipFile(archive, "w") as bundle:
        bundle.writestr("MAC使用说明.txt", "说明")
        bundle.writestr("字形归并工具.app/Contents/Info.plist", "plist")
        bundle.writestr("字形归并工具.app/Contents/MacOS/launcher", "executable")
        bundle.writestr(extra, "unexpected")
    with pytest.raises(ValueError, match="only the app"):
        assets.stage_archive(tmp_path, "group-by-character", "v1.0.0", "macos-arm64")
    assert archive.exists()


def test_missing_installer_keeps_update_archive_unstaged(tmp_path):
    archive = tmp_path / 'update.zip'
    archive.write_bytes(b'update')
    with pytest.raises(ValueError, match='installer'):
        assets.stage_archive(tmp_path, 'group-by-character', 'v1.0.0', 'windows-x64')
    assert archive.exists()
    assert not (tmp_path / 'release').exists()
