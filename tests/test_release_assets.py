"""Release artifacts must contain only the correctly named application ZIP."""
import importlib.util
from pathlib import Path
import zipfile

import pytest

spec = importlib.util.spec_from_file_location(
    "release_assets", Path(__file__).resolve().parents[1] / "packaging/release_assets.py")
assets = importlib.util.module_from_spec(spec)
spec.loader.exec_module(assets)


@pytest.mark.parametrize("target", ["windows-x64", "macos-x86_64", "macos-arm64"])
def test_release_contains_only_application_zip(tmp_path, target):
    archive = tmp_path / "旧名称.zip"
    with zipfile.ZipFile(archive, "w") as bundle:
        bundle.writestr("MAC使用说明.txt", "首次打开与使用方法")
        bundle.writestr("中文字符标注工具.app/Contents/Info.plist", "plist")
        bundle.writestr("中文字符标注工具.app/Contents/MacOS/launcher", "launcher")
    archive.with_suffix(".zip.sha256").write_text("obsolete checksum")
    result = assets.stage_archive(tmp_path, "char-detection-main", "v1.2.3-rc.1", target)
    assert result.name == f"切字-v1.2.3-rc.1-{target}.zip"
    assert list(result.parent.iterdir()) == [result]
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
