"""Validate release names and stage consistently named application ZIPs."""
import argparse
import os
from pathlib import Path
import re
import zipfile

VERSION = re.compile(r"v(?:0|[1-9][0-9]*)\.(?:0|[1-9][0-9]*)\.(?:0|[1-9][0-9]*)(?:-[0-9A-Za-z-]+(?:\.[0-9A-Za-z-]+)*)?")
from hanzi_update.core import ASSET_PREFIXES, APP_NAMES
PROJECTS = {product: (prefix, APP_NAMES[product]) for product, prefix in ASSET_PREFIXES.items()}
TARGETS = {"windows-x64", "macos-arm64", "macos-x86_64"}


def asset_stem(project, version, target):
    if project not in PROJECTS:
        raise ValueError("Invalid project name")
    if not (VERSION.fullmatch(version) or re.fullmatch(r"dev-[0-9a-f]{7,40}", version)):
        raise ValueError("Version must be vMAJOR.MINOR.PATCH (optionally -rc.1) or dev-<commit>")
    if target not in TARGETS:
        raise ValueError("Unsupported package target")
    return f"{PROJECTS[project][0]}-{version}-{target}"


def validate_mac_archive(archive, project):
    """Mac downloads contain exactly the app and its TXT at ZIP root."""
    app = PROJECTS[project][1]
    with zipfile.ZipFile(archive) as bundle:
        names = []
        for info in bundle.infolist():
            name = info.filename
            # ditto may store UTF-8 filenames without setting the UTF-8 flag.
            if not info.flag_bits & 0x800:
                try:
                    name = name.encode("cp437").decode("utf-8")
                except (UnicodeEncodeError, UnicodeDecodeError):
                    pass
            names.append(name)
        if {name.split("/", 1)[0] for name in names} != {app, "MAC使用说明.txt"}:
            raise ValueError("Mac ZIP root must contain only the app and MAC使用说明.txt")
        if f"{app}/Contents/Info.plist" not in names:
            raise ValueError("Missing application Info.plist")
        if f"{app}/Contents/MacOS/launcher" not in names:
            raise ValueError("Missing application launcher")


def stage_archive(dist, project, version, target):
    stem = asset_stem(project, version, target)
    archives = list(dist.glob("*.zip"))
    if len(archives) != 1:
        raise ValueError(f"Expected exactly one built ZIP, found {len(archives)}")
    source = archives[0]
    if target.startswith("macos-"):
        validate_mac_archive(source, project)
    destination = dist / "release" / (stem + ".zip")
    destination.parent.mkdir(parents=True, exist_ok=True)
    source.replace(destination)
    source.with_suffix(".zip.sha256").unlink(missing_ok=True)
    destination.with_suffix(".zip.sha256").unlink(missing_ok=True)
    return destination


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", required=True)
    parser.add_argument("--version", required=True)
    parser.add_argument("--target", required=True, choices=sorted(TARGETS))
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--dist", type=Path, default=Path(__file__).resolve().parents[1] / "dist")
    args = parser.parse_args()
    if os.environ.get("GITHUB_EVENT_NAME") == "release" and not VERSION.fullmatch(args.version):
        parser.error("Release tags must use vMAJOR.MINOR.PATCH, optionally with a prerelease suffix")
    if args.validate_only:
        asset_stem(args.project, args.version, args.target)
    else:
        stage_archive(args.dist, args.project, args.version, args.target)


if __name__ == "__main__":
    main()
