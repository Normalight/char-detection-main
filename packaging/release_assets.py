"""Validate release names and stage consistently named application ZIPs."""
import argparse
import os
from pathlib import Path
import re

VERSION = re.compile(r"v(?:0|[1-9][0-9]*)\.(?:0|[1-9][0-9]*)\.(?:0|[1-9][0-9]*)(?:-[0-9A-Za-z-]+(?:\.[0-9A-Za-z-]+)*)?")
TARGETS = {"windows-x64", "macos-x86_64", "macos-arm64"}


def asset_stem(project, version, target):
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", project):
        raise ValueError("Invalid project name")
    if not (VERSION.fullmatch(version) or re.fullmatch(r"dev-[0-9a-f]{7,40}", version)):
        raise ValueError("Version must be vMAJOR.MINOR.PATCH (optionally -rc.1) or dev-<commit>")
    if target not in TARGETS:
        raise ValueError("Unsupported package target")
    return f"{project}-{version}-{target}"


def stage_archive(dist, project, version, target):
    stem = asset_stem(project, version, target)
    archives = list(dist.glob("*.zip"))
    if len(archives) != 1:
        raise ValueError(f"Expected exactly one built ZIP, found {len(archives)}")
    source = archives[0]
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
