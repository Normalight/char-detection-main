"""Fetch pinned official updater SDKs and prepare native build inputs."""
import argparse
import hashlib
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import urllib.request
import zipfile

SDKS = {
    'windows-x64': ('https://github.com/vslavik/winsparkle/releases/download/v0.9.4/WinSparkle-0.9.4.zip',
                    '6037df37fc263bd1650a1c4949681a9d40ffe991d01f35892a406cb5d103c976'),
    'macos-arm64': ('https://github.com/sparkle-project/Sparkle/releases/download/2.10.0/Sparkle-2.10.0.tar.xz',
                    'c2bf58aa8387266ac179357b1415d6f2635f044da8be41042af32425dae6da0c'),
}
SDKS['macos-x86_64'] = SDKS['macos-arm64']


def prepare(root, target):
    url, expected = SDKS[target]
    cache = root / 'build/native-sdk' / target
    cache.mkdir(parents=True, exist_ok=True)
    archive = cache / url.rsplit('/', 1)[-1]
    if not archive.is_file() or hashlib.sha256(archive.read_bytes()).hexdigest() != expected:
        temporary = archive.with_suffix('.download')
        with urllib.request.urlopen(url, timeout=120) as source, temporary.open('wb') as output:
            shutil.copyfileobj(source, output)
        if hashlib.sha256(temporary.read_bytes()).hexdigest() != expected:
            temporary.unlink()
            raise ValueError('SDK SHA-256 不匹配')
        temporary.replace(archive)
    if target == 'windows-x64':
        with zipfile.ZipFile(archive) as bundle:
            bundle.extractall(cache)
        sdk = cache / 'WinSparkle-0.9.4'
        compiler = shutil.which('ISCC') or str(Path(os.environ.get('ProgramFiles(x86)', 'C:/Program Files (x86)')) / 'Inno Setup 6/ISCC.exe')
        if not Path(compiler).is_file():
            raise ValueError('需要安装 Inno Setup 6')
        return {'WINSPARKLE_DLL': str(sdk / 'x64/Release/WinSparkle.dll'),
                'WINSPARKLE_LICENSE': str(sdk / 'COPYING'), 'ISCC': compiler}
    with tarfile.open(archive) as bundle:
        bundle.extractall(cache, filter='data')
    bridge = cache / 'SparkleBridge.dylib'
    subprocess.run(['xcrun', 'clang++', '-arch', target.removeprefix('macos-'), '-mmacosx-version-min=14.0',
        '-fobjc-arc', '-dynamiclib', str(root / 'packaging/sparkle_bridge/SparkleBridge.mm'),
        '-F' + str(cache), '-framework', 'Sparkle', '-framework', 'Cocoa',
        '-Wl,-rpath,@loader_path', '-Wl,-install_name,@rpath/SparkleBridge.dylib',
        '-o', str(bridge)], check=True)
    return {'SPARKLE_FRAMEWORK': str(cache / 'Sparkle.framework'), 'SPARKLE_BRIDGE': str(bridge),
            'HANZI_MAC_SIGNING': 'adhoc'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--target', required=True, choices=SDKS)
    parser.add_argument('--root', type=Path, default=Path.cwd())
    args = parser.parse_args()
    values = prepare(args.root.resolve(), args.target)
    if os.environ.get('GITHUB_ENV'):
        with open(os.environ['GITHUB_ENV'], 'a', encoding='utf-8') as output:
            output.writelines(f'{key}={value}\n' for key, value in values.items())
    for key, value in values.items():
        print(f'{key}={value}')


if __name__ == '__main__':
    main()
