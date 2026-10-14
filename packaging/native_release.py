"""Native updater build/package helpers, shared by both desktop products.

No credentials are persisted. Signing uses the platform keychain or SDK tools.
"""
import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import plistlib
import shutil
import subprocess
import sys
from hanzi_update.backends.base import UpdateConfig, release_version
from hanzi_update.core import REGISTRY

# (应用名去 .app, 可执行名去 .exe, 外层 bundle 标识) 全部来自 hanzi-update 产品注册表。
PRODUCTS = {product: (info['app_name'][:-len('.app')], info['exe_name'][:-len('.exe')], info['bundle_id'])
            for product, info in REGISTRY.items()}


def config_for(root):
    feed = os.environ.get('HANZI_UPDATE_FEED', '')
    key = os.environ.get('HANZI_UPDATE_PUBLIC_KEY', '')
    if not feed and not key:
        return None
    return UpdateConfig(root.name, os.environ.get('PACKAGE_VERSION', ''), feed, key,
                        os.environ.get('HANZI_UPDATE_CHANNEL', 'stable'))


def analysis_data(root):
    root = Path(root).resolve()
    config = config_for(root)
    if config is None:
        return []
    path = root / 'build/native_update.json'
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(asdict(config)), encoding='utf-8')
    return [(str(path), '.')]


def exe_version(root):
    config = config_for(Path(root).resolve())
    if config is None:
        return None
    from PyInstaller.utils.win32.versioninfo import (VSVersionInfo, FixedFileInfo,
        StringFileInfo, StringTable, StringStruct, VarFileInfo, VarStruct)
    version, _ = release_version(config.version)
    values = tuple(map(int, version.split('.'))) + (0,)
    return VSVersionInfo(ffi=FixedFileInfo(filevers=values, prodvers=values, fileType=1), kids=[
        StringFileInfo([StringTable('040904B0', [StringStruct('CompanyName', 'Hanzi'),
            StringStruct('ProductName', config.product), StringStruct('ProductVersion', version),
            StringStruct('FileVersion', version)])]), VarFileInfo([VarStruct('Translation', [1033, 1200])])])


def mac_version_info():
    raw = os.environ.get('PACKAGE_VERSION', 'v0.0.0-dev')
    if raw == 'v0.0.0-dev':
        version, build = '0.0.0', '0'
    else:
        version, build = release_version(raw)
    return dict(CFBundleVersion=build, CFBundleShortVersionString=version,
                CFBundleDevelopmentRegion='zh_CN', CFBundleLocalizations=['zh_CN', 'zh-Hans'])


def runtime_bundle_info(root):
    name, _, identifier = PRODUCTS[Path(root).resolve().name]
    return dict(mac_version_info(), CFBundleName=name, CFBundleDisplayName=name,
                CFBundleIdentifier=identifier + '.runtime',
                LSMinimumSystemVersion='14.0', NSHighResolutionCapable=True)


def bundle_version(app):
    path = app / 'Contents/Info.plist'
    data = plistlib.loads(path.read_bytes())
    data.update(mac_version_info())
    path.write_bytes(plistlib.dumps(data))


def verify_built_config(root, path):
    config = config_for(root)
    if config is None:
        raise ValueError('原生构建需要 feed 和公钥')
    if json.loads(path.read_text(encoding='utf-8')) != asdict(config):
        raise ValueError('构建产物配置不匹配，请重新打包')
    return config


def package_windows(root):
    name, executable, _ = PRODUCTS[root.name]
    folder = root / 'dist' / executable
    config = verify_built_config(root, folder / '_internal/native_update.json')
    sdk = Path(os.environ['WINSPARKLE_DLL']).resolve()
    if not sdk.is_file():
        raise ValueError('需要 WinSparkle >= 0.9 的 x64 DLL')
    shutil.copy2(sdk, folder / 'WinSparkle.dll')
    # Preserve the third-party license supplied with the SDK.
    shutil.copy2(Path(os.environ['WINSPARKLE_LICENSE']), folder / 'WinSparkle-LICENSE.txt')
    version, _ = release_version(config.version)
    subprocess.run([os.environ.get('ISCC', 'ISCC.exe'), '/DProductId=' + root.name,
        '/DProductName=' + name, '/DProductExe=' + executable, '/DProductVersion=' + version,
        '/DSourceDir=' + str(folder), '/DOutputDir=' + str(root / 'dist'),
        str(root / 'packaging/native_installer.iss')], check=True)


def mac_signing_config():
    identity = os.environ.get('APPLE_SIGN_IDENTITY', '').strip()
    mode = os.environ.get('HANZI_MAC_SIGNING', '').strip() or (
        'developer-id' if identity and identity != '-' else 'adhoc')
    if mode == 'adhoc':
        return '-', None
    if mode != 'developer-id':
        raise ValueError('HANZI_MAC_SIGNING 必须为 adhoc 或 developer-id')
    if not identity.startswith('Developer ID Application:'):
        raise ValueError('developer-id 模式需要 APPLE_SIGN_IDENTITY')
    profile = os.environ.get('APPLE_NOTARY_PROFILE', '').strip()
    if not profile:
        raise ValueError('developer-id 模式需要 APPLE_NOTARY_PROFILE')
    return identity, profile


def sign_macos(app, identity):
    def sign(path):
        options = (['--timestamp=none', '--options', '0'] if identity == '-'
                   else ['--timestamp', '--options', 'runtime'])
        subprocess.run(['codesign', '--force', *options, '--sign', identity, str(path)], check=True)
    # Sign inside-out, without following framework aliases or using --deep to sign.
    magic = {b'\xcf\xfa\xed\xfe', b'\xfe\xed\xfa\xcf', b'\xca\xfe\xba\xbe', b'\xbe\xba\xfe\xca'}
    paths = sorted(app.rglob('*'), key=lambda p: len(p.parts), reverse=True)
    # Signing a bundle's main executable also signs its enclosing bundle. Leave
    # that executable to the bundle step, after all its libraries are signed.
    main_executables = set()
    for bundle in [app, *(p for p in paths if p.suffix in {'.app', '.xpc'} and p.is_dir() and not p.is_symlink())]:
        info = bundle / 'Contents/Info.plist'
        if info.is_file():
            name = plistlib.loads(info.read_bytes()).get('CFBundleExecutable')
            if name:
                main_executables.add(bundle / 'Contents/MacOS' / name)
    for path in paths:
        if path.is_symlink() or path in main_executables:
            continue
        if path.is_file():
            with path.open('rb') as file:
                is_macho = file.read(4) in magic
            if is_macho:
                sign(path)
        elif path.suffix in {'.framework', '.app', '.xpc'}:
            sign(path)
    sign(app)
    subprocess.run(['codesign', '--verify', '--deep', '--strict', str(app)], check=True)


def package_macos(root, app):
    bundle_version(app)
    config = config_for(root)
    if config is None:
        return False
    arch = os.environ.get('MAC_ARCH', 'arm64')
    if arch not in {'arm64', 'x86_64'}:
        raise ValueError('Unsupported macOS architecture')
    config = verify_built_config(root, app / f'Contents/Resources/runtime/{arch}/Runtime.app/Contents/Resources/native_update.json')
    identity, profile = mac_signing_config()
    framework = Path(os.environ['SPARKLE_FRAMEWORK']).resolve()
    bridge = Path(os.environ['SPARKLE_BRIDGE']).resolve()
    destination = app / 'Contents/Frameworks'
    destination.mkdir(exist_ok=True)
    subprocess.run(['ditto', str(framework), str(destination / 'Sparkle.framework')], check=True)
    shutil.copy2(bridge, destination / 'SparkleBridge.dylib')
    shutil.copy2(framework.parent / 'LICENSE', app / 'Contents/Resources/Sparkle-LICENSE.txt')
    plist_path = app / 'Contents/Info.plist'
    data = plistlib.loads(plist_path.read_bytes())
    data.update(SUFeedURL=config.feed_url, SUPublicEDKey=config.public_key,
                SUEnableAutomaticChecks=False, SUAllowsAutomaticUpdates=False,
                SUVerifyUpdateBeforeExtraction=True, SURelaunchHostBundle=True)
    plist_path.write_bytes(plistlib.dumps(data))
    sign_macos(app, identity)
    archive = root / 'dist' / f'{root.name}-{config.version}-macos-{arch}.zip'
    archive.unlink(missing_ok=True)
    subprocess.run(['ditto', '-c', '-k', '--sequesterRsrc', '--keepParent', str(app), str(archive)], check=True)
    if profile:
        subprocess.run(['xcrun', 'notarytool', 'submit', str(archive), '--keychain-profile', profile, '--wait'], check=True)
        subprocess.run(['xcrun', 'stapler', 'staple', str(app)], check=True)
        subprocess.run(['spctl', '--assess', '--type', 'execute', str(app)], check=True)
        archive.unlink()
        subprocess.run(['ditto', '-c', '-k', '--sequesterRsrc', '--keepParent', str(app), str(archive)], check=True)
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['windows', 'macos'])
    parser.add_argument('--root', type=Path, default=Path.cwd())
    parser.add_argument('--app', type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    if root.name not in PRODUCTS:
        parser.error('未知客户端仓库')
    if args.command == 'windows':
        package_windows(root)
    else:
        if args.app is None:
            parser.error('macos 需要 --app')
        package_macos(root, args.app.resolve())


if __name__ == '__main__':
    main()
