"""Build first-install DMGs/Setup EXEs alongside the existing in-app update ZIPs."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import time

PRODUCTS = {
    'char-detection-main': ('中文字符标注工具', 'interactive_annotator', '切字'),
    'group-by-character': ('字形归并工具', 'group_by_character', '归并'),
}


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def identity(product, version, target):
    if product not in PRODUCTS or target not in {'macos-arm64', 'macos-x86_64', 'macos-universal', 'windows-x64'}:
        raise ValueError('Unknown product or target')
    if not re.fullmatch(r'v\d+\.\d+\.\d+(?:-[A-Za-z0-9.-]+)?|dev-[0-9a-f]{7,40}', version):
        raise ValueError('Invalid version')
    return f'{PRODUCTS[product][2]}-{version}-{target}'


def validate_bundle(bundle, product, version, target):
    name, exe, _ = PRODUCTS[product]
    entry = bundle / ('Contents/MacOS/launcher' if target.startswith('macos-') else exe + '.exe')
    if not entry.is_file():
        raise ValueError('Application entry is missing')
    records = [p for p in bundle.rglob('build_info.json') if not p.is_symlink()]
    if not records:
        raise ValueError('Build metadata is missing')
    for path in records:
        info = json.loads(path.read_text(encoding='utf-8'))
        if (info.get('product'), info.get('version')) != (product, version):
            raise ValueError('Bundle identity differs from installer')
        if target != 'macos-universal' and info.get('target') != target:
            raise ValueError('Bundle architecture differs from installer')


def background(path, title):
    from PIL import Image, ImageDraw, ImageFont
    font_path = '/System/Library/Fonts/Supplemental/Arial Unicode.ttf'
    if not Path(font_path).is_file():
        font_path = '/System/Library/Fonts/PingFang.ttc'
    image = Image.new('RGB', (1320, 800), '#f6f5f1')
    draw = ImageDraw.Draw(image)
    def text(y, value, size, fill):
        draw.text((660, y), value, font=ImageFont.truetype(font_path, size), fill=fill, anchor='mt')
    text(50, '安装 ' + title, 48, '#24352e')
    text(128, '将软件拖到右侧的“应用程序”文件夹', 28, '#68756e')
    draw.line((600, 380, 720, 380), fill='#6e8e7b', width=6)
    draw.line((697, 357, 720, 380, 697, 403), fill='#6e8e7b', width=6)
    text(674, '安装完成后，从“应用程序”中打开软件。', 26, '#68756e')
    image.save(path, dpi=(144, 144))


def finder_layout(mount, app_name):
    from ds_store import DSStore
    from mac_alias import Alias
    bg = mount / '.background'
    bg.mkdir()
    background(bg / 'install.png', app_name.removesuffix('.app'))
    with DSStore.open(str(mount / '.DS_Store'), 'w+') as store:
        store['.']['vSrn'] = ('long', 1)
        store['.']['bwsp'] = {'ShowStatusBar': False, 'ShowToolbar': False,
                             'ShowPathbar': False, 'ShowSidebar': False,
                             'WindowBounds': '{{180, 180}, {660, 400}}', 'SidebarWidth': 0}
        store['.']['icvp'] = {'viewOptionsVersion': 1, 'backgroundType': 2,
                             'backgroundImageAlias': Alias.for_file(str((bg / 'install.png').resolve())).to_bytes(),
                             'iconSize': 96.0, 'textSize': 13.0, 'gridSpacing': 100.0,
                             'gridOffsetX': 0.0, 'gridOffsetY': 0.0, 'arrangeBy': 'none',
                             'scrollPositionX': 0.0, 'scrollPositionY': 0.0,
                             'backgroundColorRed': 1.0, 'backgroundColorGreen': 1.0, 'backgroundColorBlue': 1.0,
                             'labelOnBottom': True, 'showIconPreview': False, 'showItemInfo': False}
        store['.']['icvl'] = ('type', b'icnv')
        store[app_name]['Iloc'] = (170, 185)
        store['Applications']['Iloc'] = (490, 185)


def build_dmg(bundle, output):
    if sys.platform != 'darwin':
        raise RuntimeError('DMGs must be built on macOS')
    with tempfile.TemporaryDirectory(prefix='hanzi-dmg-') as temporary:
        temp = Path(temporary)
        stage = temp / 'contents'
        stage.mkdir()
        subprocess.run(['ditto', str(bundle), str(stage / bundle.name)], check=True)
        (stage / 'Applications').symlink_to('/Applications')
        writable = temp / 'installer.dmg'
        subprocess.run(['hdiutil', 'create', '-quiet', '-fs', 'HFS+', '-format', 'UDRW',
                        '-volname', '安装 ' + bundle.stem, '-srcfolder', str(stage), str(writable)], check=True)
        mount = temp / 'mount'
        mount.mkdir()
        subprocess.run(['hdiutil', 'attach', '-quiet', '-nobrowse', '-noautoopen',
                        '-mountpoint', str(mount), str(writable)], check=True)
        try:
            finder_layout(mount, bundle.name)
        finally:
            subprocess.run(['hdiutil', 'detach', '-quiet', str(mount)], check=True)
        subprocess.run(['hdiutil', 'convert', '-quiet', str(writable), '-format', 'UDZO',
                        '-imagekey', 'zlib-level=9', '-o', str(output)], check=True)
    subprocess.run(['hdiutil', 'verify', '-quiet', str(output)], check=True)


def inno_script(bundle, output, product, version):
    name, executable, _ = PRODUCTS[product]
    # Inputs originate from the validated build configuration; quote literal paths.
    def literal(value):
        return str(value).replace('"', '""').replace('{', '{{')
    bundle_path, output_dir = literal(bundle.resolve()), literal(output.parent.resolve())
    numeric = version[1:].split('-')[0] if version.startswith('v') else '0.0.0'
    return f'''[Setup]
AppId=HanziTools.{product}
AppName={name}
AppVersion={version}
VersionInfoVersion={numeric}.0
AppPublisher=HanziTools
DefaultDirName={{localappdata}}\\Programs\\HanziTools\\{product}
DefaultGroupName={name}
PrivilegesRequired=lowest
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
MinVersion=10.0
DisableDirPage=no
DisableProgramGroupPage=yes
UsePreviousAppDir=yes
WizardStyle=modern
OutputDir="{output_dir}"
OutputBaseFilename={output.stem}
Compression=lzma2
SolidCompression=yes
UninstallDisplayIcon={{app}}\\{executable}.exe
AppMutex=Local\\HanziTools.{product}
SetupMutex=Local\\HanziTools.Setup.{product}
CloseApplications=yes
RestartApplications=no

[Languages]
Name: "chinesesimp"; MessagesFile: "compiler:Languages\\ChineseSimplified.isl"
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "{{cm:CreateDesktopIcon}}"; Flags: unchecked

[Files]
Source: "{bundle_path}\\*"; DestDir: "{{app}}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{{autoprograms}}\\{name}"; Filename: "{{app}}\\{executable}.exe"; WorkingDir: "{{app}}"
Name: "{{autodesktop}}\\{name}"; Filename: "{{app}}\\{executable}.exe"; WorkingDir: "{{app}}"; Tasks: desktopicon

[Run]
Filename: "{{app}}\\{executable}.exe"; Description: "{{cm:LaunchProgram,{name}}}"; WorkingDir: "{{app}}"; Flags: nowait postinstall skipifsilent

[UninstallDelete]
Type: filesandordirs; Name: "{{app}}\\_internal"
'''


def build_windows(bundle, output, product, version):
    if sys.platform != 'win32':
        raise RuntimeError('Windows installers must be built on Windows')
    compiler = os.getenv('ISCC') or shutil.which('ISCC')
    if not compiler:
        for base in ('ProgramFiles', 'ProgramFiles(x86)'):
            candidate = Path(os.getenv(base, 'C:/Program Files')) / 'Inno Setup 7/ISCC.exe'
            if candidate.is_file():
                compiler = str(candidate)
                break
    if not compiler:
        raise RuntimeError('Install Inno Setup 7 or set ISCC to ISCC.exe')
    script = output.parent / (output.stem + '.iss')
    script.write_text(inno_script(bundle, output, product, version), encoding='utf-8-sig')
    try:
        subprocess.run([compiler, '/Qp', str(script)], check=True)
    finally:
        script.unlink(missing_ok=True)


def smoke_windows(installer, product):
    """Install, run and uninstall in an isolated current-user directory."""
    name, executable, _ = PRODUCTS[product]
    with tempfile.TemporaryDirectory(prefix='hanzi-setup-smoke-') as directory:
        destination = Path(directory) / '应用 程序'
        subprocess.run([str(installer), '/VERYSILENT', '/SUPPRESSMSGBOXES', '/NORESTART', '/SP-',
                        '/DIR=' + str(destination), '/GROUP=HanziInstallerSmoke',
                        '/TASKS=', '/NOICONS'], check=True, timeout=180)
        try:
            subprocess.run([str(destination / (executable + '.exe')), '--smoke-test'],
                           cwd=destination, check=True, timeout=240)
        finally:
            subprocess.run([str(destination / 'unins000.exe'), '/VERYSILENT', '/SUPPRESSMSGBOXES',
                            '/NORESTART'], check=True, timeout=120)
        deadline = time.monotonic() + 60
        while (destination / (executable + '.exe')).exists() or (destination / '_internal').exists():
            if time.monotonic() >= deadline:
                raise RuntimeError('Uninstaller left application binaries behind')
            time.sleep(0.2)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--product', required=True, choices=PRODUCTS)
    parser.add_argument('--version', required=True)
    parser.add_argument('--target', required=True)
    parser.add_argument('--bundle', required=True, type=Path)
    parser.add_argument('--archive', required=True, type=Path)
    parser.add_argument('--smoke-test', action='store_true')
    args = parser.parse_args()
    stem = identity(args.product, args.version, args.target)
    validate_bundle(args.bundle, args.product, args.version, args.target)
    output = args.archive.parent / (stem + ('.dmg' if args.target.startswith('macos-') else '-Setup.exe'))
    output.unlink(missing_ok=True)
    if args.target.startswith('macos-'):
        build_dmg(args.bundle, output)
    else:
        build_windows(args.bundle, output, args.product, args.version)
        if args.smoke_test:
            smoke_windows(output, args.product)
    record = dict(product=args.product, version=args.version, target=args.target,
                  name=output.name, size=output.stat().st_size, sha256=digest(output),
                  update_sha256=digest(args.archive))
    output.with_suffix(output.suffix + '.json').write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding='utf-8')
    print('Installer verified:', output)


if __name__ == '__main__':
    main()
