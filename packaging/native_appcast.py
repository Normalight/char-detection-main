"""Create signed native-update feeds locally; does not upload anything."""
import argparse
import base64
import os
from pathlib import Path
import re
import subprocess
from urllib.parse import urlsplit
import xml.etree.ElementTree as ET
from hanzi_update.backends.base import UpdateConfig

NS = 'http://www.andymatuschak.org/xml-namespaces/sparkle'
ET.register_namespace('sparkle', NS)


def https_url(value):
    parts = urlsplit(value)
    if (parts.scheme != 'https' or not parts.hostname or parts.username or parts.password
            or parts.query or parts.fragment or any(c.isspace() for c in value)):
        raise ValueError('发布 URL 必须为不含凭据的 HTTPS 地址')
    return value


def signature_value(text):
    """Accept bare base64 from winsparkle-tool or an attribute line from sign_update."""
    text = (text or '').strip()
    if 'edSignature' in text:
        match = re.search(r'edSignature="([^"]+)"', text)
        if not match:
            raise ValueError('签名文件格式无效')
        text = match.group(1).strip()
    if not text or any(c.isspace() for c in text):
        raise ValueError('签名文件格式无效')
    base64.b64decode(text, validate=True)
    return text


MAX_DESCRIPTION = 2000


def plain_notes(text):
    """Render the Chinese markdown release note as plain dialog text.

    原生更新弹窗直接显示纯文本说明，不提供任何外链，避免把用户引向
    列出全部历史版本的发行版页面。
    """
    lines = []
    for line in (text or '').splitlines():
        line = line.strip()
        if not line:
            lines.append('')
            continue
        line = re.sub(r'^#{1,6}\s*', '', line)
        line = re.sub(r'`([^`]*)`', r'\1', line)
        line = re.sub(r'\*\*([^*]+)\*\*', r'\1', line)
        line = re.sub(r'\*([^*]+)\*', r'\1', line)
        line = re.sub(r'\[([^\]]+)\]\([^)]*\)', r'\1', line)
        line = re.sub(r'^[-*+]\s+', '· ', line)
        lines.append(line)
    merged = '\n'.join(lines)
    merged = re.sub(r'\n{3,}', '\n\n', merged).strip()
    if len(merged) > MAX_DESCRIPTION:
        merged = merged[:MAX_DESCRIPTION - 1].rstrip() + '…'
    return merged


def inline_notes(feed, notes):
    """Replace external release-note links with an inline plain-text description."""
    for item in feed.findall('.//item'):
        for tag in ('releaseNotesLink', 'fullReleaseNotesLink',
                    'releaseNotesURL', 'fullReleaseNotesURL'):
            for element in item.findall(f'{{{NS}}}{tag}'):
                item.remove(element)
        if notes is not None:
            description = item.find('description')
            if description is None:
                description = ET.SubElement(item, 'description')
            description.text = plain_notes(notes)
            description.set(f'{{{NS}}}format', 'plain-text')


def verify_enclosure_signature(path, signature, public_key):
    """Re-verify the official signer's output against the exact shipped public key."""
    from nacl.signing import VerifyKey
    VerifyKey(base64.b64decode(public_key, validate=True)).verify(
        Path(path).read_bytes(), base64.b64decode(signature, validate=True))


def windows_appcast(config, archive, url, signature, notes, output):
    url = https_url(url)
    expected = f'{config.product}-v{config.version.removeprefix("v")}-windows-x64.exe'
    if archive.name != expected:
        raise ValueError('安装包名称与软件/版本不匹配')
    if not archive.read_bytes().startswith(b'MZ'):
        raise ValueError('需要 Windows 安装器 EXE')
    verify_enclosure_signature(archive, signature, config.public_key)
    rss = ET.Element('rss', {'version': '2.0'})
    channel = ET.SubElement(rss, 'channel')
    ET.SubElement(channel, 'title').text = config.product + ' ' + config.channel
    item = ET.SubElement(channel, 'item')
    ET.SubElement(item, 'title').text = config.version
    ET.SubElement(item, 'description', {f'{{{NS}}}format': 'plain-text'}).text = plain_notes(notes)
    ET.SubElement(item, f'{{{NS}}}version').text = config.build_version
    ET.SubElement(item, f'{{{NS}}}shortVersionString').text = config.version.removeprefix('v')
    ET.SubElement(item, 'enclosure', {'url': url, 'length': str(archive.stat().st_size),
        'type': 'application/octet-stream', f'{{{NS}}}edSignature': signature,
        f'{{{NS}}}os': 'windows-x64', f'{{{NS}}}installerArguments': '/SILENT /SP- /NOICONS'})
    output.parent.mkdir(parents=True, exist_ok=True)
    ET.indent(rss)
    temporary = output.with_suffix('.xml.tmp')
    ET.ElementTree(rss).write(temporary, encoding='utf-8', xml_declaration=True)
    temporary.replace(output)


def macos_appcast(args):
    if bool(args.account) == bool(args.ed_key_file):
        raise ValueError('macos 需要 --account（钥匙串）或 --ed-key-file（密钥文件）其中之一')
    command = [str(args.tool.resolve()), '-o', str(args.archives.resolve() / 'appcast.xml')]
    if args.ed_key_file:
        command += ['--ed-key-file', str(Path(args.ed_key_file).resolve())]
    else:
        command += ['--account', args.account]
    command += ['--maximum-deltas', '0', '--download-url-prefix', https_url(args.download_url_prefix),
                str(args.archives.resolve())]
    subprocess.run(command, check=True)
    feed = ET.parse(args.archives / 'appcast.xml')
    enclosures = feed.findall('.//enclosure')
    if not enclosures or any(not e.get(f'{{{NS}}}edSignature') for e in enclosures):
        raise ValueError('Sparkle appcast 缺少更新包签名')
    for enclosure in enclosures:
        https_url(enclosure.attrib['url'])
        if args.public_key:
            archive = args.archives / enclosure.attrib['url'].rsplit('/', 1)[-1]
            verify_enclosure_signature(archive, enclosure.get(f'{{{NS}}}edSignature'), args.public_key)
    notes = args.notes.read_text(encoding='utf-8') if args.notes else None
    inline_notes(feed, notes)
    feed.write(args.archives / 'appcast.xml', encoding='utf-8', xml_declaration=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='platform', required=True)
    win = sub.add_parser('windows')
    win.add_argument('archive', type=Path)
    win.add_argument('--product', required=True)
    win.add_argument('--version', default=os.environ.get('PACKAGE_VERSION'))
    win.add_argument('--feed', default=os.environ.get('HANZI_UPDATE_FEED'))
    win.add_argument('--public-key', default=os.environ.get('HANZI_UPDATE_PUBLIC_KEY'))
    win.add_argument('--channel', choices=['stable', 'beta'], default='stable')
    win.add_argument('--url', required=True)
    win.add_argument('--signature-file', type=Path, required=True)
    win.add_argument('--notes', type=Path, required=True)
    win.add_argument('--output', type=Path, required=True)
    mac = sub.add_parser('macos')
    mac.add_argument('archives', type=Path)
    mac.add_argument('--tool', type=Path, required=True)
    mac.add_argument('--account', help='Sparkle signing keychain account name')
    mac.add_argument('--ed-key-file', help='私钥文件路径；CI 自动签名使用，不读钥匙串')
    mac.add_argument('--notes', type=Path, help='该版本中文更新说明；以内嵌纯文本展示，不生成外链')
    mac.add_argument('--public-key', default=os.environ.get('HANZI_UPDATE_PUBLIC_KEY'),
                     help='复验 appcast 签名的公钥')
    mac.add_argument('--download-url-prefix', required=True)
    args = parser.parse_args()
    if args.platform == 'windows':
        config = UpdateConfig(args.product, args.version, args.feed, args.public_key, args.channel)
        windows_appcast(config, args.archive, args.url,
                        signature_value(args.signature_file.read_text()),
                        args.notes.read_text(encoding='utf-8'), args.output)
    else:
        macos_appcast(args)


if __name__ == '__main__':
    main()
