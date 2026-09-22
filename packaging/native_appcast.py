"""Create signed native-update feeds locally; does not upload anything."""
import argparse
import base64
import os
from pathlib import Path
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


def windows_appcast(config, archive, url, signature, notes, output):
    from nacl.signing import VerifyKey
    url = https_url(url)
    expected = f'{config.product}-v{config.version.removeprefix("v")}-windows-x64-setup.exe'
    if archive.name != expected:
        raise ValueError('安装包名称与软件/版本不匹配')
    content = archive.read_bytes()
    if not content.startswith(b'MZ'):
        raise ValueError('需要 Windows 安装器 EXE')
    # Verify the official signer's output against the exact shipped public key.
    # Never sign or modify the EXE after this stage (including Authenticode).
    VerifyKey(base64.b64decode(config.public_key, validate=True)).verify(
        content, base64.b64decode(signature, validate=True))
    rss = ET.Element('rss', {'version': '2.0'})
    channel = ET.SubElement(rss, 'channel')
    ET.SubElement(channel, 'title').text = config.product + ' ' + config.channel
    item = ET.SubElement(channel, 'item')
    ET.SubElement(item, 'title').text = config.version
    ET.SubElement(item, 'description').text = notes
    ET.SubElement(item, f'{{{NS}}}version').text = config.build_version
    ET.SubElement(item, f'{{{NS}}}shortVersionString').text = config.version.removeprefix('v')
    ET.SubElement(item, 'enclosure', {'url': url, 'length': str(len(content)),
        'type': 'application/octet-stream', f'{{{NS}}}edSignature': signature,
        f'{{{NS}}}os': 'windows-x64', f'{{{NS}}}installerArguments': '/SILENT /SP- /NOICONS'})
    output.parent.mkdir(parents=True, exist_ok=True)
    ET.indent(rss)
    temporary = output.with_suffix('.xml.tmp')
    ET.ElementTree(rss).write(temporary, encoding='utf-8', xml_declaration=True)
    temporary.replace(output)


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
    mac.add_argument('--download-url-prefix', required=True)
    mac.add_argument('--account', required=True, help='Sparkle signing keychain account name')
    args = parser.parse_args()
    if args.platform == 'windows':
        config = UpdateConfig(args.product, args.version, args.feed, args.public_key, args.channel)
        windows_appcast(config, args.archive, args.url, args.signature_file.read_text().strip(),
                        args.notes.read_text(encoding='utf-8'), args.output)
    else:
        subprocess.run([str(args.tool.resolve()), '-o', str(args.archives.resolve() / 'appcast.xml'),
            '--account', args.account,
            '--maximum-deltas', '0', '--download-url-prefix', https_url(args.download_url_prefix),
            str(args.archives.resolve())], check=True)
        feed = ET.parse(args.archives / 'appcast.xml')
        enclosures = feed.findall('.//enclosure')
        if not enclosures or any(not e.get(f'{{{NS}}}edSignature') for e in enclosures):
            raise ValueError('Sparkle appcast 缺少更新包签名')
        for enclosure in enclosures:
            https_url(enclosure.attrib['url'])


if __name__ == '__main__':
    main()
