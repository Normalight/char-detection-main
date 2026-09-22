"""Verify signed native builds and publish them to the public distribution repository.

Upload order matters: immutable version installers first, then the stable feed
files that point at them. Existing version attachments are never replaced with
different content; the stable appcast is atomically clobbered instead.
"""
import argparse
import base64
import hashlib
import json
import os
import re
import sys
import tempfile
import time
from pathlib import Path
from urllib.parse import urlsplit
from urllib.request import Request, urlopen
import xml.etree.ElementTree as ET

NS = 'http://www.andymatuschak.org/xml-namespaces/sparkle'
TARGETS = {'windows-appcast.xml': 'windows-x64', 'macos-appcast.xml': 'macos-arm64'}


class PublishError(RuntimeError):
    pass


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        while chunk := stream.read(1024 * 1024):
            value.update(chunk)
    return value.hexdigest()


def split_version(value):
    version = value.removeprefix('v')
    if not re.fullmatch(r'(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)', version):
        raise PublishError('原生发布需要 vMAJOR.MINOR.PATCH 正式版本')
    return version


def verify_appcasts(product, version, assets, config):
    """Every feed item must point at this version's uploaded file and verify locally."""
    from nacl.signing import VerifyKey
    feed_base = config['feed_base']
    if urlsplit(feed_base).scheme != 'https':
        raise PublishError('更新源必须为 HTTPS')
    base = feed_base.rsplit('/', 1)[0]
    version_url = f'{base}/{product}-{version}'
    expected = {f'{product}-{version}-windows-x64.exe', f'{product}-{version}-macos-arm64.zip'}
    for name, target in TARGETS.items():
        appcast = assets / name
        if not appcast.is_file():
            raise PublishError(f'缺少 {name}；请用 publish 输入重新构建以生成 feed')
        items = ET.parse(appcast).findall('.//item')
        if len(items) != 1:
            raise PublishError(f'{name} 必须只包含一个更新条目')
        enclosure = items[0].find('enclosure')
        if enclosure is None:
            raise PublishError(f'{name} 缺少 enclosure')
        filename = enclosure.get('url', '').rsplit('/', 1)[-1]
        if (enclosure.get('url') != f'{version_url}/{filename}' or filename not in expected
                or TARGETS[name] not in filename):
            raise PublishError(f'{name} 的下载地址与本次发布不符')
        local = assets / filename
        if not local.is_file():
            raise PublishError(f'缺少 feed 指向的安装包 {filename}')
        if enclosure.get('length') != str(local.stat().st_size):
            raise PublishError(f'{name} 的大小记录与安装包不符')
        signature = enclosure.get(f'{{{NS}}}edSignature')
        if not signature:
            raise PublishError(f'{name} 缺少更新包签名')
        try:
            VerifyKey(base64.b64decode(config['public_key'], validate=True)).verify(
                local.read_bytes(), base64.b64decode(signature, validate=True))
        except Exception as exc:
            raise PublishError(f'{name} 签名复验失败，请核对签名密钥与客户端公钥配对') from exc
        expected.discard(filename)
    if expected:
        raise PublishError('两个平台 feed 未覆盖全部安装包')


def verify_remote_asset(remote, path):
    if remote.get('size') != Path(path).stat().st_size or remote.get('digest') != 'sha256:' + digest(path):
        raise PublishError(f'公开仓已有同名附件 {Path(path).name} 且内容不同；请发布新版本号')


def upload_asset(client, repo, tag, path, release_id, clobber=False):
    """Upload with retry; an existing identical asset is accepted, a different one is not."""
    for attempt in range(4):
        assets = {a['name']: a for a in client.assets(release_id)}
        if path.name in assets:
            if clobber:
                break
            verify_remote_asset(assets[path.name], path)
            return
        args = ['release', 'upload', tag, str(path), '--repo', repo]
        if clobber:
            args.append('--clobber')
        try:
            client.call(*args)
        except PublishError:
            current = {a['name']: a for a in client.assets(release_id)}
            if path.name in current:
                verify_remote_asset(current[path.name], path)
                return
            if attempt == 3:
                raise
            time.sleep(2)
            continue
        current = {a['name']: a for a in client.assets(release_id)}
        if path.name in current:
            if not clobber:
                verify_remote_asset(current[path.name], path)
            return
        if attempt == 3:
            raise PublishError(f'附件上传后未确认：{path.name}')
        time.sleep(2)
    if clobber:
        current = {a['name']: a for a in client.assets(release_id)}
        if path.name not in current:
            raise PublishError(f'附件上传后未确认：{path.name}')


def ensure_release(client, repo, tag, title, notes_path, draft):
    release = client.release(tag)
    if release is not None:
        return release
    args = ['release', 'create', tag, '--repo', repo, '--target', client.api('repos/' + repo)['default_branch'],
            '--title', title, '--notes-file', str(notes_path)]
    if draft:
        args.append('--draft')
    try:
        client.call(*args)
    except PublishError:
        if client.release(tag) is None:
            raise
    for _ in range(6):
        release = client.release(tag)
        if release is not None:
            return release
        time.sleep(2)
    raise PublishError(f'Release {tag} 创建后不可见，请重试以复用')


def verify_public_urls(urls):
    for url in urls:
        request = Request(url, method='HEAD', headers={'User-Agent': 'hanzi-publish-check'})
        with urlopen(request, timeout=30) as response:
            if response.status != 200 or urlsplit(response.geturl()).scheme != 'https':
                raise PublishError(f'公开下载验证失败：{url}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--product', required=True)
    parser.add_argument('--version', required=True)
    parser.add_argument('--assets', type=Path, required=True, help='包含两平台安装包与两个 appcast 的目录')
    parser.add_argument('--notes', type=Path, required=True, help='该版本中文更新说明文件')
    parser.add_argument('--source-commit', default=os.environ.get('GITHUB_SHA', ''))
    parser.add_argument('--verify-only', action='store_true', help='只做本地校验，不上传')
    args = parser.parse_args()

    config = json.loads((Path('packaging/native_update_public.json')).read_text(encoding='utf-8'))
    if config.get('product') != args.product:
        raise PublishError('公开更新配置属于另一款软件')
    if not args.notes.is_file():
        raise PublishError('缺少该版本的中文更新说明文件')
    version = 'v' + split_version(args.version)
    exe = args.assets / f'{args.product}-{version}-windows-x64.exe'
    archive = args.assets / f'{args.product}-{version}-macos-arm64.zip'
    for path in (exe, archive):
        if not path.is_file():
            raise PublishError(f'缺少安装包 {path.name}')

    verify_appcasts(args.product, version, args.assets, config)
    if args.verify_only:
        print('校验通过：两平台安装包签名与 feed 一致')
        return

    if not os.getenv('GH_TOKEN'):
        raise PublishError('需要 PUBLIC_RELEASE_TOKEN 作为 GH_TOKEN')
    from publish_public import GitHub, PRODUCTS

    client = GitHub()
    parts = urlsplit(config['feed_base'])
    repository = '/'.join(parts.path.strip('/').split('/')[:2])
    info = client.api('repos/' + repository)
    if info.get('private') is not False:
        raise PublishError('公开分发仓库必须是 public 仓库')
    base = config['feed_base'].rsplit('/', 1)[0]

    tag = f'{args.product}-{version}'
    release = ensure_release(client, repository, tag, f"{PRODUCTS[args.product]} {version}", args.notes, draft=True)
    with tempfile.TemporaryDirectory(prefix='native-publish-') as stage:
        stage = Path(stage)
        records = []
        for path, target in ((exe, 'windows-x64'), (archive, 'macos-arm64')):
            records.append({'name': path.name, 'target': target, 'size': path.stat().st_size,
                            'sha256': digest(path)})
        (stage / 'SHA256SUMS.txt').write_text(
            ''.join(item['sha256'] + '  ' + item['name'] + '\n' for item in records), encoding='utf-8')
        (stage / 'release-manifest.json').write_text(json.dumps({
            'schema': 1, 'product': args.product, 'version': version,
            'source_commit': args.source_commit, 'assets': records,
        }, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
        files = [exe, archive, args.notes, stage / 'SHA256SUMS.txt', stage / 'release-manifest.json']
        for path in files:
            upload_asset(client, repository, tag, path, release['id'])
        final = {a['name'] for a in client.assets(release['id'])}
        if final != {path.name for path in files}:
            raise PublishError('版本 Release 附件不完整，请人工核对后重试')
        if release['draft']:
            client.call('release', 'edit', tag, '--repo', repository, '--draft=false')

    stable_tag = f'{args.product}-stable'
    notes = stage_stable_notes(stable_tag)
    stable = ensure_release(client, repository, stable_tag, PRODUCTS[args.product] + ' 稳定更新源', notes, draft=False)
    for name in TARGETS:
        upload_asset(client, repository, stable_tag, args.assets / name, stable['id'], clobber=True)

    verify_public_urls([
        config['feed_base'] + '/windows-appcast.xml', config['feed_base'] + '/macos-appcast.xml',
        f'{base}/{tag}/{exe.name}', f'{base}/{tag}/{archive.name}',
    ])
    url = f'https://github.com/{repository}/releases/tag/{tag}'
    Path('public-release.json').write_text(json.dumps({'url': url, 'product': args.product,
        'version': version, 'stable_feed': stable_tag}, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print('已发布：', url)


def stage_stable_notes(stable_tag):
    descriptor, name = tempfile.mkstemp(prefix='stable-notes-', suffix='.md')
    os.close(descriptor)
    notes = Path(name)
    notes.write_text(f'{stable_tag}：Windows 与 macOS 稳定更新 appcast（windows-appcast.xml / macos-appcast.xml）。\n',
                     encoding='utf-8')
    return notes


if __name__ == '__main__':
    try:
        main()
    except PublishError as exc:
        print(str(exc), file=sys.stderr)
        sys.exit(1)
