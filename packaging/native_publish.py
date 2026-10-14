"""Verify signed native builds and publish them to the public distribution repository.

Upload order matters: immutable version installers first, then the stable feed
files that point at them. Verify public GitCode bytes before marking publication
complete, including when an attachment already existed.
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
from urllib.parse import quote, urlsplit
from urllib.request import Request, urlopen
import xml.etree.ElementTree as ET

NS = 'http://www.andymatuschak.org/xml-namespaces/sparkle'
ET.register_namespace('sparkle', NS)
TARGETS = {'windows-appcast.xml': 'windows-x64', 'macos-appcast.xml': 'macos-arm64',
           'macos-x86_64-appcast.xml': 'macos-x86_64'}

from hanzi_update.core import PRODUCT_TARGETS

def product_feeds(product, selected=None):
    feeds = {name: target for name, target in TARGETS.items()
             if target in PRODUCT_TARGETS.get(product, ())}
    if selected and selected != 'all':
        feeds = {name: target for name, target in feeds.items() if target == selected}
        if not feeds:
            raise PublishError('该软件不支持此发布平台')
    return feeds



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


def verify_appcasts(product, version, assets, config, target=None):
    """Every feed item must point at this version's uploaded file and verify locally."""
    from nacl.signing import VerifyKey
    feed_base = config['feed_base']
    if urlsplit(feed_base).scheme != 'https':
        raise PublishError('更新源必须为 HTTPS')
    base = feed_base.rsplit('/', 1)[0]
    version_url = f'{base}/{product}-{version}'
    feeds = product_feeds(product, target)
    expected = {f'{product}-{version}-{target}' + ('.exe' if target == 'windows-x64' else '.zip')
                for target in feeds.values()}
    for name, target in feeds.items():
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
        raise PublishError('各平台 feed 未覆盖全部安装包')


def verify_remote_asset(remote, path):
    if remote.get('size') != Path(path).stat().st_size or remote.get('digest') != 'sha256:' + digest(path):
        raise PublishError(f'公开仓已有同名附件 {Path(path).name} 且内容不同；请发布新版本号')


def upload_asset(client, repo, tag, path, release_id, clobber=False):
    """Upload with retry; an existing identical asset is accepted, a different one is not."""
    for attempt in range(4):
        assets = {a['name']: a for a in client.assets(release_id)}
        if path.name in assets and not clobber:
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
            verify_remote_asset(current[path.name], path)
            return
        if attempt == 3:
            raise PublishError(f'附件上传后未确认：{path.name}')
        time.sleep(2)


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


def gitcode_destination(feed_base, product):
    parts = urlsplit(feed_base)
    segments = parts.path.strip('/').split('/')
    if (parts.scheme != 'https' or parts.hostname != 'gitcode.com' or parts.username
            or parts.password or parts.query or parts.fragment or len(segments) != 5
            or segments[2:4] != ['releases', 'download']
            or segments[4] != product + '-stable'):
        raise PublishError('GitCode 更新源配置无效')
    return segments[0], segments[1], segments[4]


def verify_gitcode_file(url, path):
    """Verify public bytes without forwarding either publishing credential."""
    import requests
    parts = urlsplit(url)
    if (parts.scheme != 'https' or not parts.hostname or parts.username or parts.password
            or not (parts.hostname == 'gitcode.com' or parts.hostname.endswith('.gitcode.com'))):
        raise PublishError('GitCode 返回了不可信的附件地址')
    expected = Path(path).stat().st_size
    try:
        with requests.get(url, stream=True, timeout=(15, 120)) as response:
            if (response.status_code != 200 or urlsplit(response.url).scheme != 'https'):
                raise PublishError('GitCode 附件尚未可公开下载：' + Path(path).name)
            sha, count = hashlib.sha256(), 0
            for chunk in response.iter_content(1024 * 1024):
                count += len(chunk)
                if count > expected:
                    raise PublishError('GitCode 附件大小不符：' + Path(path).name)
                sha.update(chunk)
    except requests.RequestException:
        raise PublishError('GitCode 公开下载失败：' + Path(path).name) from None
    if (count, sha.hexdigest()) != (expected, digest(path)):
        raise PublishError('GitCode 附件内容不符：' + Path(path).name)


def public_gitcode_asset(publisher, tag, path, *, replace=False, verify=verify_gitcode_file):
    """Publish one asset and verify its publicly served bytes before returning."""
    path = Path(path)
    for attempt in range(6):
        release = publisher.release(tag)
        if release is None:
            time.sleep(2)
            continue
        matches = [item for item in release.get('assets', []) if item.get('name') == path.name]
        if len(matches) > 1:
            raise PublishError('GitCode 存在多个同名附件：' + path.name)
        if matches:
            url = matches[0]['browser_download_url']
            try:
                verify(url, path)
                return url
            except PublishError:
                if not replace:
                    raise
                # GitCode's attachment API has no clobber operation.
                publisher.api('DELETE', '/releases/' + quote(tag, safe='')
                              + '/attach_files/' + str(int(matches[0]['id'])))
        if attempt == 5:
            break
        publisher.upload(tag, path)
        time.sleep(2)
    raise PublishError('GitCode 附件上传后未通过公开校验：' + path.name)


def gitcode_appcast(source, filename, url, output):
    feed = ET.parse(source)
    items = feed.findall('.//item')
    if len(items) != 1:
        raise PublishError('原生更新源必须只有一个版本条目')
    item = items[0]
    enclosure = item.find('enclosure')
    if enclosure is None or enclosure.get('url', '').rsplit('/', 1)[-1] != filename:
        raise PublishError('原生更新源与 GitCode 安装包不匹配')
    for tag in ('releaseNotesLink', 'fullReleaseNotesLink',
                'releaseNotesURL', 'fullReleaseNotesURL'):
        for element in item.findall(f'{{{NS}}}{tag}'):
            item.remove(element)
    enclosure.set('url', url)
    feed.write(output, encoding='utf-8', xml_declaration=True)


def publish_gitcode(product, version, config, assets, notes, publisher=None,
                    verify=verify_gitcode_file, target=None):
    owner, repo, stable_tag = gitcode_destination(config['gitcode_feed_base'], product)
    if publisher is None:
        from publish_gitcode import Publisher
        publisher = Publisher(owner, repo, os.environ['GITCODE_TOKEN'])
    client = publisher
    client.api('GET')  # Fail before creating any GitCode release if access is absent.
    tag = product + '-' + version
    client.ensure_release(tag, tag, notes.read_text(encoding='utf-8'))
    feeds = product_feeds(product, target)
    urls = {}
    for target in feeds.values():
        path = assets / (f'{tag}-{target}' + ('.exe' if target == 'windows-x64' else '.zip'))
        urls[target] = public_gitcode_asset(client, tag, path, verify=verify)
    with tempfile.TemporaryDirectory(prefix='gitcode-native-feeds-') as directory:
        staged = Path(directory)
        for name, target in feeds.items():
            filename = f'{tag}-{target}' + ('.exe' if target == 'windows-x64' else '.zip')
            gitcode_appcast(assets / name, filename, urls[target], staged / name)
        client.ensure_release(stable_tag, stable_tag, '稳定更新源')
        for name in feeds:
            public_gitcode_asset(client, stable_tag, staged / name, replace=True, verify=verify)
            # Check the exact URL embedded in new clients, not only the API's asset URL.
            verify(config['gitcode_feed_base'] + '/' + name, staged / name)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--product', required=True)
    parser.add_argument('--version', required=True)
    parser.add_argument('--assets', type=Path, required=True, help='包含各平台安装包与对应 appcast 的目录')
    parser.add_argument('--notes', type=Path, required=True, help='该版本中文更新说明文件')
    parser.add_argument('--source-commit', default=os.environ.get('GITHUB_SHA', ''))
    parser.add_argument('--target', choices=['all', *TARGETS.values()], default='all')
    parser.add_argument('--github-only', action='store_true', help='只发布 GitHub，由平台镜像器完成 GitCode 同步')
    parser.add_argument('--verify-only', action='store_true', help='只做本地校验，不上传')
    args = parser.parse_args()

    config = json.loads((Path('packaging/native_update_public.json')).read_text(encoding='utf-8'))
    if config.get('product') != args.product:
        raise PublishError('公开更新配置属于另一款软件')
    if not args.notes.is_file():
        raise PublishError('缺少该版本的中文更新说明文件')
    version = 'v' + split_version(args.version)
    feeds = product_feeds(args.product, args.target)
    installers = [(args.assets / (f'{args.product}-{version}-{target}' +
                   ('.exe' if target == 'windows-x64' else '.zip')), target)
                  for target in feeds.values()]
    for path, target in installers:
        if not path.is_file():
            raise PublishError(f'缺少安装包 {path.name}')

    verify_appcasts(args.product, version, args.assets, config, args.target)
    if args.verify_only:
        print('校验通过：各平台安装包签名与 feed 一致')
        return

    if not args.github_only and not os.getenv('GITCODE_TOKEN'):
        raise PublishError('需要 GITCODE_TOKEN 才能发布客户端使用的 GitCode 更新源')
    from publish_public import GitHub, PRODUCTS
    from publish_gitcode import Publisher

    gitcode = None
    if not args.github_only:
        owner, gitcode_repo, _ = gitcode_destination(config['gitcode_feed_base'], args.product)
        gitcode = Publisher(owner, gitcode_repo, os.environ['GITCODE_TOKEN'])
        gitcode.api('GET')

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
        for path, target in installers:
            records.append({'name': path.name, 'target': target, 'size': path.stat().st_size,
                            'sha256': digest(path)})
        suffix = '' if args.target == 'all' else '-' + args.target
        sums_name = 'SHA256SUMS' + suffix + '.txt'
        manifest_name = 'release-manifest' + suffix + '.json'
        notes_file = args.notes
        if suffix:
            notes_file = stage / (args.notes.stem + suffix + '.md')
            notes_file.write_bytes(args.notes.read_bytes())
        (stage / sums_name).write_text(
            ''.join(item['sha256'] + '  ' + item['name'] + '\n' for item in records), encoding='utf-8')
        (stage / manifest_name).write_text(json.dumps({
            'schema': 1, 'product': args.product, 'version': version,
            'source_commit': args.source_commit, 'assets': records,
        }, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
        files = [path for path, _ in installers] + [notes_file, stage / sums_name, stage / manifest_name]
        for path in files:
            upload_asset(client, repository, tag, path, release['id'])
        final = {a['name'] for a in client.assets(release['id'])}
        expected_files = {path.name for path in files}
        if (final != expected_files if args.target == 'all' else not expected_files.issubset(final)):
            raise PublishError('版本 Release 附件不完整，请人工核对后重试')
        client.call('release', 'edit', tag, '--repo', repository, '--notes-file', str(args.notes))
        if release['draft']:
            client.call('release', 'edit', tag, '--repo', repository, '--draft=false')

    stable_tag = f'{args.product}-stable'
    notes = stage_stable_notes(stable_tag)
    stable = ensure_release(client, repository, stable_tag, PRODUCTS[args.product] + ' 稳定更新源', notes, draft=False)
    for name in feeds:
        upload_asset(client, repository, stable_tag, args.assets / name, stable['id'], clobber=True)

    verify_public_urls([config['feed_base'] + '/' + name for name in feeds] +
                       [f'{base}/{tag}/{path.name}' for path, _ in installers])
    if not args.github_only:
        publish_gitcode(args.product, version, config, args.assets, args.notes, publisher=gitcode, target=args.target)
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
