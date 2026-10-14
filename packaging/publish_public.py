"""Publish verified desktop ZIPs to the public distribution repository, without rebuilding."""
import argparse
import hashlib
import json
import logging
import os
from pathlib import Path
import re
import shutil
import stat
import subprocess
import tempfile
import time
import zipfile

LOGGER = logging.getLogger(__name__)
PUBLIC_REPO = 'Normalight/hanzi-soft'
from hanzi_update.core import PRODUCT_LABELS as PRODUCTS, PRODUCT_TARGETS
TARGETS = {'windows-x64', 'macos-arm64', 'macos-x86_64'}


class PublishError(RuntimeError):
    pass


def digest(path):
    value = hashlib.sha256()
    with path.open('rb') as stream:
        while chunk := stream.read(1024 * 1024):
            value.update(chunk)
    return value.hexdigest()


def stage_assets(directory, stage, product, version, source_commit):
    if product not in PRODUCTS or not re.fullmatch(r'v\d+\.\d+\.\d+(?:-[A-Za-z0-9.-]+)?', version):
        raise PublishError('Invalid product or version')
    if not re.fullmatch('[0-9a-f]{40}', source_commit):
        raise PublishError('Expected the source commit SHA')
    selected = {}
    for path in Path(directory).glob('*.zip'):
        target = next((t for t in TARGETS if path.name.endswith(f'-{version}-{t}.zip')), None)
        if not target or target in selected:
            raise PublishError('Unexpected or duplicate platform ZIP')
        if path.is_symlink() or not path.is_file():
            raise PublishError('Expected a regular ZIP file')
        with zipfile.ZipFile(path) as archive:
            entries = [i for i in archive.infolist() if i.filename.split('/')[-1] == 'build_info.json'
                       and not stat.S_ISLNK(i.external_attr >> 16)]
            if not entries or len(entries) > 16:
                raise PublishError('Missing build_info.json; rebuild the installer')
            for entry in entries:
                if entry.file_size > 4096:
                    raise PublishError('Invalid build metadata')
                info = json.loads(archive.read(entry))
                if (info.get('product'), info.get('version'), info.get('target')) != (product, version, target):
                    raise PublishError('Installer identity does not match the release')
        selected[target] = path
    required = set(PRODUCT_TARGETS[product])
    if set(selected) != required:
        raise PublishError('Wait for all required platform ZIPs')
    stage = Path(stage)
    stage.mkdir(parents=True, exist_ok=True)
    assets = []
    for target, source in sorted(selected.items()):
        destination = stage / f'{product}-{version}-{target}.zip'
        shutil.copyfile(source, destination)
        assets.append(dict(name=destination.name, target=target, size=destination.stat().st_size,
                           sha256=digest(destination)))
    manifest = dict(schema=1, product=product, version=version, source_commit=source_commit, assets=assets)
    (stage/'release-manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    (stage/'SHA256SUMS.txt').write_text(''.join(a['sha256']+'  '+a['name']+'\n' for a in assets), encoding='utf-8')
    return manifest


class GitHub:
    def call(self, *args, missing_ok=False):
        result = subprocess.run(['gh', *args], capture_output=True, text=True, timeout=900)
        if result.returncode:
            if missing_ok and 'HTTP 404' in result.stderr:
                return None
            # Never echo CLI arguments, environment, signed URLs or response bodies.
            raise PublishError('GitHub request failed; check PUBLIC_RELEASE_TOKEN permissions and network')
        return result.stdout

    def api(self, path, missing_ok=False):
        result = self.call('api', path, missing_ok=missing_ok)
        return json.loads(result) if result is not None else None

    def release(self, tag):
        published = self.api(f'repos/{PUBLIC_REPO}/releases/tags/{tag}', missing_ok=True)
        if published is not None:
            return published
        # The tag endpoint does not return a draft before its tag is published.
        # Authenticated release listings include drafts and let retries reuse them.
        for page in range(1, 101):
            releases = self.api(f'repos/{PUBLIC_REPO}/releases?per_page=100&page={page}')
            matches = [r for r in releases if r['tag_name'] == tag]
            if len(matches) > 1:
                raise PublishError('Multiple release drafts share this tag; review them manually')
            if matches:
                return matches[0]
            if len(releases) < 100:
                return None
        raise PublishError('Release listing is too large to safely locate the draft')

    def assets(self, release_id):
        return self.api(f'repos/{PUBLIC_REPO}/releases/{release_id}/assets?per_page=100')


def verify_remote(remote, expected):
    if remote.get('size') != expected['size'] or remote.get('digest') != 'sha256:' + expected['sha256']:
        raise PublishError('An existing public attachment differs; publish a new version instead')


def publish(client, stage, manifest):
    repo = client.api('repos/' + PUBLIC_REPO)
    if repo.get('private') is not False:
        raise PublishError('The distribution repository must be public')
    product, version = manifest['product'], manifest['version']
    tag = product + '-' + version
    release = client.release(tag)
    if release is None:
        notes = Path(stage) / 'public-notes.md'
        notes.write_text(f"{PRODUCTS[product]} {version}\n\n提供 Windows x64 和 macOS Apple Silicon 安装包。\n\n"
                         '下载后可通过 SHA256SUMS.txt 校验文件。\n', encoding='utf-8')
        try:
            client.call('release', 'create', tag, '--repo', PUBLIC_REPO, '--draft',
                        '--target', repo['default_branch'], '--title', PRODUCTS[product]+' '+version,
                        '--notes-file', str(notes))
        except PublishError:
            if client.release(tag) is None:
                raise
        for attempt in range(6):
            release = client.release(tag)
            if release is not None:
                break
            time.sleep(2)
        if release is None:
            raise PublishError('New draft is not visible yet; rerun publication to reuse it')
    files = [Path(stage)/a['name'] for a in manifest['assets']]
    files += [Path(stage)/'SHA256SUMS.txt', Path(stage)/'release-manifest.json']
    expected = {f.name: dict(size=f.stat().st_size, sha256=digest(f)) for f in files}
    remote = {a['name']: a for a in client.assets(release['id'])}
    if not release['draft'] and set(remote) != set(expected):
        raise PublishError('Published release is incomplete or has unexpected assets; repair it manually')
    if set(remote) - set(expected):
        raise PublishError('Unexpected assets in the public release; review the draft')
    for file in files:
        if file.name in remote:
            verify_remote(remote[file.name], expected[file.name])
            continue
        for attempt in range(3):
            try:
                client.call('release', 'upload', tag, str(file), '--repo', PUBLIC_REPO)
                break
            except PublishError:
                current = {a['name']: a for a in client.assets(release['id'])}
                if file.name in current:
                    verify_remote(current[file.name], expected[file.name])
                    break
                if attempt == 2:
                    raise
                time.sleep(2)
    remote = {a['name']: a for a in client.assets(release['id'])}
    if set(remote) != set(expected):
        raise PublishError('Public attachment set is incomplete')
    for name, value in expected.items():
        verify_remote(remote[name], value)
    if release['draft']:
        args = ['release', 'edit', tag, '--repo', PUBLIC_REPO, '--draft=false']
        if '-' in version:
            args += ['--prerelease']
        client.call(*args)
    return f'https://github.com/{PUBLIC_REPO}/releases/tag/{tag}'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--product', required=True, choices=sorted(PRODUCTS))
    parser.add_argument('--version', required=True)
    parser.add_argument('--source-commit', required=True)
    parser.add_argument('--assets', type=Path, default=Path('release-assets'))
    args = parser.parse_args()
    if not os.getenv('GH_TOKEN'):
        raise PublishError('PUBLIC_RELEASE_TOKEN is required as GH_TOKEN')
    with tempfile.TemporaryDirectory(prefix='public-installers-') as stage:
        manifest = stage_assets(args.assets, stage, args.product, args.version, args.source_commit)
        url = publish(GitHub(), stage, manifest)
    Path('public-release.json').write_text(json.dumps({'url': url, 'manifest': manifest}, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    LOGGER.info('Published %s', url)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    main()
