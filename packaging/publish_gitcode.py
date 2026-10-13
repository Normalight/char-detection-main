"""Publish existing Actions artifacts to GitCode without rebuilding them."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time
from urllib.parse import quote, urlsplit
import requests


class APIError(RuntimeError):
    def __init__(self, status):
        self.status = status
        super().__init__(f"GitCode API returned HTTP {status}")


class UploadReader:
    """Keep a known Content-Length and report progress without logging signed URLs."""
    mode = 'rb'

    def __init__(self, stream, label):
        self.stream = stream
        self.label = label
        self.total = os.fstat(stream.fileno()).st_size
        self.reported = 0
        self.started = time.monotonic()

    def __len__(self):
        return self.total

    def tell(self):
        return self.stream.tell()

    def read(self, size=-1):
        chunk = self.stream.read(size)
        done = self.stream.tell()
        if done - self.reported >= 4 * 1024 * 1024 or not chunk:
            elapsed = max(time.monotonic() - self.started, 0.001)
            print(f'Upload {self.label}: {done}/{self.total} bytes, {done / elapsed / 1024:.0f} KiB/s', flush=True)
            self.reported = done
        return chunk


class Publisher:
    def __init__(self, owner, repo, token):
        self.base = f"https://api.gitcode.com/api/v5/repos/{quote(owner, safe='')}/{quote(repo, safe='')}"
        self.token = token

    def api(self, method, path='', **kwargs):
        for attempt in range(4):
            try:
                response = requests.request(method, self.base + path,
                    headers={"Authorization": "Bearer " + self.token},
                    timeout=(30, 120), allow_redirects=False, **kwargs)
            except requests.RequestException:
                if method != 'GET' or attempt == 3:
                    raise APIError('network') from None
            else:
                if response.status_code < 300:
                    return response.json() if response.content else None
                if method != 'GET' or response.status_code not in (429, 500, 502, 503, 504) or attempt == 3:
                    raise APIError(response.status_code)
            time.sleep(2 ** attempt)

    def release(self, tag):
        try:
            return self.api('GET', '/releases/tags/' + quote(tag, safe=''))
        except APIError as exc:
            if exc.status == 404:
                return None
            raise

    def ensure_release(self, tag, title, body):
        for attempt in range(4):
            existing = self.release(tag)
            if existing is not None:
                return existing
            try:
                return self.api('POST', '/releases', json={
                    'tag_name': tag, 'name': title, 'body': body,
                    'release_status': 'latest'})
            except APIError as exc:
                # A timed-out POST might have succeeded; query before another POST.
                existing = self.release(tag)
                if existing is not None:
                    return existing
                if exc.status not in ('network', 409, 429, 500, 502, 503, 504) or attempt == 3:
                    raise
                time.sleep(2 ** attempt)

    def upload(self, tag, file):
        for attempt in range(4):
            existing = self.release(tag)
            if any(asset.get('name') == file.name for asset in (existing or {}).get('assets', [])):
                print('Existing attachment retained:', file.name)
                return
            print('Requesting upload ticket:', file.name, flush=True)
            ticket = self.api('GET', '/releases/' + quote(tag, safe='') + '/upload_url',
                              params={'file_name': file.name})
            url = ticket['url']
            if urlsplit(url).scheme != 'https':
                raise RuntimeError('Upload URL must use HTTPS')
            print('Uploading bytes:', file.stat().st_size, 'to', urlsplit(url).hostname, flush=True)
            try:
                # Separate request: never forward the GitCode bearer to object storage.
                with file.open('rb') as data:
                    result = requests.put(url, headers=ticket['headers'], data=UploadReader(data, file.name),
                                          timeout=(30, 120), allow_redirects=False)
                if 200 <= result.status_code < 300:
                    print('Uploaded:', file.name)
                    return
                print('Storage response:', result.status_code, flush=True)
            except requests.RequestException as exc:
                print('Storage transport:', type(exc).__name__, flush=True)
            if attempt == 3:
                raise RuntimeError('Attachment upload failed: ' + file.name)
            time.sleep(2 ** attempt)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--preflight', action='store_true')
    parser.add_argument('--event', type=Path, help='Saved release event for standalone synchronization')
    parser.add_argument('--assets', type=Path, default=Path('release-assets'))
    args = parser.parse_args()
    owner, repo, token = (os.environ.get(key, '') for key in
                         ('GITCODE_OWNER', 'GITCODE_REPO', 'GITCODE_TOKEN'))
    if not all((owner, repo, token)):
        raise RuntimeError('Configure GITCODE_OWNER, GITCODE_REPO and GITCODE_TOKEN')
    publisher = Publisher(owner, repo, token)
    publisher.api('GET')
    if args.preflight:
        print('GitCode repository access verified:', owner + '/' + repo)
        return
    event = json.loads((args.event or Path(os.environ['GITHUB_EVENT_PATH'])).read_text())
    release = event['release']
    tag = release['tag_name']
    files = sorted(args.assets.rglob('*.zip'))
    targets = ('windows-x64', 'macos-arm64')
    if any(f.name.endswith('-macos-x86_64.zip') for f in files):
        targets += ('macos-x86_64',)
    if len(files) != len(targets) or not all(sum(f.name.endswith('-' + t + '.zip') for f in files) == 1 for t in targets):
        raise RuntimeError('Expected exactly one installer for each release platform')
    body = release.get('body') or ''
    publisher.ensure_release(tag, release.get('name') or tag, body)
    for file in files:
        publisher.upload(tag, file)
    for attempt in range(6):
        published = publisher.release(tag)
        assets = {a['name']: a for a in published.get('assets', [])}
        if all(f.name in assets for f in files):
            break
        time.sleep(5)
    else:
        raise RuntimeError('GitCode attachment listing did not confirm every upload')
    records = []
    for file in files:
        with file.open('rb') as stream:
            digest = hashlib.file_digest(stream, 'sha256').hexdigest()
        # Stable public API link, no signed upload URL or credential in output.
        url = quote(assets[file.name]['browser_download_url'], safe=':/?&=%+@')
        host = urlsplit(url).hostname or ''
        if urlsplit(url).scheme != 'https' or not (host == 'gitcode.com' or host.endswith('.gitcode.com')):
            raise RuntimeError('GitCode returned an unexpected public download host')
        records.append({'name': file.name, 'version': tag, 'size': file.stat().st_size,
                        'sha256': digest, 'url': url})
    Path('gitcode-release.json').write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding='utf-8')
    manifest = Path('hanzi-update.json')
    manifest.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding='utf-8')
    publisher.upload(tag, manifest)
    for attempt in range(6):
        published = publisher.release(tag)
        if any(a.get('name') == manifest.name for a in published.get('assets', [])):
            break
        time.sleep(5)
    else:
        raise RuntimeError('GitCode update manifest upload was not confirmed')
    summary = os.environ.get('GITHUB_STEP_SUMMARY')
    if summary:
        with open(summary, 'a', encoding='utf-8') as out:
            out.write('## GitCode ' + tag + '\n\n')
            for item in records:
                out.write(f"- [{item['name']}]({item['url']}) — {item['size']} bytes; SHA-256 `{item['sha256']}`\n")
    print('GitCode release verified:', owner + '/' + repo, tag)


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        # Requests errors can contain signed URLs; never print their raw text.
        print(str(exc) if isinstance(exc, (APIError, RuntimeError)) else
              'Publication failed (' + type(exc).__name__ + ')', file=sys.stderr)
        sys.exit(1)
