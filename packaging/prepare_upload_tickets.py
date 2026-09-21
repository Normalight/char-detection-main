"""Issue short-lived upload tickets for a local publisher; never export the API token."""
import json
import os
from pathlib import Path
from urllib.parse import urlsplit
from publish_gitcode import Publisher

publisher = Publisher(os.environ['GITCODE_OWNER'], os.environ['GITCODE_REPO'], os.environ['GITCODE_TOKEN'])
event = json.loads(Path('release-event.json').read_text())['release']
tag = event['tag_name']
repo = os.environ['GITHUB_REPOSITORY'].split('/')[-1]
prefix = '归并' if repo == 'group-by-character' else '切字'
records = []
for asset in event['assets']:
    name = prefix + asset['name'] if asset['name'].startswith('-') else asset['name']
    if not name.endswith('.zip'):
        continue
    ticket = publisher.api('GET', '/releases/' + tag + '/upload_url', params={'file_name': name})
    if urlsplit(ticket['url']).scheme != 'https':
        raise RuntimeError('HTTPS upload required')
    records.append({'name': name, 'size': asset['size'], 'digest': asset.get('digest'), 'ticket': ticket})
text = json.dumps(records, ensure_ascii=False)
if publisher.token in text:
    raise RuntimeError('Refusing to export an API token')
path = Path('upload-tickets.json')
path.write_text(text, encoding='utf-8')
path.chmod(0o600)
print('Prepared scoped upload tickets:', len(records))
