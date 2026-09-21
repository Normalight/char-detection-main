import importlib.util
from pathlib import Path
from unittest.mock import Mock
import pytest

spec = importlib.util.spec_from_file_location('publish_gitcode', Path(__file__).resolve().parents[1] / 'packaging/publish_gitcode.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def test_create_timeout_queries_before_retry():
    p = m.Publisher('owner','repo','private-token')
    release = {'tag_name':'v1.0.0'}
    p.release = Mock(side_effect=[None, release])
    p.api = Mock(side_effect=m.APIError('network'))
    assert p.ensure_release('v1.0.0','title','body') == release
    assert p.api.call_count == 1


def test_existing_attachment_does_not_reupload(tmp_path):
    p = m.Publisher('owner','repo','private-token')
    f = tmp_path/'app.zip'
    p.release = Mock(return_value={'assets':[{'name':f.name}]})
    p.api = Mock()
    p.upload('v1.0.0',f)
    p.api.assert_not_called()


def test_upload_keeps_auth_off_storage(monkeypatch, tmp_path):
    p = m.Publisher('owner','repo','private-token')
    f = tmp_path/'app.zip'
    f.write_bytes(b'archive')
    p.release = Mock(return_value={'assets':[]})
    headers = {'x-obs-callback':'callback','Content-Type':'application/zip'}
    p.api = Mock(return_value={'url':'https://storage.example/upload','headers':headers})
    put = Mock(return_value=Mock(status_code=200))
    monkeypatch.setattr(m.requests,'put',put)
    p.upload('v1.0.0',f)
    assert put.call_args.kwargs['headers'] == headers
    assert 'Authorization' not in put.call_args.kwargs['headers']
    assert put.call_args.kwargs['allow_redirects'] is False


def test_auth_failure_is_not_treated_as_missing_release():
    p = m.Publisher('owner','repo','private-token')
    p.api = Mock(side_effect=m.APIError(403))
    with pytest.raises(m.APIError):
        p.ensure_release('v1.0.0','title','body')
    assert p.api.call_count == 1


def test_standalone_sync_reads_explicit_release_event(monkeypatch, tmp_path):
    import json
    import sys
    event = tmp_path / 'release-event.json'
    event.write_text(json.dumps({'release': {'tag_name': 'v1.0.0', 'name': '1.0.0', 'body': 'notes'}}))
    files = []
    for target in ('windows-x64', 'macos-arm64', 'macos-x86_64'):
        f = tmp_path / ('工具-v1.0.0-' + target + '.zip')
        f.write_bytes(b'test')
        files.append(f)
    client = Mock()
    client.release.return_value = {'assets': [
        {'name': f.name, 'browser_download_url': 'https://gitcode.com/release/' + f.name}
        for f in files] + [{'name': 'hanzi-update.json'}]}
    monkeypatch.setattr(m, 'Publisher', Mock(return_value=client))
    monkeypatch.setattr(sys, 'argv', ['publish', '--event', str(event), '--assets', str(tmp_path)])
    for key in ('GITCODE_OWNER', 'GITCODE_REPO', 'GITCODE_TOKEN'):
        monkeypatch.setenv(key, 'test-value')
    monkeypatch.setenv('GITHUB_EVENT_PATH', str(tmp_path / 'unused-dispatch-event.json'))
    monkeypatch.delenv('GITHUB_STEP_SUMMARY', raising=False)
    monkeypatch.chdir(tmp_path)
    m.main()
    client.ensure_release.assert_called_once_with('v1.0.0', '1.0.0', 'notes')
    assert client.upload.call_count == 4
    assert json.loads((tmp_path / 'hanzi-update.json').read_text()) == json.loads((tmp_path / 'gitcode-release.json').read_text())
    records = json.loads((tmp_path / 'gitcode-release.json').read_text())
    assert len(records) == 3
    assert all(record['url'].isascii() for record in records)


def test_upload_reader_preserves_content_length(tmp_path):
    f = tmp_path / 'file.zip'
    f.write_bytes(b'archive fixture')
    with f.open('rb') as stream:
        reader = m.UploadReader(stream, f.name)
        request = m.requests.Request('PUT', 'https://storage.example/upload', data=reader).prepare()
        assert request.headers['Content-Length'] == str(f.stat().st_size)
        assert 'Transfer-Encoding' not in request.headers
        assert reader.read() == b'archive fixture'
