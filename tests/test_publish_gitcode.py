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
