import importlib.util
from pathlib import Path
from unittest.mock import Mock
import pytest

spec = importlib.util.spec_from_file_location('next_version', Path(__file__).resolve().parents[1] / 'packaging/next_version.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


@pytest.mark.parametrize('tags,bump,expected', [
    ([], 'patch', 'v1.0.0'),
    (['v1.0.0'], 'patch', 'v1.0.1'),
    (['v1.9.9', 'v1.10.0'], 'patch', 'v1.10.1'),
    (['v1.10.9'], 'minor', 'v1.11.0'),
    (['v1.10.9'], 'major', 'v2.0.0'),
    (['v1.2.0', 'v2.0.0-rc.1', 'dev-abc', 'README'], 'patch', 'v1.2.1'),
])
def test_increment(tags, bump, expected):
    assert m.next_version(tags, bump) == expected


def test_remote_versions_include_only_matching_public_product(monkeypatch):
    run = Mock(side_effect=[Mock(stdout='v1.0.0\n'), Mock(stdout='v1.0.1\n'),
                            Mock(stdout='repo-v1.1.0\nother-v9.0.0\n')])
    monkeypatch.setattr(m.subprocess, 'run', run)
    tags = m.remote_tags('owner/repo', None, None)
    assert m.next_version(tags) == 'v1.1.1'
    assert run.call_count == 3
