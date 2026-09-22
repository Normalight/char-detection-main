"""Annotation and Excel exports must discover identical current/legacy paths."""
import os

import pytest

from utils.paths import get_json_candidates, get_output_dir_for_item
from utils.excel import get_item_folder_and_prefix
from utils.storage import get_json_path_for_item


@pytest.mark.parametrize('item,parts,prefix', [
    ({'type': 'image', 'source_path': '/输入/碑文.png'}, ['碑文'], '碑文'),
    ({'type': 'pdf', 'source_path': '/输入/古籍.pdf', 'page_index': 2},
     ['古籍', 'page_003'], '古籍_p003'),
])
def test_exports_share_paths_and_legacy_lookup(tmp_path, item, parts, prefix):
    folder = os.path.join(str(tmp_path), *parts)
    assert get_output_dir_for_item(tmp_path, item) == (folder, prefix)
    assert get_item_folder_and_prefix(tmp_path, item) == (folder, prefix)
    candidates = get_json_candidates(tmp_path, item)
    assert candidates[0] == get_json_path_for_item(tmp_path, item)
    assert os.path.join(folder, os.path.basename(folder) + '.json') in candidates
    assert len(candidates) == len(set(candidates))
