"""Annotation and Excel exports must discover identical current/legacy paths."""
import os
import json
from pathlib import Path

import pytest
import numpy as np
from PIL import Image

from utils.paths import get_json_candidates, get_output_dir_for_item, assign_image_output_names
from utils.excel import get_item_folder_and_prefix
from utils.storage import get_json_path_for_item
from utils.storage import save_current_slices_and_json, load_item_annotation_record_if_exists
from utils.excel import build_excel_for_single_item, load_boxes_from_json


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


def test_same_stem_images_keep_separate_pixels_annotations_and_workbooks(tmp_path):
    box = np.asarray([[0, 0], [9, 0], [9, 9], [0, 9]], dtype=np.float32)
    items = [{'type': 'image', 'source_path': f'/input/page.{ext}'} for ext in ('jpg', 'png')]
    assign_image_output_names(items)
    assert [it['output_stem'] for it in items] == ['page_jpg', 'page_png']
    for index, item in enumerate(items):
        save_current_slices_and_json(str(tmp_path), item,
                                    np.full((20, 20, 3), index * 255, np.uint8),
                                    [box + index])
        build_excel_for_single_item(str(tmp_path), item)
    for index, item in enumerate(items):
        folder = Path(get_output_dir_for_item(str(tmp_path), item)[0])
        with Image.open(next(folder.glob('*_crop_*.png'))) as crop:
            assert crop.getpixel((0, 0)) == (index * 255,) * 3
        record = load_item_annotation_record_if_exists(str(tmp_path), item)
        np.testing.assert_array_equal(record['boxes'][0], box + index)
        assert (folder / 'crops.xlsx').is_file()
    assert get_json_path_for_item(str(tmp_path), items[0]) != get_json_path_for_item(str(tmp_path), items[1])


@pytest.mark.parametrize('saved_source', ['/old/input/page.jpg', r'C:\input\page.jpg'])
def test_legacy_annotations_are_only_read_for_matching_source(tmp_path, saved_source):
    legacy = tmp_path / 'page' / 'page.json'
    legacy.parent.mkdir()
    boxes = [[[0, 0], [9, 0], [9, 9], [0, 9]]]
    legacy.write_text(json.dumps({'source_path': saved_source, 'boxes': boxes}))
    matching = {'type': 'image', 'source_path': '/moved/page.jpg'}
    other = {'type': 'image', 'source_path': '/moved/page.png'}
    assign_image_output_names([matching, other])
    assert load_item_annotation_record_if_exists(str(tmp_path), matching)['json_path'] == str(legacy)
    assert load_boxes_from_json(str(tmp_path), matching)[0] == boxes
    assert load_item_annotation_record_if_exists(str(tmp_path), other) is None
    assert load_boxes_from_json(str(tmp_path), other) == ([], None)
    # A new save takes precedence over its legacy annotation, including empty boxes.
    save_current_slices_and_json(str(tmp_path), matching, np.zeros((10, 10, 3), np.uint8), [])
    assert load_item_annotation_record_if_exists(str(tmp_path), matching)['boxes'] == []
    assert legacy.is_file()


def test_legacy_without_source_requires_unambiguous_input(tmp_path):
    source = tmp_path / 'input'
    source.mkdir()
    (source / 'page.jpg').touch()
    output = tmp_path / 'output'
    legacy = output / 'page' / 'page.json'
    legacy.parent.mkdir(parents=True)
    legacy.write_text('{"boxes": []}')
    item = {'type': 'image', 'source_path': str(source / 'page.jpg'), 'output_stem': 'page_jpg'}
    assert load_item_annotation_record_if_exists(str(output), item) is not None
    (source / 'page.png').touch()
    assert load_item_annotation_record_if_exists(str(output), item) is None


def test_suffixes_do_not_collide_with_ordinary_stems(tmp_path):
    items = [{'type': 'image', 'source_path': str(tmp_path / name)}
             for name in ('page.jpg', 'page.png', 'page_jpg.png', 'other.png')]
    assign_image_output_names(items)
    names = [Path(get_output_dir_for_item(str(tmp_path / 'out'), item)[0]).name for item in items]
    assert names == ['page_jpg_2', 'page_png', 'page_jpg', 'other']
    assert 'output_stem' not in items[2] and 'output_stem' not in items[3]


def test_discovery_assigns_collision_names(tmp_path):
    from annotation.controller import AnnotatorCore
    for name in ('page.jpg', 'page.png', 'other.png'):
        (tmp_path / name).touch()
    core = AnnotatorCore()
    core.input_dir = str(tmp_path)
    items = core._collect_input_items()
    assert {Path(it['source_path']).name: it.get('output_stem') for it in items} == {
        'page.jpg': 'page_jpg', 'page.png': 'page_png', 'other.png': None}
