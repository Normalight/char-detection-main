"""Build specs must only reference configuration that exists in the checkout."""
import ast
from pathlib import Path

import config
import pytest


@pytest.mark.parametrize('filename', ['app.spec', 'packaging/mac.spec'])
def test_build_spec_config_references_resolve(filename):
    tree = ast.parse(Path(filename).read_text(encoding='utf-8'))
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
            if node.value.id in {'config', 'app_config'}:
                assert hasattr(config, node.attr), f'{filename}: missing config.{node.attr}'
