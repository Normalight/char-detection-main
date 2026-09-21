"""Exercise the native restart helper on each release runner."""
import base64
from pathlib import Path
import shlex
import subprocess
import sys
import time

import pytest
from hanzi_auth.updates import replacement_command


@pytest.mark.skipif(sys.platform not in {'darwin', 'win32'}, reason='Desktop release platforms only')
def test_native_update_waits_for_old_process_then_swaps(tmp_path):
    current = tmp_path / "汉字 [app] ' test"
    current.mkdir()
    (current / 'version').write_text('old')
    (current / 'user-note.txt').write_text('saved work')
    source = tmp_path / '.hanzi-update-test' / 'new'
    source.mkdir(parents=True)
    (source / 'version').write_text('new')
    (source / 'group_by_character.exe').touch()
    marker = tmp_path / 'restarted'
    old = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'])
    helper = None
    try:
        target = 'windows-x64' if sys.platform == 'win32' else 'macos-arm64'
        command = replacement_command(source, current, target, old.pid)
        if sys.platform == 'win32':
            script = base64.b64decode(command[-1]).decode('utf-16-le')
            script = '\n'.join("  Set-Content -LiteralPath '" + str(marker).replace("'", "''") + "' -Value restarted"
                               if line.strip().startswith('Start-Process ') else line for line in script.splitlines())
            command[-1] = base64.b64encode(script.encode('utf-16-le')).decode('ascii')
        else:
            path = Path(command[1])
            script = '\n'.join('/usr/bin/touch ' + shlex.quote(str(marker))
                               if line.startswith('/usr/bin/open ') else line for line in path.read_text().splitlines())
            path.write_text(script)
        helper = subprocess.Popen(command)
        time.sleep(.3)
        assert helper.poll() is None
        assert (current / 'version').read_text() == 'old'
        assert not marker.exists()
        old.terminate()
        old.wait(timeout=10)
        assert helper.wait(timeout=20) == 0
        assert (current / 'version').read_text() == 'new'
        assert (tmp_path / (current.name + '.previous-test') / 'version').read_text() == 'old'
        assert marker.exists()
        if sys.platform == 'win32':
            assert (current / 'user-note.txt').read_text() == 'saved work'
    finally:
        for process in (old, helper):
            if process is not None and process.poll() is None:
                process.kill()
                process.wait(timeout=10)
