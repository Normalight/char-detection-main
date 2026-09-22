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
    source = current / '.hanzi-update-test' / 'new'
    source.mkdir(parents=True)
    (source / 'version').write_text('new')
    (source / 'group_by_character.exe').touch()
    marker = tmp_path / 'restarted'
    old = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'])
    helper = None
    locked = None
    if sys.platform == 'win32':
        # A normal open handle denies deletion on Windows. Release it after exit
        # to verify that a transient file lock is retried, not reported as success.
        old_executable = current / 'group_by_character.exe'
        old_executable.write_bytes(b'old executable')
        locked = old_executable.open('rb')
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
        helper = subprocess.Popen(command, cwd=current)
        time.sleep(.3)
        assert helper.poll() is None
        assert (current / 'version').read_text() == 'old'
        assert not marker.exists()
        old.terminate()
        old.wait(timeout=10)
        if locked is not None:
            time.sleep(.5)
            assert helper.poll() is None
            assert locked.read() == b'old executable'
            locked.close()
        assert helper.wait(timeout=20) == 0
        assert (current / 'version').read_text() == 'new'
        assert (source.parent / 'previous' / 'version').read_text() == 'old'
        assert marker.exists()
        if sys.platform == 'win32':
            assert (current / 'user-note.txt').read_text() == 'saved work'
    finally:
        if locked is not None:
            locked.close()
        for process in (old, helper):
            if process is not None and process.poll() is None:
                process.kill()
                process.wait(timeout=10)


@pytest.mark.skipif(sys.platform != 'win32', reason='Native Windows rollback')
@pytest.mark.parametrize('with_runtime', [False, True])
def test_windows_failed_replacement_restores_old_program(tmp_path, with_runtime):
    current = tmp_path / "汉字 [app] ' rollback"
    current.mkdir()
    executable = current / 'group_by_character.exe'
    executable.write_bytes(b'old executable')
    (current / 'user-note.txt').write_text('keep')
    source = current / '.hanzi-update-rollback' / 'new'
    source.mkdir(parents=True)
    (source / executable.name).write_bytes(b'new executable')
    if with_runtime:
        (current / '_internal').mkdir()
        (current / '_internal' / 'version').write_text('old runtime')
        (source / '_internal').mkdir()
        (source / '_internal' / 'version').write_text('new runtime')
    command = replacement_command(source, current, 'windows-x64', 2147483647)
    # Fail after old entries have moved; retain the real retry and rollback logic.
    (source / executable.name).unlink()
    script = base64.b64decode(command[-1]).decode('utf-16-le')
    script = script.replace('Start-Sleep -Seconds 1', 'Start-Sleep -Milliseconds 1')
    command[-1] = base64.b64encode(script.encode('utf-16-le')).decode('ascii')
    result = subprocess.run(command, cwd=current, capture_output=True, timeout=20)
    assert result.returncode != 0
    assert executable.read_bytes() == b'old executable'
    assert (current / 'user-note.txt').read_text() == 'keep'
    if with_runtime:
        assert (current / '_internal' / 'version').read_text() == 'old runtime'
    assert not (source.parent / 'applied').exists()
    assert (source.parent / 'update.log').is_file()
    assert list(tmp_path.iterdir()) == [current]
