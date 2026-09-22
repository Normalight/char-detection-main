"""Optional save accelerators. Invalid/missing caches always trigger regeneration."""
import json
import os
import uuid


def file_stamp(path):
    try:
        stat = os.stat(path)
        return [stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns]
    except (OSError, TypeError):
        return None


def read_cache(path):
    try:
        with open(path, encoding='utf-8') as stream:
            value = json.load(stream)
        return value if isinstance(value, dict) else {}
    except (OSError, ValueError):
        return {}


def write_cache(path, value):
    temporary = str(path) + '.tmp-' + uuid.uuid4().hex
    try:
        with open(temporary, 'w', encoding='utf-8') as stream:
            json.dump(value, stream)
        os.replace(temporary, path)
    except OSError:
        pass
    finally:
        try:
            os.unlink(temporary)
        except OSError:
            pass
