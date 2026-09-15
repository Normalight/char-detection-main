#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")"
PY="${PYTHON:-python3}"
"$PY" -c 'import sys; assert (3,10) <= sys.version_info[:2] <= (3,11), "请设置 PYTHON 为 Python 3.10/3.11 路径"'
exec "$PY" app.py "$@"
