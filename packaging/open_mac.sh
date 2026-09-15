#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")"
shopt -s nullglob
apps=( *.app )
[[ ${#apps[@]} -eq 1 ]] || { echo '请将本脚本与发布包中的 .app 放在同一文件夹。' >&2; exit 1; }
exec /usr/bin/open "${apps[0]}"
