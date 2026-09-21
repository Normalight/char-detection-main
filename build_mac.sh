#!/bin/bash
# Build complete native runtimes, then wrap them in one double-clickable app.
set -euo pipefail
cd "$(dirname "$0")"
ROOT="$PWD"
APP_NAME="中文字符标注工具"
BUNDLE_ID="local.character.annotator"
TARGET="${1:-universal}"
case "$TARGET" in universal) ARCHES=(arm64 x86_64);; arm64|x86_64) ARCHES=("$TARGET");; *) echo "用法: bash build_mac.sh [universal|arm64|x86_64]" >&2; exit 2;; esac
[[ "$(uname -s)" == Darwin ]] || { echo '必须在 macOS 上构建。' >&2; exit 1; }
xcrun --find clang >/dev/null || { echo '请先执行 xcode-select --install' >&2; exit 1; }
# Use actual developer tools instead of /usr/bin xcrun stubs under Rosetta.
export PATH="$(dirname "$(xcrun --find lipo)"):$PATH"
# Framework Python does not always supply an SDK sysroot to extension builds.
export SDKROOT="$(xcrun --sdk macosx --show-sdk-path)"
export CC="$(xcrun --find clang)"
export CXX="$(xcrun --find clang++)"
for CPU in "${ARCHES[@]}"; do
    export ARCHFLAGS="-arch $CPU"
    export PYINSTALLER_CONFIG_DIR="$ROOT/build/mac-cache-$CPU"
    if [[ "$CPU" == arm64 ]]; then PY="${PYTHON_ARM64:-}"; else PY="${PYTHON_X86_64:-}"; fi
    if [[ -z "$PY" ]]; then
        command -v uv >/dev/null || { echo "请安装 uv，或设置 PYTHON_ARM64 / PYTHON_X86_64 为对应架构的 Python 3.11 路径。" >&2; exit 1; }
        UV_CPU="$CPU"; [[ "$CPU" != arm64 ]] || UV_CPU=aarch64
        uv python install "cpython-3.11-macos-${UV_CPU}-none"
        PY="$(uv python find "cpython-3.11-macos-${UV_CPU}-none")"
    fi
    /usr/bin/arch -"$CPU" "$PY" -c "import platform,sys; assert platform.machine() == '$CPU'; assert sys.version_info[:2] == (3,11), 'Use Python 3.11'"
    ENV_DIR="$ROOT/build/mac-env-$CPU"
    if [[ ! -x "$ENV_DIR/bin/python" ]]; then /usr/bin/arch -"$CPU" "$PY" -m venv "$ENV_DIR"; fi
    RUN=(/usr/bin/arch -"$CPU" "$ENV_DIR/bin/python")
    "${RUN[@]}" -c "import platform; assert platform.machine() == '$CPU'"
    "${RUN[@]}" -m pip install --timeout 120 --retries 5 --only-binary=:all: -r packaging/requirements-mac.txt
    MAC_ARCH="$CPU" "${RUN[@]}" -m PyInstaller --noconfirm --clean \
        --distpath "$ROOT/build/mac-dist-$CPU" --workpath "$ROOT/build/mac-work-$CPU" packaging/mac.spec
    # Includes ONNX inference, GUI creation, and group app's frozen worker process.
    "$ROOT/build/mac-dist-$CPU/Runtime.app/Contents/MacOS/runtime" --smoke-test
    "${RUN[@]}" -m pip freeze > "$ROOT/build/mac-dist-$CPU/dependencies.txt"
done
RELEASE="$ROOT/dist/切字-${PACKAGE_VERSION:-v0.0.0-dev}-macos-${TARGET}"
# Only generated output for this exact build target is replaced.
rm -rf "$RELEASE"
APP="$RELEASE/$APP_NAME.app"
mkdir -p "$APP/Contents/MacOS" "$APP/Contents/Resources/runtime"
CLANG_ARCH=()
for CPU in "${ARCHES[@]}"; do
    mkdir -p "$APP/Contents/Resources/runtime/$CPU"
    ditto "$ROOT/build/mac-dist-$CPU/Runtime.app" "$APP/Contents/Resources/runtime/$CPU/Runtime.app"
    CLANG_ARCH+=(-arch "$CPU")
done
xcrun clang "${CLANG_ARCH[@]}" -mmacosx-version-min=14.0 -Os packaging/launcher.c -o "$APP/Contents/MacOS/launcher"
cat > "$APP/Contents/Info.plist" <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
<key>CFBundleExecutable</key><string>launcher</string>
<key>CFBundleName</key><string>$APP_NAME</string>
<key>CFBundleDisplayName</key><string>$APP_NAME</string>
<key>CFBundleIdentifier</key><string>$BUNDLE_ID</string>
<key>CFBundlePackageType</key><string>APPL</string>
<key>CFBundleVersion</key><string>1.0</string>
<key>CFBundleShortVersionString</key><string>1.0</string>
<key>LSMinimumSystemVersion</key><string>14.0</string>
<key>NSHighResolutionCapable</key><true/>
</dict></plist>
PLIST
codesign --force --sign - "$APP"
codesign --verify --deep --strict "$APP"
for CPU in "${ARCHES[@]}"; do
    /usr/bin/arch -"$CPU" "$APP/Contents/MacOS/launcher" --smoke-test
done
cp MAC使用说明.txt "$RELEASE/MAC使用说明.txt"
ZIP="$RELEASE.zip"
rm -f "$ZIP"
ditto -c -k --norsrc --noextattr --noqtn "$RELEASE" "$ZIP"
echo "已生成并验证：$ZIP"
