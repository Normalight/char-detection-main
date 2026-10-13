#!/bin/bash
# Build the selected native macOS runtime and a double-clickable application.
set -euo pipefail
cd "$(dirname "$0")"
ROOT="$PWD"
APP_NAME="中文字符标注工具"
BUNDLE_ID="local.character.annotator"
PRODUCT="${PRODUCT:-char-detection-main}"
TARGET="${1:-arm64}"
[[ ( "$TARGET" == arm64 || "$TARGET" == x86_64 ) && "$#" -le 1 ]] || { echo "用法: bash build_mac.sh [arm64|x86_64]" >&2; exit 2; }
ARCHES=("$TARGET")
[[ "$(uname -s)" == Darwin ]] || { echo '必须在 macOS 上构建。' >&2; exit 1; }
xcrun --find clang >/dev/null || { echo '请先执行 xcode-select --install' >&2; exit 1; }
# Use the selected Xcode developer tools.
export PATH="$(dirname "$(xcrun --find lipo)"):$PATH"
# Framework Python does not always supply an SDK sysroot to extension builds.
export SDKROOT="$(xcrun --sdk macosx --show-sdk-path)"
export CC="$(xcrun --find clang)"
export CXX="$(xcrun --find clang++)"
for CPU in "${ARCHES[@]}"; do
    export ARCHFLAGS="-arch $CPU"
    export PYINSTALLER_CONFIG_DIR="$ROOT/build/mac-cache-$CPU"
    if [[ "$CPU" == arm64 ]]; then
        PY="${PYTHON_ARM64:-}"
        UV_CPU=aarch64
    else
        PY="${PYTHON_X86_64:-}"
        UV_CPU=x86_64
    fi
    if [[ -z "$PY" ]]; then
        command -v uv >/dev/null || { echo "请安装 uv，或设置对应架构的 PYTHON_ARM64 / PYTHON_X86_64 为 Python 3.11 路径。" >&2; exit 1; }
        uv python install "cpython-3.11-macos-${UV_CPU}-none"
        PY="$(uv python find "cpython-3.11-macos-${UV_CPU}-none")"
    fi
    /usr/bin/arch -"$CPU" "$PY" -c "import platform,sys; assert platform.machine() == '$CPU'; assert sys.version_info[:2] == (3,11), 'Use Python 3.11'"
    ENV_DIR="$ROOT/build/mac-env-$CPU"
    if [[ ! -x "$ENV_DIR/bin/python" ]]; then /usr/bin/arch -"$CPU" "$PY" -m venv "$ENV_DIR"; fi
    RUN=(/usr/bin/arch -"$CPU" "$ENV_DIR/bin/python")
    "${RUN[@]}" -c "import platform; assert platform.machine() == '$CPU'"
    "${RUN[@]}" -m pip install --timeout 120 --retries 5 --only-binary=:all: -r packaging/requirements-mac.txt
    export MAC_ARCH="$CPU"
    "${RUN[@]}" -m PyInstaller --noconfirm --clean \
        --distpath "$ROOT/build/mac-dist-$CPU" --workpath "$ROOT/build/mac-work-$CPU" packaging/mac.spec
    # Includes ONNX inference, GUI creation, and group app's frozen worker process.
    "$ROOT/build/mac-dist-$CPU/Runtime.app/Contents/MacOS/runtime" --smoke-test
    "${RUN[@]}" -m pip freeze > "$ROOT/build/mac-dist-$CPU/dependencies.txt"
done
RELEASE="$ROOT/dist/${PRODUCT}-${PACKAGE_VERSION:-v0.0.0-dev}-macos-${TARGET}"
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
"${RUN[@]}" packaging/native_release.py macos --app "$APP"
if [[ -z "${HANZI_UPDATE_FEED:-}" ]]; then
    codesign --force --sign - "$APP"
fi
codesign --verify --deep --strict "$APP"
for CPU in "${ARCHES[@]}"; do
    /usr/bin/arch -"$CPU" "$APP/Contents/MacOS/launcher" --smoke-test
done
cp MAC使用说明.txt "$RELEASE/MAC使用说明.txt"
# The signed updater archive shares the <product>-<version>-<target> name and is written by
# native_release.py; zipping here again would replace it and leave two same-platform ZIPs in dist.
if [[ -z "${HANZI_UPDATE_FEED:-}" ]]; then
    ZIP="$RELEASE.zip"
    rm -f "$ZIP"
    ditto -c -k --norsrc --noextattr --noqtn "$RELEASE" "$ZIP"
    echo "已生成并验证：$ZIP"
else
    echo "原生签名更新包：$RELEASE.zip"
fi
