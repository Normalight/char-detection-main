"""Build the image-only OpenCV used by the macOS arm64 desktop package.

The native extension statically links core/imgproc/imgcodecs and their codecs.
It is imported ahead of the development wheel only during release packaging.
"""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import sysconfig

COMMIT = '31b0eeea0b44b370fd0712312df4214d4ae1b158'  # OpenCV 4.11.0


def prepare(root):
    import cmake
    import numpy
    root = Path(root)
    work = root / 'build' / 'opencv-minimal'
    source = work / 'source'
    build = work / 'build'
    work.mkdir(parents=True, exist_ok=True)
    if not source.exists():
        subprocess.run(['git', 'clone', '--depth', '1', '--branch', '4.11.0',
                        'https://github.com/opencv/opencv.git', str(source)], check=True)
    actual = subprocess.check_output(['git', '-C', str(source), 'rev-parse', 'HEAD'], text=True).strip()
    if actual != COMMIT:
        raise RuntimeError(f'Unexpected OpenCV source revision: {actual}')
    subprocess.run(['git', '-C', str(source), 'diff', '--quiet', 'HEAD'], check=True)
    cmake_bin = str(Path(cmake.CMAKE_BIN_DIR) / 'cmake')
    ninja = Path(sysconfig.get_path('scripts')) / 'ninja'
    if not ninja.is_file():
        raise RuntimeError('Install packaging/requirements-mac.txt in the build environment')
    options = {
        'CMAKE_MAKE_PROGRAM': str(ninja),
        'CMAKE_POLICY_VERSION_MINIMUM': '3.5',
        'CMAKE_BUILD_TYPE': 'Release', 'CMAKE_OSX_DEPLOYMENT_TARGET': '14.0',
        'CMAKE_OSX_ARCHITECTURES': 'arm64',
        'BUILD_LIST': 'core,imgproc,imgcodecs,python3',
        'BUILD_SHARED_LIBS': 'OFF', 'BUILD_TESTS': 'OFF', 'BUILD_PERF_TESTS': 'OFF',
        'BUILD_EXAMPLES': 'OFF', 'BUILD_opencv_apps': 'OFF', 'BUILD_JAVA': 'OFF',
        'WITH_FFMPEG': 'OFF', 'WITH_AVFOUNDATION': 'OFF', 'WITH_GSTREAMER': 'OFF',
        'WITH_QT': 'OFF', 'WITH_OPENEXR': 'OFF', 'WITH_OPENCL': 'OFF', 'WITH_LAPACK': 'OFF',
        'BUILD_ZLIB': 'ON', 'BUILD_PNG': 'ON', 'BUILD_JPEG': 'ON',
        'BUILD_TIFF': 'ON', 'BUILD_WEBP': 'ON', 'BUILD_OPENJPEG': 'ON',
        'PYTHON3_EXECUTABLE': sys.executable,
        'PYTHON3_INCLUDE_DIR': sysconfig.get_path('include'),
        'PYTHON3_INCLUDE_PATH': sysconfig.get_path('include'),
        'PYTHON3_LIBRARY': str(Path(sysconfig.get_config_var('LIBDIR')) / sysconfig.get_config_var('LDLIBRARY')),
        'PYTHON3_NUMPY_INCLUDE_DIRS': numpy.get_include(),
    }
    subprocess.run([cmake_bin, '-S', str(source), '-B', str(build), '-G', 'Ninja',
                    *[f'-D{name}={value}' for name, value in options.items()]], check=True)
    subprocess.run([cmake_bin, '--build', str(build), '--parallel',
                    str(min(6, os.cpu_count() or 1))], check=True)
    modules = list((build / 'lib' / 'python3').glob('cv2*.so'))
    if len(modules) != 1:
        raise RuntimeError('Expected one minimal OpenCV Python extension')
    module_dir = modules[0].parent
    subprocess.run([sys.executable, '-c',
                    "import cv2; assert cv2.__version__ == '4.11.0'; "
                    "assert hasattr(cv2, 'imdecode') and hasattr(cv2, 'warpPerspective'); "
                    "assert not hasattr(cv2, 'VideoCapture')"],
                   env={**os.environ, 'PYTHONPATH': str(module_dir)}, check=True)
    # Retain upstream and third-party license notices with the static build.
    notices = work / 'licenses'
    notices.mkdir(exist_ok=True)
    shutil.copy2(source / 'LICENSE', notices / 'OpenCV-LICENSE')
    for path in (source / '3rdparty').rglob('*'):
        if path.is_file() and (path.name.lower().startswith(('license', 'copying', 'copyright', 'readme'))
                               or path.relative_to(source).as_posix() == '3rdparty/zlib/zlib.h'):
            dest = notices / path.relative_to(source)
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, dest)
    (notices / 'build.json').write_text(json.dumps({
        'source': 'https://github.com/opencv/opencv', 'commit': COMMIT,
        'options': options,
    }, indent=2), encoding='utf-8')
    return str(module_dir), (str(notices), 'licenses/opencv')
