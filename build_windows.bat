@echo off
setlocal EnableExtensions

rem Build the Windows x64 onedir distribution from the conda chardet env.
rem Run this script from a checkout that already contains the model files.

cd /d "%~dp0"
set "CONDA_ENV=%CHAR_DETECTION_CONDA_ENV%"
if not defined CONDA_ENV set "CONDA_ENV=chardet"

where conda >nul 2>&1
if errorlevel 1 (
    echo [ERROR] conda was not found on PATH.
    echo         Install Miniconda/Anaconda and run this from an activated shell.
    exit /b 1
)

for %%F in (
    "models\ppocrv6\small\inference.onnx"
    "models\ppocrv6\small\inference.yml"
    "models\ppocrv6_rec\small\inference.onnx"
    "models\ppocrv6_rec\small\inference.yml"
) do (
    if not exist "%%~F" (
        echo [ERROR] Missing model file: %%~F
        echo         Download/copy the models before building.
        exit /b 1
    )
)

if not exist "models\dbnet.onnx" echo [WARN] Legacy det model not found; v6-small remains usable.
if not exist "models\crnn_lite_lstm.onnx" echo [WARN] Legacy rec model not found; v6-small remains usable.

call conda run --no-capture-output -n "%CONDA_ENV%" python -c "import PyInstaller" >nul 2>&1
if errorlevel 1 (
    echo [ERROR] PyInstaller is not installed in conda env %CONDA_ENV%.
    echo         Install it with: conda run -n %CONDA_ENV% python -m pip install pyinstaller
    exit /b 1
)

echo [INFO] Building interactive_annotator onedir distribution ...
if exist "dist\interactive_annotator.exe" (
    echo [INFO] Removing obsolete one-file executable ...
    del /q "dist\interactive_annotator.exe"
    if exist "dist\interactive_annotator.exe" (
        echo [ERROR] The obsolete one-file executable is still in use.
        echo         Close interactive_annotator.exe and run this script again.
        exit /b 1
    )
)
call conda run --no-capture-output -n "%CONDA_ENV%" python -m PyInstaller --noconfirm --clean app.spec
if errorlevel 1 (
    echo [ERROR] PyInstaller failed.
    exit /b 1
)

if not exist "dist\interactive_annotator\interactive_annotator.exe" (
    echo [ERROR] Build reported success but the onedir executable was not found.
    exit /b 1
)

echo.
echo [OK] Created: %CD%\dist\interactive_annotator\interactive_annotator.exe
echo [INFO] Distribute the entire dist\interactive_annotator directory; do not copy only the exe.
exit /b 0
