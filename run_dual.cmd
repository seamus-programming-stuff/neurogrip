@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
    echo Missing PC environment. Run: py -m venv .venv
    echo Then: .venv\Scripts\python.exe -m pip install -r requirements.txt
    exit /b 1
)
if not exist "dual_config.json" (
    ".venv\Scripts\python.exe" dual_calibrate.py
) else (
    ".venv\Scripts\python.exe" dual_camera.py --config dual_config.json
)
exit /b %errorlevel%
