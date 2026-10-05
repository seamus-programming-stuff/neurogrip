@echo off
setlocal
if "%~1"=="" (
    echo Usage: setup_neon.cmd CAMERA_IP [SSH_USER]
    exit /b 1
)
set "NEUROGRIP_CAMERA=%~1"
set "NEUROGRIP_USER=%~2"
if "%NEUROGRIP_USER%"=="" set "NEUROGRIP_USER=adlink"
scp "%~dp0neon_raw.py" "%~dp0neon_install.py" "%NEUROGRIP_USER%@%NEUROGRIP_CAMERA%:"
if errorlevel 1 exit /b 1
ssh -t "%NEUROGRIP_USER%@%NEUROGRIP_CAMERA%" python3 -u neon_install.py
exit /b %errorlevel%
