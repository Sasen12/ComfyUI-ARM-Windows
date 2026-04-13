@echo off
setlocal
powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\launch-arm.ps1" -Runtime QNN %*
exit /b %ERRORLEVEL%
