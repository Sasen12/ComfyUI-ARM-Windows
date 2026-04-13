@echo off
setlocal
powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\launch-arm.ps1" -Runtime QNN -SafeMode %*
exit /b %ERRORLEVEL%
