@echo off
setlocal
powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\launch-arm.ps1" -SafeMode %*
exit /b %ERRORLEVEL%
