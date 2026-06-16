@echo off
setlocal

pushd "%~dp0" >nul
if errorlevel 1 (
    echo ERROR: Could not enter the LSDetector project folder.
    pause
    exit /b 1
)

powershell -NoProfile -ExecutionPolicy Bypass -File ".\tools\start_lsd_app.ps1"
set "EXIT_CODE=%ERRORLEVEL%"

popd >nul
echo.
echo [LSDetector] Press any key to close this window.
pause >nul
exit /b %EXIT_CODE%
