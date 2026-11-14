@echo off
setlocal
pushd "%~dp0"
if errorlevel 1 (
    echo Could not open the GAID project directory.
    pause
    exit /b 1
)

if not exist ".venv\Scripts\gaid.exe" (
    echo GAID was not found in .venv\Scripts. See docs\development.md for setup.
    popd
    pause
    exit /b 1
)

".venv\Scripts\gaid.exe" ui %*
set "GAID_EXIT=%ERRORLEVEL%"
popd
if not "%GAID_EXIT%"=="0" (
    echo GAID exited with code %GAID_EXIT%.
    pause
)
exit /b %GAID_EXIT%
