@echo off
setlocal
cd /d "%~dp0"

echo Installing Novel AI Agent dependencies...
echo Project: %CD%

where py >nul 2>nul
if %errorlevel%==0 (
    py -3.11 -m venv .venv
) else (
    python -m venv .venv
)

if not exist ".venv\Scripts\python.exe" (
    python -m venv .venv
)

if not exist ".venv\Scripts\python.exe" (
    echo Failed to create .venv. Please install Python 3.11 or newer.
    pause
    exit /b 1
)

".venv\Scripts\python.exe" -m pip install --upgrade pip
".venv\Scripts\python.exe" -m pip install -r requirements.txt

echo.
echo Install complete.
echo You can now run run_desktop.bat.
pause
