@echo off
setlocal
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    echo Local environment not found. Running install_windows.bat first...
    call install_windows.bat
)

if not exist ".venv\Scripts\python.exe" (
    echo Cannot start API because .venv was not created.
    pause
    exit /b 1
)

echo Starting local API...
echo Docs: http://127.0.0.1:8000/docs
".venv\Scripts\python.exe" -m uvicorn api_server:app --host 127.0.0.1 --port 8000
pause
