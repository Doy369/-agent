@echo off
setlocal
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    echo Local environment not found. Running install_windows.bat first...
    call install_windows.bat
)

if not exist ".venv\Scripts\python.exe" (
    echo Cannot start app because .venv was not created.
    pause
    exit /b 1
)

".venv\Scripts\python.exe" main.py
