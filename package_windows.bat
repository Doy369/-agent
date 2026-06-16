@echo off
setlocal
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    echo Local environment not found. Running install_windows.bat first...
    call install_windows.bat
)

if not exist ".venv\Scripts\python.exe" (
    echo Cannot package because .venv was not created.
    pause
    exit /b 1
)

".venv\Scripts\python.exe" -m PyInstaller --noconfirm --clean --windowed --onefile --name NovelAIAgent --add-data "knowledge.txt;." main.py

echo.
echo Local package created:
echo %CD%\dist\NovelAIAgent.exe
pause
