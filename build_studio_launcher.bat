@echo off
setlocal
set PYTHONIOENCODING=utf-8

cd /d "%~dp0"
set "VENV_PYTHON=%CD%\.venv\Scripts\python.exe"

if not exist "%VENV_PYTHON%" (
    echo Missing virtual environment Python: %VENV_PYTHON%
    echo Run start_studio.bat once first, then rebuild the launcher.
    pause
    exit /b 1
)

"%VENV_PYTHON%" -m pip install pyinstaller
if errorlevel 1 goto :error

"%VENV_PYTHON%" -m PyInstaller --onefile --noconsole --name StartAIBookWriterStudio studio_launcher.py
if errorlevel 1 goto :error

copy /Y "dist\StartAIBookWriterStudio.exe" "StartAIBookWriterStudio.exe" >nul
if errorlevel 1 goto :error

echo.
echo Built: %CD%\StartAIBookWriterStudio.exe
exit /b 0

:error
echo.
echo Launcher build failed.
pause
exit /b 1
