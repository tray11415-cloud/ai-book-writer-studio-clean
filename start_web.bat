@echo off
setlocal
set PYTHONIOENCODING=utf-8

cd /d "%~dp0"
set "VENV_DIR=%CD%\.venv"
set "VENV_PYTHON=%VENV_DIR%\Scripts\python.exe"

if not exist "%VENV_DIR%" (
    echo [1/4] Creating virtual environment...
    py -m venv "%VENV_DIR%"
    if errorlevel 1 goto :error
)

echo [2/4] Activating virtual environment...
call "%VENV_DIR%\Scripts\activate.bat"
if errorlevel 1 goto :error
set "VIRTUAL_ENV=%VENV_DIR%"
set "PATH=%VENV_DIR%\Scripts;%PATH%"

echo [3/4] Installing dependencies...
"%VENV_PYTHON%" -m pip install --upgrade pip
"%VENV_PYTHON%" -m pip install -r requirements.txt
if errorlevel 1 goto :error

echo [4/4] Starting web app...
start "" http://127.0.0.1:5000/
"%VENV_PYTHON%" web_app.py
if errorlevel 1 goto :error

exit /b 0

:error
echo.
echo Web app startup failed. Please check the messages above.
pause
exit /b 1
