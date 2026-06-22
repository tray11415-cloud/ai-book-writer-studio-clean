@echo off
setlocal
set PYTHONIOENCODING=utf-8

cd /d "%~dp0"
set "VENV_DIR=%CD%\.venv"
set "VENV_PYTHON=%VENV_DIR%\Scripts\python.exe"

if not exist "%VENV_DIR%" (
    echo [1/5] Creating virtual environment...
    py -m venv "%VENV_DIR%"
    if errorlevel 1 goto :error
)

echo [2/5] Activating virtual environment...
call "%VENV_DIR%\Scripts\activate.bat"
if errorlevel 1 goto :error
set "VIRTUAL_ENV=%VENV_DIR%"
set "PATH=%VENV_DIR%\Scripts;%PATH%"

echo [3/5] Installing dependencies...
"%VENV_PYTHON%" -m pip install --upgrade pip
"%VENV_PYTHON%" -m pip install -r requirements.txt
if errorlevel 1 goto :error

if not exist "book_output" mkdir "book_output"

echo [4/5] Starting compatibility proxy...
start "AI Book Writer Proxy" /B "%VENV_PYTHON%" compat_proxy.py
timeout /t 2 >nul

echo [5/5] Starting generator...
"%VENV_PYTHON%" main.py
if errorlevel 1 goto :error

echo.
echo Finished.
pause
exit /b 0

:error
echo.
echo Startup failed. Please check the messages above.
pause
exit /b 1
