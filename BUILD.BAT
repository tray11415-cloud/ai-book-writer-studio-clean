@echo off
REM ===========================================================================
REM  AI Book Writer Studio - One-click build & launch
REM  Installs all dependencies into a local .venv, then starts the Gradio studio.
REM  Just double-click this file. Re-running is safe (it reuses the venv).
REM ===========================================================================
setlocal
chcp 65001 >nul
set PYTHONIOENCODING=utf-8

cd /d "%~dp0"
set "VENV_DIR=%CD%\.venv"
set "VENV_PYTHON=%VENV_DIR%\Scripts\python.exe"

echo ===========================================================================
echo  AI Book Writer Studio - Setup
echo ===========================================================================
echo.

REM --- Locate a Python interpreter (prefer the py launcher) ----------------
set "PYLAUNCHER="
where py >nul 2>nul && set "PYLAUNCHER=py"
if not defined PYLAUNCHER (
    where python >nul 2>nul && set "PYLAUNCHER=python"
)
if not defined PYLAUNCHER (
    echo [ERROR] Python was not found on this machine.
    echo         Please install Python 3.10+ from https://www.python.org/downloads/
    echo         and tick "Add Python to PATH" during installation.
    goto :error
)

REM --- Create the virtual environment if missing --------------------------
if not exist "%VENV_PYTHON%" (
    echo [1/4] Creating virtual environment...
    %PYLAUNCHER% -m venv "%VENV_DIR%"
    if errorlevel 1 goto :error
) else (
    echo [1/4] Virtual environment already exists. Reusing it.
)

REM --- Activate the venv --------------------------------------------------
echo [2/4] Activating virtual environment...
call "%VENV_DIR%\Scripts\activate.bat"
if errorlevel 1 goto :error
set "VIRTUAL_ENV=%VENV_DIR%"
set "PATH=%VENV_DIR%\Scripts;%PATH%"

REM --- Install dependencies ----------------------------------------------
echo [3/4] Installing dependencies (this may take a few minutes the first time)...
"%VENV_PYTHON%" -m pip install --upgrade pip
"%VENV_PYTHON%" -m pip install -r requirements.txt
if errorlevel 1 goto :error

REM --- Make sure a .env exists (copy from the example if needed) ----------
if not exist ".env" (
    if exist ".env.example" (
        echo.
        echo [NOTE] No .env found. Creating one from .env.example.
        echo        Open .env and fill in your API keys before generating text.
        copy /y ".env.example" ".env" >nul
    )
)

if not exist "book_output" mkdir "book_output"

REM --- Launch the studio -------------------------------------------------
echo [4/4] Starting AI Book Writer Studio...
echo.
"%VENV_PYTHON%" app_gradio.py
if errorlevel 1 goto :error

exit /b 0

:error
echo.
echo ===========================================================================
echo  Setup or startup failed. Please check the messages above.
echo ===========================================================================
pause
exit /b 1
