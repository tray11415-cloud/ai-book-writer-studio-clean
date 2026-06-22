@echo off
setlocal
set PYTHONIOENCODING=utf-8

cd /d "%~dp0"
set "VENV_DIR=%CD%\lora_training\.venv"
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

echo [3/4] Installing text conversion dependency...
"%VENV_PYTHON%" -m pip install opencc-python-reimplemented
if errorlevel 1 goto :error

echo [4/4] Preparing LoRA training data...
if not exist "lora_training\source_texts" mkdir "lora_training\source_texts"
"%VENV_PYTHON%" lora_training\prepare_corpus.py --source "%CD%" --out "%CD%\lora_training\data"
if errorlevel 1 goto :error

echo.
echo Done. Check lora_training\data\dataset_report.json
pause
exit /b 0

:error
echo.
echo LoRA data preparation failed. Please check the messages above.
pause
exit /b 1
