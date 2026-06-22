@echo off
setlocal
set PYTHONIOENCODING=utf-8

cd /d "%~dp0"
set "VENV_DIR=%CD%\lora_training\.venv"
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

echo [3/5] Installing LoRA training dependencies...
"%VENV_PYTHON%" -m pip install --upgrade pip
"%VENV_PYTHON%" -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
"%VENV_PYTHON%" -m pip install -r lora_training\requirements-lora.txt
if errorlevel 1 goto :error

echo [4/5] Checking training data...
if not exist "lora_training\data\train.jsonl" (
    if not exist "lora_training\source_texts" mkdir "lora_training\source_texts"
    "%VENV_PYTHON%" lora_training\prepare_corpus.py --source "%CD%" --out "%CD%\lora_training\data"
    if errorlevel 1 goto :error
) else (
    echo Existing lora_training\data\train.jsonl found. Skipping data preparation.
)

echo [5/5] Training Qwen LoRA...
"%VENV_PYTHON%" lora_training\train_qwen_lora.py --model Qwen/Qwen3-4B --train-file lora_training\data\train.jsonl --output-dir lora_training\lora_output\qwen3_4b_novel_lora --max-length 2048 --epochs 2 --grad-accum 8 --batch-size 1 --target-loss 2.7 --target-loss-patience 3 --target-loss-min-steps 500 --save-steps 250
if errorlevel 1 goto :error

echo.
echo Done. LoRA adapter saved under lora_training\lora_output\qwen3_4b_novel_lora
pause
exit /b 0

:error
echo.
echo Qwen LoRA training failed. Please check the messages above.
pause
exit /b 1
