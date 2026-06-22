@echo off
setlocal
set PYTHONIOENCODING=utf-8

cd /d "%~dp0"
set "VENV_DIR=%CD%\lora_training\.venv"
set "VENV_PYTHON=%VENV_DIR%\Scripts\python.exe"

if not exist "%VENV_PYTHON%" (
    echo LoRA virtual environment not found: %VENV_PYTHON%
    echo Run train_lora_qwen.bat once, or reinstall lora_training\requirements-lora.txt.
    pause
    exit /b 1
)

if not exist "lora_training\lora_output\qwen3_4b_novel_lora\adapter_model.safetensors" (
    echo LoRA adapter not found: lora_training\lora_output\qwen3_4b_novel_lora
    pause
    exit /b 1
)

echo Starting local Qwen LoRA OpenAI-compatible server...
echo URL: http://127.0.0.1:8010/v1
echo Model: qwen3-4b-novel-lora
"%VENV_PYTHON%" lora_training\serve_lora_openai.py --host 127.0.0.1 --port 8010 --adapter "%CD%\lora_training\lora_output\qwen3_4b_novel_lora" --model-name qwen3-4b-novel-lora
pause
