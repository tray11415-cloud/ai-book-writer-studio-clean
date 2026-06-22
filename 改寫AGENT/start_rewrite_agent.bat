@echo off
setlocal
rem Pure ASCII launcher. %~dp0 is this .bat's own folder (the 改寫AGENT folder),
rem expanded by cmd at runtime, so the Chinese folder name never appears as
rem literal bytes here. That avoids the cp950/UTF-8 batch-encoding corruption
rem (and the chcp 65001 byte-misalignment bug) that broke the old launcher.

set "VENV_PY=%~dp0..\.venv\Scripts\python.exe"

if exist "%VENV_PY%" (
    "%VENV_PY%" "%~dp0app.py"
) else (
    echo [warn] .venv not found, falling back to system python.
    python "%~dp0app.py"
)

pause
