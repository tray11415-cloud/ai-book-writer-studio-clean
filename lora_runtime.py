"""Helpers for running the local Qwen LoRA server."""
from __future__ import annotations

import os
import socket
import subprocess
import sys
import time
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
LORA_HOST = os.getenv("LORA_HOST", "127.0.0.1")
LORA_PORT = int(os.getenv("LORA_PORT", "8010"))
LORA_BASE_URL = f"http://{LORA_HOST}:{LORA_PORT}/v1"
LORA_MODEL_NAME = os.getenv("LORA_MODEL_NAME", "qwen3-4b-novel-lora")
LORA_ADAPTER_DIR = PROJECT_ROOT / "lora_training" / "lora_output" / "qwen3_4b_novel_lora"
LORA_SERVER_SCRIPT = PROJECT_ROOT / "lora_training" / "serve_lora_openai.py"
LORA_VENV_PYTHON = PROJECT_ROOT / "lora_training" / ".venv" / "Scripts" / "python.exe"


def _normalize_url(url: str) -> str:
    return (url or "").strip().rstrip("/")


def is_lora_base_url(base_url: str) -> bool:
    return _normalize_url(base_url).lower() == LORA_BASE_URL.lower()


def is_port_open(host: str = LORA_HOST, port: int = LORA_PORT, timeout: float = 0.5) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(timeout)
        return sock.connect_ex((host, port)) == 0


def ensure_lora_server_running(wait_seconds: float = 2.0) -> None:
    if is_port_open():
        return

    if not LORA_VENV_PYTHON.exists():
        raise RuntimeError(f"LoRA Python environment not found: {LORA_VENV_PYTHON}")
    if not LORA_SERVER_SCRIPT.exists():
        raise RuntimeError(f"LoRA server script not found: {LORA_SERVER_SCRIPT}")
    if not (LORA_ADAPTER_DIR / "adapter_model.safetensors").exists():
        raise RuntimeError(f"LoRA adapter not found: {LORA_ADAPTER_DIR}")

    log_dir = PROJECT_ROOT / "lora_training" / "runs" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "lora_openai_server.log"

    creationflags = subprocess.CREATE_NO_WINDOW if sys.platform.startswith("win") else 0
    print(f"Starting LoRA server; logs at {log_path}")
    # Open the log via a context manager so the parent's handle is always closed.
    # Popen duplicates the underlying OS file descriptor for the child, so the
    # subprocess keeps writing to the log after the parent closes its reference.
    with log_path.open("ab") as log_handle:
        process = subprocess.Popen(
            [
                str(LORA_VENV_PYTHON),
                str(LORA_SERVER_SCRIPT),
                "--host",
                LORA_HOST,
                "--port",
                str(LORA_PORT),
                "--adapter",
                str(LORA_ADAPTER_DIR),
                "--model-name",
                LORA_MODEL_NAME,
            ],
            cwd=str(PROJECT_ROOT),
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            creationflags=creationflags,
        )

    deadline = time.time() + wait_seconds
    while time.time() < deadline:
        if is_port_open(timeout=0.2):
            return
        # If the subprocess has already exited, surface its log tail instead of
        # waiting out the full deadline and silently misconfiguring the app.
        if process.poll() is not None:
            break
        time.sleep(0.2)

    if not is_port_open():
        tail = ""
        try:
            tail = log_path.read_text(encoding="utf-8", errors="replace")[-1000:]
        except OSError:
            pass
        exit_info = (
            f" (process exited with code {process.returncode})"
            if process.poll() is not None
            else ""
        )
        raise RuntimeError(
            f"LoRA server failed to start within {wait_seconds}s on "
            f"{LORA_HOST}:{LORA_PORT}{exit_info}. See {log_path}."
            + (f"\nLast logs:\n{tail}" if tail else "")
        )
