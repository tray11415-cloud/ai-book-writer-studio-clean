"""Windows EXE launcher for AI Book Writer Studio."""
from __future__ import annotations

import ctypes
import os
import socket
import subprocess
import sys
import time
import webbrowser
from pathlib import Path


APP_TITLE = "AI Book Writer Studio"
PORT_START = 7860
PORT_END = 7899


def app_root() -> Path:
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent


def show_message(text: str) -> None:
    try:
        ctypes.windll.user32.MessageBoxW(None, text, APP_TITLE, 0x40)
    except Exception:
        print(text)


def can_bind(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind(("127.0.0.1", port))
        except OSError:
            return False
    return True


def find_open_port() -> int:
    for port in range(PORT_START, PORT_END + 1):
        if can_bind(port):
            return port
    raise RuntimeError(f"No open Gradio port found in {PORT_START}-{PORT_END}.")


def wait_for_port(port: int, timeout_seconds: int = 180) -> bool:
    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.settimeout(1)
            if sock.connect_ex(("127.0.0.1", port)) == 0:
                return True
        time.sleep(1)
    return False


def launch_studio() -> int:
    root = app_root()
    start_bat = root / "start_studio.bat"
    if not start_bat.exists():
        show_message(f"Cannot find start_studio.bat beside the launcher:\n{root}")
        return 1

    port = find_open_port()
    url = f"http://127.0.0.1:{port}/"
    env = os.environ.copy()
    env["GRADIO_SERVER_PORT"] = str(port)
    env.setdefault("PYTHONIOENCODING", "utf-8")

    subprocess.Popen(
        ["cmd.exe", "/k", str(start_bat)],
        cwd=str(root),
        env=env,
        creationflags=subprocess.CREATE_NEW_CONSOLE,
    )

    if wait_for_port(port):
        webbrowser.open(url)
        return 0

    show_message(
        "Studio startup was launched, but the local web server did not answer in time.\n"
        f"Check the console window, then open this URL manually:\n{url}"
    )
    return 2


def main() -> int:
    if "--check" in sys.argv:
        root = app_root()
        start_bat = root / "start_studio.bat"
        try:
            port = find_open_port()
        except Exception as exc:
            print(f"[ERROR] {exc}")
            return 1
        print("[OK] Launcher check passed.")
        print(f"Root: {root}")
        print(f"start_studio.bat: {start_bat.exists()}")
        print(f"Next open port: {port}")
        return 0 if start_bat.exists() else 1

    try:
        return launch_studio()
    except Exception as exc:
        show_message(f"Launcher failed:\n{exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
