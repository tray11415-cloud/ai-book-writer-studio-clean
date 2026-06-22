"""Helpers for loading runtime environment files in source or bundled mode."""
from __future__ import annotations

import os
import sys
from pathlib import Path


def get_runtime_dir() -> Path:
    """Return the directory that should contain runtime assets like .env."""
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent


def get_dotenv_path() -> str:
    """Return the preferred .env path for the current runtime."""
    runtime_env = get_runtime_dir() / ".env"
    if runtime_env.exists():
        return str(runtime_env)
    return str(Path(os.getcwd()) / ".env")
