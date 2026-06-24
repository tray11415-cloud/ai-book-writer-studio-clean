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
    """Return the preferred .env path for the current runtime.

    Resolution order:
      1. The runtime directory (next to the executable when frozen, or next to
         this module in source mode) -- the canonical location for bundled apps.
      2. The current working directory, for ad-hoc source runs.

    When no .env exists in either location we return the runtime-dir path: it is
    the canonical bundled location and gives the clearest hint if the file is
    missing, rather than silently pointing at a non-existent cwd file.
    """
    runtime_env = get_runtime_dir() / ".env"
    if runtime_env.exists():
        return str(runtime_env)

    cwd_env = Path(os.getcwd()) / ".env"
    if cwd_env.exists():
        return str(cwd_env)

    return str(runtime_env)
