"""Configuration for the book generation system."""
import os
from typing import Dict

from dotenv import load_dotenv
from env_utils import get_dotenv_path


DEFAULT_BASE_URL = "http://127.0.0.1:8000/v1"
DEFAULT_MODEL = "nalang-turbo-0826"


def _normalize_base_url(base_url: str) -> str:
    """Normalize OpenAI-compatible base URLs."""
    normalized = base_url.strip().rstrip("/")
    if not normalized:
        return DEFAULT_BASE_URL
    return normalized


def get_config(local_url: str | None = None) -> Dict:
    """Get the configuration for the agents."""
    load_dotenv(get_dotenv_path())

    base_url = _normalize_base_url(
        local_url
        or os.getenv("OPENAI_BASE_URL")
        or os.getenv("LLM_BASE_URL")
        or DEFAULT_BASE_URL
    )
    model = (
        os.getenv("OPENAI_MODEL")
        or os.getenv("LLM_MODEL")
        or DEFAULT_MODEL
    )
    api_key = (
        os.getenv("OPENAI_API_KEY")
        or os.getenv("LLM_API_KEY")
        or "not-needed"
    )

    config_list = [{
        "model": model,
        "base_url": base_url,
        "api_key": api_key,
    }]

    agent_config = {
        "seed": 42,
        "temperature": 0.7,
        "config_list": config_list,
        "timeout": 600,
        "cache_seed": None,
    }

    return agent_config
