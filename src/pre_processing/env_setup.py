from __future__ import annotations

import getpass
import os
from pathlib import Path

from dotenv import load_dotenv

_ENV_INITIALISED = False


def _load_dotenv_once() -> None:
    global _ENV_INITIALISED
    if _ENV_INITIALISED:
        return
    env_path = Path(__file__).parent.parent.parent / ".env"
    if env_path.exists():
        load_dotenv(env_path)
    _ENV_INITIALISED = True


def ensure_openai_api_key(prompt: str = "openai api") -> str:
    _load_dotenv_once()
    key = os.getenv("OPENAI_API_KEY") or ""
    key = key.strip()
    if not key:
        key = getpass.getpass(prompt).strip()
    os.environ["OPENAI_API_KEY"] = key
    return key

