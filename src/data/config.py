"""Tiny config / secrets loader.

Reads API keys from the environment or a local ``.env`` file. No third-party
library needed (so no install). Keys are NEVER printed or committed.

Security posture (security-lab/AGENT_RULES.md):
  * Keys live only in your local ``.env`` (git-ignored).
  * This module never logs key values -- only whether a key is present.
  * The agent does NOT create accounts or obtain keys for you; you sign up and
    paste the key in. See security-lab and DATA_SIGNUP.md.
"""

from __future__ import annotations

import os

# Logical key name -> environment variable.
KEY_ENV = {
    "football_data": "FOOTBALL_DATA_API_KEY",
    "odds_api": "ODDS_API_KEY",
}


def _load_dotenv(path: str = ".env") -> None:
    """Load KEY=VALUE lines from a local .env into os.environ (if present)."""
    if not os.path.exists(path):
        return
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def get_key(name: str) -> str | None:
    """Return the API key for ``name`` (e.g. 'football_data'), or None."""
    _load_dotenv()
    env = KEY_ENV.get(name)
    return os.environ.get(env) if env else None


def have_key(name: str) -> bool:
    return bool(get_key(name))


def key_status() -> dict[str, bool]:
    """Which keys are present? (booleans only -- never the values.)"""
    return {name: have_key(name) for name in KEY_ENV}
